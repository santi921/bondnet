import pandas as pd
import numpy as np
import networkx as nx
import itertools
import torch
from rdkit import Chem
import itertools, copy, dgl


def get_dataset_species(molecules):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        molecules (list): rdkit molecules

    Returns:
        list: a sequence of species string
    """
    system_species = set()
    for mol in molecules:
        if mol is None:
            continue
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        system_species.update(species)

    return sorted(system_species)


def get_dataset_species_from_json(json_file):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        json_file: file containing reaction w/ column

    Returns:
        list: a sequence of species string
    """
    df = pd.read_json(json_file)

    list(df["composition"][0].keys())

    system_species = set()
    for _, row in df.iterrows():
        if row is None:
            continue
        species = list(row["composition"].keys())
        system_species.update(species)

    return sorted(system_species)


def get_hydro_data_functional_groups(json_file):
    """
    Get all the unique functional groups of the hydrolysis reactions compiled in the dataset

    Args:
        json_file: hydrolysis dataset containing the reactions

    Returns:
        list: a sequence of all the unique reacted functional groups in the dataset
    """
    df = pd.read_json(json_file)
    return sorted(list(df["functional_group_reacted"].unique()))


def one_hot_encoding(x, allowable_set):
    """One-hot encoding.

    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.

    Returns
    -------
    list
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1, then we must have x == allowable_set[i].
    """
    return list(map(int, list(map(lambda s: x == s, allowable_set))))


def multi_hot_encoding(x, allowable_set):
    """Multi-hot encoding.

    Args:
        x (list): any type that can be compared with elements in allowable_set
        allowable_set (list): allowed values for x to take

    Returns:
        list: List of int (0 or 1) where zero or more values can be 1.
            If the i-th value is 1, then we must have allowable_set[i] in x.
    """
    return list(map(int, list(map(lambda s: s in x, allowable_set))))


def h_count_and_degree(atom_ind, bond_list, species_order):
    """
    gets the number of H-atoms connected to an atom + degree of bonding
    takes:
        atom_ind(int): index of atom
        bond_list(list of lists): list of bonds in graph
        species_order: order of atoms in graph to match nodes
    """
    # h count
    atom_bonds = []
    h_count = 0
    for i in bond_list:
        if atom_ind in i:
            atom_bonds.append(i)
    if atom_bonds != 0:
        for bond in atom_bonds:
            bond_copy = bond[:]
            bond_copy.remove(atom_ind)
            if species_order[bond_copy[0]] == "H":
                h_count += 1
    return h_count, int(len(atom_bonds))


def ring_features_from_atom(atom_ind, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        atom_ind(int) - an atom's index
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
    returns:
        ring_inclusion - int of whether atom is in a ring
        ring_size_ret_list - one-hot list of whether

    """
    ring_inclusion = 0
    ring_size = 0  # find largest allowable ring that this atom is a part o
    ring_size_ret_list = [0 for i in allowed_ring_size]

    if cycles != []:
        min_ring = 100
        for i in cycles:
            if atom_ind in i:
                if len(i) in allowed_ring_size and len(i) < min_ring:
                    ring_inclusion = 1
                    ring_size = int(len(i))
        if min_ring < 100:
            ring_size = min_ring

    # one hot encode the detected ring size
    if ring_size != 0:
        ring_size_ret_list[allowed_ring_size.index(ring_size)] = 1

    return ring_inclusion, ring_size_ret_list


def ring_features_from_atom_full(atom_num, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        atom_num: number of atoms in molecule
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
    returns:
        ret_dict - dictionary of ring information w/keys as atom inds

    """
    ret_dict = {}
    for i in range(atom_num):
        ret_dict[i] = ring_features_from_atom(i, cycles, allowed_ring_size)
    return ret_dict


def ring_features_from_bond(bond, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        bond_ind(int) - bond's index
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
    returns:
        ring_inclusion - int of whether atom is in a ring
        ring_size_ret_list - one-hot list of whether
    """
    ring_inclusion = 0
    ring_size_ret_list = [0 for i in range(len(allowed_ring_size))]

    for cycle in cycles:
        if tuple(bond) in cycle or (bond[-1], bond[0]) in cycle:
            ring_inclusion = 1
            ring_size_ret_list[allowed_ring_size.index(len(cycle))] = 1

    return ring_inclusion, ring_size_ret_list


def ring_features_for_bonds_full(bonds, no_metal_binary, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        bonds - list of bonds with metal bonds included
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
        non_metal_binary - array with one-hot encoding of whether bonds are metal/not
    returns:
        ret_dict - dictionary with bond(formatted in root-to-target) with metal-bond binary,
        ring inclusion and ring_one_hot

    """
    ret_dict = {}
    for i, bond in enumerate(bonds):
        if no_metal_binary[i] == 1:
            inclusion, ring_one_hot = ring_features_from_bond(
                bond, cycles, allowed_ring_size
            )
            ret_dict[tuple(bond)] = (
                0,
                inclusion,
                ring_one_hot,
            )
        else:  # we're never including metal bonds in ring formations
            ret_dict[tuple(bond)] = (1, 0, [0 for i in range(len(allowed_ring_size))])
    return ret_dict


def organize_list(cycle_list):
    """
    a helper function to orient cycles identically. Finds max value then adds values in the direction where the
    neighbor node's index is greater
    takes:
        cycle_list - arbitrarily defined cycles
    returns:
        cycle_org - consistently organized cycles
    """

    cycle_org = []
    for cycle in cycle_list:
        new_cycle = []
        cycle_len = int(len(cycle))
        new_cycle_start = np.argmax(cycle)
        new_cycle_direction = 1
        plus_1 = new_cycle_start + 1
        if plus_1 > cycle_len - 1:
            plus_1 -= cycle_len
        if int(cycle[new_cycle_start - 1]) > int(cycle[plus_1]):
            new_cycle_direction = -1

        for ind, node in enumerate(cycle):
            ind_next = new_cycle_start + new_cycle_direction * ind
            if ind_next > cycle_len - 1:
                ind_next -= cycle_len
            new_cycle.append(cycle[ind_next])
        cycle_org.append(new_cycle)

    return cycle_org


def filter_rotations(cycle_list):
    """
    helper function to filter repeated/rotated/reflected cycles from a list of cycles
    takes:
        cycle_list - a list of unfiltered cycles
    returns:
        ret_cycles - a list of filtered cycles
    """
    ret_cycles = []
    cycle_list = organize_list(cycle_list)
    for cycle in cycle_list:
        if cycle not in ret_cycles:
            ret_cycles.append(cycle)

    return ret_cycles


def find_rings(atom_num, bond_list, allowed_ring_size=[], edges=True):
    cycle_graphs, cycle_list = [], []
    nx_graph = nx.Graph()
    [nx_graph.add_node(i) for i in range(atom_num)]
    #print("find rings: ", bond_list)
    nx_graph.add_edges_from(bond_list)

    for i in range(atom_num):
        try:
            cycle_edges = nx.find_cycle(nx_graph, source=i)
        except:
            cycle_edges = []

        nx_graph_cycle = nx.Graph()
        nx_graph_cycle.add_edges_from(cycle_edges)

        if cycle_graphs == []:  # adds fir cycle/graph
            cycle_list.append(cycle_edges)
            cycle_graphs.append(nx_graph_cycle)

        for cycle_graph in cycle_graphs:
            # filter isomorphic edges
            if not nx.is_isomorphic(cycle_graph, nx_graph_cycle):
                cycle_list.append(cycle_edges)
                cycle_graphs.append(nx_graph_cycle)
                break

    # convert cycles found to node lists
    cycle_list_nodes = []
    for cycle in cycle_list:
        node_list = [edge[0] for edge in cycle]
        cycle_list_nodes.append(node_list)
    cycle_list = cycle_list_nodes

    # filter for allowed ring sizes
    if allowed_ring_size != []:
        cycle_list_filtered = []
        for cycle in cycle_list:
            if len(cycle) in allowed_ring_size:
                cycle_list_filtered.append(cycle)
        cycle_list = cycle_list_filtered

    cycle_list.sort()
    cycle_list = list(cycle_list for cycle_list, _ in itertools.groupby(cycle_list))
    for i in range(len(cycle_list)):
        try:
            cycle_list.remove([])
        except:
            pass

    if len(cycle_list) > 1:
        cycle_list = filter_rotations(cycle_list)

    if edges == True:
        edge_list_list = []
        for cycle in cycle_list:
            edge_list = []
            for ind, node in enumerate(cycle[:-1]):
                edge_list.append((node, cycle[ind + 1]))
            edge_list.append((cycle[-1], cycle[0]))
            edge_list_list.append(edge_list)
        return edge_list_list
    return cycle_list


def rdkit_bond_desc(mol):
    """
    uses rdkit to get a dictionary with detected bond features to allow aromaticity, bond types to be detected
    takes:
        an rdkit molecule
    returns:
        ret_dict - a dictionary with the bonds(as root, target nodes) + descriptor info

    """
    detected_bonds_dict = {}
    allowed_bond_type = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    num_atoms = len(mol.GetAtoms())

    for i in range(num_atoms):
        for j in range(num_atoms):
            try:
                bond = mol[0].GetBondBetweenAtoms(i, j)
            except:
                try:
                    bond = mol.GetBondBetweenAtoms(i, j)
                except:
                    ft = []
            if not bond is None:  # checks if rdkit has detected the bond
                ft = [int(bond.GetIsConjugated())]
                ft += one_hot_encoding(bond.GetBondType(), allowed_bond_type)
                detected_bonds_dict[i, j] = ft  # adds with key = to the bond

    return detected_bonds_dict


# final model helper functions (for reaction graph generation)
def _split_batched_output(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.

    Returns:
        list of tensor.

    """
    nbonds = graph.batch_num_nodes("bond")

    return torch.split(value, nbonds)


def mol_graph_to_rxn_graph(
    graph,
    feats,
    reactions,
    device=None,
    reverse=False,
    reactant_only=False,
):
    """
    Convert a batched molecule graph to a batched reaction graph.

    Essentially, a reaction graph has the same graph structure as the reactant and
    its features are the difference between the products features and reactant features.

    Args:
        graph (BatchedDGLHeteroGraph): batched graph representing molecules.
        feats (dict): node features with node type as key and the corresponding
            features as value.
        reactions (list): a sequence of :class:`bondnet.data.reaction_network.Reaction`,
            each representing a reaction.

    Returns:
        batched_graph (BatchedDGLHeteroGraph): a batched graph representing a set of
            reactions.
        feats (dict): features for the batched graph
    """
    # updates node features on batches
    for nt, ft in feats.items():
        graph.nodes[nt].data.update({"ft": ft})
    # unbatch molecule graph
    graphs = dgl.unbatch(graph)
    reaction_graphs, reaction_feats = [], []
    batched_feats = {}

    for rxn in reactions:
        # print("rxn keys: ", rxn.keys())
        if type(rxn) == dict:
            if "reaction_molecule_info" in rxn.keys():
                reactants = [
                    graphs[i]
                    for i in rxn["reaction_molecule_info"]["reactants"]["reactants"]
                ]
                products = [
                    graphs[i]
                    for i in rxn["reaction_molecule_info"]["products"]["products"]
                ]

                mappings = rxn["mappings"]
                #reactants = [graphs[i] for i in rxn.reactants]
                #products = [graphs[i] for i in rxn.products]
               

                has_bonds = {
                    "reactants": [
                        True if len(mp) > 0 else False for mp in mappings["bond_map"][0]
                    ],
                    "products": [
                        True if len(mp) > 0 else False for mp in mappings["bond_map"][1]
                    ],
                }

                if len(has_bonds["reactants"]) != len(reactants) or len(
                    has_bonds["products"]
                    ) != len(products): print("unequal mapping & graph len")

                g, fts = create_rxn_graph(
                    reactants=reactants,
                    products=products,
                    mappings=mappings,
                    device=device,
                    has_bonds=None,
                    reverse=reverse,
                    reactant_only=reactant_only,
                    #empty_graph_fts={
                    #    "empty_graph": rxn["reaction_graph"],
                    #    "zero_feats": rxn["reaction_feature"],
                    #},
                    #empty_graph_fts=None
                )
        # check if reaction has key "mappings"
        else:
            reactants = [graphs[i] for i in rxn.reactants]
            products = [graphs[i] for i in rxn.products]

            mappings = {
                "bond_map": rxn.bond_mapping,
                "atom_map": rxn.atom_mapping,
                "total_bonds": rxn.total_bonds,
                "total_atoms": rxn.total_atoms,
                "num_bonds_total": rxn.num_bonds_total,
                "num_atoms_total": rxn.num_atoms_total,
            }
            has_bonds = {
                "reactants": [
                    True if len(mp) > 0 else False for mp in rxn.bond_mapping[0]
                ],
                "products": [
                    True if len(mp) > 0 else False for mp in rxn.bond_mapping[1]
                ],
            }

            if len(has_bonds["reactants"]) != len(reactants) or len(
                has_bonds["products"]
            ) != len(products):
                print("unequal mapping & graph len")

            g, fts = create_rxn_graph(
                reactants=reactants,
                products=products,
                mappings=mappings,
                device=device,
                has_bonds=has_bonds,
                reverse=reverse,
                reactant_only=reactant_only,
            )

        reaction_graphs.append(g)
        # if(device !=None):
        #    fts = {k: v.to(device) for k, v in fts.items()}
        reaction_feats.append(fts)
        ##################################################

    for i, g, ind in zip(reaction_feats, reaction_graphs, range(len(reaction_feats))):
        feat_len_dict = {}
        total_feats = 0

        for nt in i.keys():
            total_feats += i[nt].shape[0]
            feat_len_dict[nt] = i[nt].shape[0]

        assert total_feats == g.number_of_nodes(), "error in graph construction"

    batched_graph = dgl.batch(reaction_graphs)

    for nt in feats:  # batch features
        batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])
    return batched_graph, batched_feats


def construct_rxn_graph_empty(mappings, device=None, self_loop=True):
    bonds_compile = mappings["total_bonds"]
    atoms_compile = mappings["total_atoms"]
    num_bonds = len(bonds_compile)
    num_atoms = len(atoms_compile)
    a2b, b2a = [], []

    if num_bonds == 0:
        num_bonds = 1
        a2b, b2a = [(0, 0)], [(0, 0)]

    else:
        a2b, b2a = [], []
        for b in range(num_bonds):
            u = bonds_compile[b][0]
            v = bonds_compile[b][1]
            b2a.extend([[b, u], [b, v]])
            a2b.extend([[u, b], [v, b]])

    a2g = [(a, 0) for a in range(num_atoms)]
    g2a = [(0, a) for a in range(num_atoms)]
    b2g = [(b, 0) for b in range(num_bonds)]
    g2b = [(0, b) for b in range(num_bonds)]

    edges_dict = {
        ("atom", "a2b", "bond"): a2b,
        ("bond", "b2a", "atom"): b2a,
        ("atom", "a2g", "global"): a2g,
        ("global", "g2a", "atom"): g2a,
        ("bond", "b2g", "global"): b2g,
        ("global", "g2b", "bond"): g2b,
    }

    if self_loop:
        a2a = [(i, i) for i in atoms_compile]
        b2b = [(i, i) for i in range(num_bonds)]
        g2g = [(0, 0)]
        edges_dict.update(
            {
                ("atom", "a2a", "atom"): a2a,
                ("bond", "b2b", "bond"): b2b,
                ("global", "g2g", "global"): g2g,
            }
        )
    if device is not None:
        rxn_graph_zeroed = dgl.heterograph(edges_dict, device=device)
    else: 
        rxn_graph_zeroed = dgl.heterograph(edges_dict)
    
    return rxn_graph_zeroed


def create_rxn_graph(
    reactants,
    products,
    mappings,
    has_bonds=None,
    device=None,
    ntypes=("global", "atom", "bond"),
    ft_name="ft",
    reverse=False,
    reactant_only=False,
    zero_fts=False,
    empty_graph_fts=None,
):
    """
    A reaction is represented by:

    feats of products - feats of reactant

    Args:
        reactants (list of DGLHeteroGraph): a sequence of reactants graphs
        products (list of DGLHeteroGraph): a sequence of product graphs
        mappings (dict): with node type as the key (e.g. `atom` and `bond`) and a list
            as value, which is a mapping between reactant feature and product feature
            of the same atom (bond).
        has_bonds (dict): whether the reactants and products have bonds.
        ntypes (list): node types of which the feature are manipulated
        ft_name (str): key of feature inf data dict
        reverse (bool): whether to reverse the reaction direction
        zero_fts (bool): whether to zero out the features of the reactants and products

    Returns:
        graph (DGLHeteroGraph): a reaction graph with feats constructed from between
            reactant and products.
        feats (dict): features of reaction graph
    """
    # if reactant_only == True:
    #    print("REACTANT ONLY MODEL")

    verbose = False
    reactants_ft, products_ft = [], []
    feats = dict()

    # note, this assumes we have one reactant
    num_products = int(len(products))
    num_reactants = int(len(reactants))

    if empty_graph_fts is None:
        graph = construct_rxn_graph_empty(mappings, device=device)
    else:        
        graph = empty_graph_fts["empty_graph"]

    if verbose:
        print(
            "# reactions: {}, # products: {}".format(
                int(len(reactants)), int(len(products))
            )
        )

    #print("has bonds {}".format(has_bonds))
    for nt in ntypes:
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]
        # printy number of nodes of a type
        #print("num nodes of type ", nt, " in reactants: ", [p.num_nodes(nt) for p in reactants])
        #print("num nodes of type ", nt, " in products: ", [p.num_nodes(nt) for p in products])
        #print(
        #    "# reactions: {}, # products: {}".format(
        #        int(len(reactants)), int(len(products))
        #    )
        #)
        #if device is not None:
        #    reactants_ft = [r.to(device) for r in reactants_ft]
        #    products_ft = [p.to(device) for p in products_ft]


        if nt == "bond":
            if has_bonds == None:
                #print("manually constructing has bond maps")
                has_bonds = {
                    "reactants": [True if len(mp) > 0 else False for mp in mappings["bond_map"][0]],
                    "products": [True if len(mp) > 0 else False for mp in mappings["bond_map"][1]],
                }
                #print(has_bonds)
                #print(mappings["bond_map"])
                
            if num_products > 1:
                products_ft = list(
                    itertools.compress(products_ft, has_bonds["products"])
                )
                filter_maps = list(
                    itertools.compress(mappings[nt + "_map"][1], has_bonds["products"])
                )
                mappings[nt + "_map"] = [mappings[nt + "_map"][0], filter_maps]
                

                #if mappings[nt + "_map"] == [] and True in has_bonds["products"]:
                #    ind_t = has_bonds["products"].index(True)
                #    mappings[nt + "_map"] = mappings[nt + "_map"][0][ind_t]

            if num_reactants > 1:
                reactants_ft = list(
                    itertools.compress(reactants_ft, has_bonds["reactants"])
                )
                filter_maps = list(
                    itertools.compress(mappings[nt + "_map"][0], has_bonds["reactants"])
                )
                mappings[nt + "_map"] = [filter_maps, mappings[nt + "_map"][1]]

                #if mappings[nt + "_map"] == [] and True in has_bonds["reactants"]:
                #    ind_t = has_bonds["reactants"].index(True)
                #    mappings[nt + "_map"] = mappings[nt + "_map"][1][ind_t]
            

            #mappings["bond_map"][0]
            #mappings["bond_map"][1]

            """
            if num_reactants > 1 or num_products > 1:
                if False in has_bonds["products"]:
                    if False in has_bonds["products"]:
                        print("has bonds {}".format(has_bonds["products"]))
                        print("{}".format(mappings[nt + "_map"]))
                        print("{}".format(has_bonds["products"]))
                        print("has bonds {}".format(has_bonds["products"]))
                        print("{}".format(mappings[nt + "_map"]))
                        print("{}".format(has_bonds["products"]))
            """
            #assert len(has_bonds["reactants"]) == len(mappings["bond_map"][0]), "has_bond not the same length as mappings {} {} \n {} {}".format(has_bonds["reactants"], mappings["bond_map"][0],has_bonds["products"], mappings["bond_map"][1])
            #assert len(has_bonds["products"]) == len(mappings["bond_map"][1]), "has_bond not the same length as mappings {} {} \n {} {}".format(has_bonds["products"], mappings["bond_map"][1], has_bonds["reactants"], mappings["bond_map"][0])


        if nt == "global":
            reactants_ft = [
                torch.sum(reactant_ft, dim=0, keepdim=True)
                for reactant_ft in reactants_ft
            ]
            products_ft = [
                torch.sum(product_ft, dim=0, keepdim=True) for product_ft in products_ft
            ]
            #if device is not None:
            #    reactants_ft = [r.to(device) for r in reactants_ft]
            #    products_ft = [p.to(device) for p in products_ft]
        
        #print("reactants_ft: type {} full {}".format(nt, reactants_ft))

        len_feature_nt = reactants_ft[0].shape[1]
        # if(len_feature_nt!=64): print(mappings)
        # if empty_graph_fts is None:
        if nt == "global":
            net_ft_full = torch.zeros(1, len_feature_nt).type_as(reactants_ft[0])
        elif nt == "bond":
            net_ft_full = torch.zeros(mappings["num_bonds_total"], len_feature_nt).type_as(reactants_ft[0])
        else:
            net_ft_full = torch.zeros(mappings["num_atoms_total"], len_feature_nt).type_as(reactants_ft[0])
        # else:
        #    net_ft_full = empty_graph_fts["zero_feats"][nt]

        #if device is not None:
        #    net_ft_full = net_ft_full.to(device)

        if not zero_fts:
            if nt == "global":
                coef = torch.tensor([1]).type_as(reactants_ft[0])
                if reverse == True:
                    coef = torch.tensor([-1]).type_as(reactants_ft[0])
                #if device is not None:
                #    coef = coef.to(device)
                # don't add product features if we're only looking at reactants
                if reactant_only == False:
                    for product_ft in products_ft:
                        #print("device coef: ", coef.device, " reactant_ft: ", product_ft.device, "sum: ", torch.sum(product_ft, dim=0, keepdim=True).device)

                        net_ft_full += coef * torch.sum(product_ft, dim=0, keepdim=True)

                for reactant_ft in reactants_ft:
                    net_ft_full -= coef * torch.sum(reactant_ft, dim=0, keepdim=True)

            else:
                net_ft_full_zeros = copy.deepcopy(net_ft_full)

                coef = torch.tensor([1]).type_as(reactants_ft[0])
                if reverse == True:
                    coef = torch.tensor([-1]).type_as(reactants_ft[0])

                #if device is not None:
                #    coef = coef.to(device)
                
                # reactants
                #print("maps full {}".format(mappings[nt + "_map"]))
                #print("nt {}".format(nt))
                #print("nt maps {}".format(mappings[nt + "_map"]))
                for ind, reactant_ft in enumerate(reactants_ft):
                    #print(mappings[nt + "_map"][0], ind)

                    net_ft_full_temp = copy.deepcopy(net_ft_full_zeros)
                    mappings_raw = mappings[nt + "_map"][0][ind]
                    mappings_react = list(mappings_raw.keys())
                    mappings_total = [
                        mappings_raw[mapping] for mapping in mappings_react
                    ]
                    assert np.max(np.array(mappings_total)) < len(
                        net_ft_full_temp
                    ), f"invalid index  {mappings}"
                    #print("reactant ft ", net_ft_full_temp.shape, reactant_ft.shape, len(mappings_react), len(mappings_total), num_reactants, nt)
                    #net_ft_full_temp[mappings_total]
                    #reactant_ft[mappings_react]
                    net_ft_full_temp[mappings_total] = reactant_ft[mappings_react]
                    net_ft_full[mappings_total] -= (
                        coef * net_ft_full_temp[mappings_total]
                    )

                # products
                if reactant_only == False:
                    for ind, product_ft in enumerate(products_ft):
                        net_ft_full_temp = copy.deepcopy(net_ft_full_zeros)
                        #print(mappings[nt + "_map"])
                        #print(mappings[nt + "_map"][1], ind)
                        mappings_raw = mappings[nt + "_map"][1][ind]
                        mappings_prod = list(mappings_raw.keys())
                        mappings_total = [
                            mappings_raw[mapping] for mapping in mappings_prod
                        ]
                        assert np.max(np.array(mappings_total)) < len(
                            net_ft_full_temp
                        ), f"invalid index  {mappings}"
                        #print(product_ft.shape, len(mappings_prod), len(mappings_total), num_products, nt)
                        net_ft_full_temp[mappings_total] = product_ft[mappings_prod]
                        net_ft_full[mappings_total] += (
                            coef * net_ft_full_temp[mappings_total]
                        )

        feats[nt] = net_ft_full

    #print("graph device: ", graph.device, " feats device: ", feats["atom"].device)
    return graph, feats
