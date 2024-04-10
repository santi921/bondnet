import pandas as pd
import numpy as np
import networkx as nx
import itertools
import torch
from rdkit import Chem
import itertools, copy, dgl
import torch.autograd.profiler as profiler

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


def process_batch_mol_rxn(
    graph,
    feats,  #!mol features
    reactions, #!reactions
    device,
    reverse,
    reactant_only,
    atom_batch_indices,
    bond_batch_indices,
    global_batch_indices,
    batched_rxn_graphs, 
    batched_atom_reactant, 
    batched_atom_product, 
    batched_bond_reactant, 
    batched_bond_product ,
    batched_global_reactant,
    batched_global_product,
    global_batch_indices_reactant,
    global_batch_indices_product,
    mappings = None, #!needed for atom and bond.
    has_bonds = None, #!needed for atom and bond.
    ntypes=("global", "atom", "bond"),
    ft_name="ft",
    zero_fts=False,
    empty_graph_fts=True,
):
    
    #breakpoint()
    #!TODO: TypeError: object of type 'NoneType' has no len()
    distinguishable_value = torch.iinfo(torch.long).max
    batched_feats = {}
    for nt in ntypes:
        _features = feats[nt]
        coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)

        if nt == "global":

            num_batches = len(reactions)
            batch_one_hot = torch.nn.functional.one_hot(global_batch_indices_reactant, num_batches).float()
            gathered_content = _features.index_select(0, batched_global_reactant)
            reactant_sum = torch.matmul(batch_one_hot.t(), gathered_content)

            #!product why float64?
            batch_one_hot = torch.nn.functional.one_hot(global_batch_indices_product, num_batches).float()
            gathered_content = _features.index_select(0, batched_global_product)
            product_sum = torch.matmul(batch_one_hot.t(), gathered_content)

            batched_feats[nt] = - reactant_sum*coef \
                                + product_sum*coef

            # #!reactant  why float64?
            # batch_one_hot = torch.nn.functional.one_hot(global_batch_indices_reactant, num_batches).to(torch.float64)
            # gathered_content = _features.index_select(0, batched_global_reactant).to(torch.float64)
            # reactant_sum = torch.matmul(batch_one_hot.t(), gathered_content)

            # #!product why float64?
            # batch_one_hot = torch.nn.functional.one_hot(global_batch_indices_product, num_batches).to(torch.float64)
            # gathered_content = _features.index_select(0, batched_global_product).to(torch.float64)
            # product_sum = torch.matmul(batch_one_hot.t(), gathered_content)

            # batched_feats[nt] = - reactant_sum*coef \
            #                     + product_sum*coef
            # batched_feats[nt] = batched_feats[nt].to(torch.float32)
            
        if nt == "atom":
            batched_feats[nt] = - _features[batched_atom_reactant]*coef \
                                + _features[batched_atom_product]*coef
        if nt=="bond":
            #breakpoint()
            net_full_feats_reactant = torch.zeros(len(batched_bond_reactant), _features.shape[1], device=device)

            #!batched_bond_reactant is correct equal to extracted features from 0 to N in bond. 
            valid_mask = batched_bond_reactant != distinguishable_value

            # filtered_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze()
            # net_full_feats_reactant[filtered_indices] = _features[filtered_indices]
            filtered_indices = batched_bond_reactant[valid_mask]
            net_full_feats_reactant[valid_mask] = _features[filtered_indices]

            net_full_feats_product = torch.zeros(len(batched_bond_reactant), _features.shape[1], device=device)
            valid_mask = batched_bond_product != distinguishable_value
            filtered_indices = batched_bond_product[valid_mask]
            net_full_feats_product[valid_mask] = _features[filtered_indices]

            batched_feats[nt] = - net_full_feats_reactant*coef \
                                + net_full_feats_product*coef
        #if nt=="global":
    return batched_feats



    # len_feature_nt = atom_bond_features.size(1)

    # net_ft_full = torch.zeros(mappings[mappings_length], len_feature_nt, device=atom_bond_features.device)

    # reaction_graphs, reaction_feats = [], []
    # batched_feats = {}

    # for rxn in reactions:
    #     reation_feats = dict() #for each reaction there should be a reaction_fts
    #     reactant_ids= rxn["reaction_molecule_info"]["reactants"]["reactants"]
    #     product_ids = rxn["reaction_molecule_info"]["products"]["products"]
    #     mappings = rxn["mappings"]
    #     for nt in ntypes:
    #         if nt == "global":
    #             #get reactants and products features
    #             global_features = feats[nt] #! (80x512)
    #             reactants_features = global_features[reactant_ids]
    #             products_features = global_features[product_ids]
    #             reactants_ft_sum = reactants_features.sum(dim=0, keepdim=True)
    #             products_ft_sum = products_features.sum(dim=0, keepdim=True)
    #             net_ft_full = torch.zeros(1, global_features.size(1), device=device)
    #             #coef for reverse label
    #             coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)
    #             # Calculate the net effect, considering reactant_only flag
    #             if not reactant_only:
    #                 net_ft_full += coef * products_ft_sum
    #             net_ft_full -= coef * reactants_ft_sum
    #             # Store the computed net effect in feats dictionary under 'global' key
    #             reation_feats["global"] = net_ft_full

    #         else:
    #             if nt == "bond":
    #                 mappings_length = "num_bonds_total"
    #                 mappings_keys = "bond_map"
    #                 atom_bond_batch_indices = bond_batch_indices
    #             else:
    #                 mappings_length = "num_atoms_total"
    #                 mappings_keys = "atom_map"
    #                 atom_bond_batch_indices = atom_batch_indices

    #             atom_bond_features = feats[nt]

    #             coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)

    #             len_feature_nt = atom_bond_features.size(1)

    #             net_ft_full = torch.zeros(mappings[mappings_length], len_feature_nt, device=atom_bond_features.device)

    #             def apply_masks_and_reorder_with_coef(atom_bond_batch_indices, atom_bond_features, ids_list, mappings_list, coef, add_to_net=True):
    #                 for ids, mappings in zip(ids_list, mappings_list):
    #                     mask = torch.zeros_like(atom_bond_batch_indices, dtype=torch.bool)
    #                     mask |= (atom_bond_batch_indices == ids)
                        
    #                     selected_features = atom_bond_features[mask]
    #                     # Assuming mappings is a list of dictionaries for each reactant/product
    #                     mol_idx = list(mappings.keys())
    #                     react_idx = list(mappings.values())
    #                     # Apply coef directly during feature selection and placement
    #                     if add_to_net:  # Adding features to net_ft_full
    #                         net_ft_full[react_idx] += selected_features[mol_idx] * coef
    #                     else:  # Subtracting features from net_ft_full
    #                         net_ft_full[react_idx] -= selected_features[mol_idx] * coef

    #             apply_masks_and_reorder_with_coef(atom_bond_batch_indices, atom_bond_features, reactant_ids, mappings[mappings_keys][0], coef, add_to_net=False)
    #             if not reactant_only:  # Assuming 'reactant_only' flag is defined
    #                 apply_masks_and_reorder_with_coef(atom_bond_batch_indices, atom_bond_features, product_ids, mappings[mappings_keys][1], coef, add_to_net=True)
    #             # At this point, net_ft_full contains the computed net effect of features.
    #             reation_feats[nt] = net_ft_full
    #     #breakpoint()

    #     if empty_graph_fts:
    #         graph = rxn["reaction_graph"]
    #     reaction_graphs.append(graph)
    #     reaction_feats.append(reation_feats)
        
    # for i, g, ind in zip(reaction_feats, reaction_graphs, range(len(reaction_feats))):
    #     feat_len_dict = {}
    #     total_feats = 0

    #     for nt in i.keys():
    #         total_feats += i[nt].shape[0]
    #         feat_len_dict[nt] = i[nt].shape[0]

    #     assert total_feats == g.number_of_nodes(), "error in graph construction"

    # #breakpoint()
    # batched_graph = dgl.batch(reaction_graphs)

    # #reaction_feats [20 (reactions)] [atom 44x512, bond 45x512, global 1x512]
    # #batched empty graph, batched feats of reactions.
    # for nt in feats:  # batch features
    #     batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])
    
    # return batched_graph, batched_feats


def unbatch_mol_graph_to_rxn_graph(
    graph,
    feats,
    reactions,
    device=None,
    reverse=False,
    reactant_only=False,
    atom_batch_indices = None,
    bond_batch_indices = None,
    global_batch_indices = None,
    mappings = None,
    has_bonds = None,
    ntypes=("global", "atom", "bond"),
    ft_name="ft",
    zero_fts=False,
    empty_graph_fts=None,
):

    """
    Convert a unbatched molecule graph to a batched reaction graph.
    Note: ubatched graph is not used instead of their features

    Args:
        graph (BatchedDGLHeteroGraph): unbatched graph representing molecules.
        feats (dict): unbatched feats
        reactions (list): a sequence of :class:`bondnet.data.reaction_network.Reaction`,
            each representing a reaction.
    Returns:
    TBD
    """
    reaction_graphs, reaction_feats = [], []
    batched_feats = {}

    for rxn in reactions:
        reation_feats = dict() #for each reaction there should be a reaction_fts
        reactant_ids= rxn["reaction_molecule_info"]["reactants"]["reactants"]
        product_ids = rxn["reaction_molecule_info"]["products"]["products"]
        mappings = rxn["mappings"]
        for nt in ntypes:
            if nt == "global":
                #get reactants and products features
                global_features = feats[nt] #! (80x512)
                reactants_features = global_features[reactant_ids]
                products_features = global_features[product_ids]
                reactants_ft_sum = reactants_features.sum(dim=0, keepdim=True)
                products_ft_sum = products_features.sum(dim=0, keepdim=True)
                net_ft_full = torch.zeros(1, global_features.size(1), device=device)
                #coef for reverse label
                coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)
                # Calculate the net effect, considering reactant_only flag
                if not reactant_only:
                    net_ft_full += coef * products_ft_sum
                net_ft_full -= coef * reactants_ft_sum
                # Store the computed net effect in feats dictionary under 'global' key
                reation_feats["global"] = net_ft_full

            else:
                if nt == "bond":
                    mappings_length = "num_bonds_total"
                    mappings_keys = "bond_map"
                    atom_bond_batch_indices = bond_batch_indices
                else:
                    mappings_length = "num_atoms_total"
                    mappings_keys = "atom_map"
                    atom_bond_batch_indices = atom_batch_indices

                atom_bond_features = feats[nt]

                coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)

                len_feature_nt = atom_bond_features.size(1)

                net_ft_full = torch.zeros(mappings[mappings_length], len_feature_nt, device=atom_bond_features.device)

                def apply_masks_and_reorder_with_coef(atom_bond_batch_indices, atom_bond_features, ids_list, mappings_list, coef, add_to_net=True):
                    for ids, mappings in zip(ids_list, mappings_list):
                        mask = torch.zeros_like(atom_bond_batch_indices, dtype=torch.bool)
                        mask |= (atom_bond_batch_indices == ids)
                        
                        selected_features = atom_bond_features[mask]
                        # Assuming mappings is a list of dictionaries for each reactant/product
                        mol_idx = list(mappings.keys())
                        react_idx = list(mappings.values())
                        # Apply coef directly during feature selection and placement
                        if add_to_net:  # Adding features to net_ft_full
                            net_ft_full[react_idx] += selected_features[mol_idx] * coef
                        else:  # Subtracting features from net_ft_full
                            net_ft_full[react_idx] -= selected_features[mol_idx] * coef

                apply_masks_and_reorder_with_coef(atom_bond_batch_indices, atom_bond_features, reactant_ids, mappings[mappings_keys][0], coef, add_to_net=False)
                if not reactant_only:  # Assuming 'reactant_only' flag is defined
                    apply_masks_and_reorder_with_coef(atom_bond_batch_indices, atom_bond_features, product_ids, mappings[mappings_keys][1], coef, add_to_net=True)
                # At this point, net_ft_full contains the computed net effect of features.
                reation_feats[nt] = net_ft_full
        #breakpoint()

        if empty_graph_fts:
            graph = rxn["reaction_graph"]
        reaction_graphs.append(graph)
        reaction_feats.append(reation_feats)
        
    for i, g, ind in zip(reaction_feats, reaction_graphs, range(len(reaction_feats))):
        feat_len_dict = {}
        total_feats = 0

        for nt in i.keys():
            total_feats += i[nt].shape[0]
            feat_len_dict[nt] = i[nt].shape[0]

        assert total_feats == g.number_of_nodes(), "error in graph construction"

    #breakpoint()
    batched_graph = dgl.batch(reaction_graphs)

    #reaction_feats [20 (reactions)] [atom 44x512, bond 45x512, global 1x512]
    #batched empty graph, batched feats of reactions.
    for nt in feats:  # batch features
        batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])
    
    return batched_graph, batched_feats





        #     bond_features = feats[nt] #!(2618, 512)
        #     coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)
        #     len_feature_nt = bond_features.size(1)
        #     net_ft_full = torch.zeros(mappings["num_bonds_total"], len_feature_nt, device=bond_features.device)

        #     def apply_masks_and_reorder_with_coef(bond_batch_indices, bond_features, ids_list, mappings_list, coef, add_to_net=True):
        #         breakpoint()
        #         for ids, mappings in zip(ids_list, mappings_list):
        #             mask = torch.zeros_like(bond_batch_indices, dtype=torch.bool)
        #             mask |= (bond_batch_indices == ids)
                    
        #             selected_features = bond_features[mask]
        #             # Assuming mappings is a list of dictionaries for each reactant/product
        #             mol_idx = list(mappings.keys())
        #             react_idx = list(mappings.values())
        #             # Apply coef directly during feature selection and placement
        #             if add_to_net:  # Adding features to net_ft_full
        #                 net_ft_full[react_idx] += selected_features[mol_idx] * coef
        #             else:  # Subtracting features from net_ft_full
        #                 net_ft_full[react_idx] -= selected_features[mol_idx] * coef
            
        #     apply_masks_and_reorder_with_coef(bond_batch_indices, bond_features, reactant_ids, mappings["bond_map"][0], coef, add_to_net=False)
        #     if not reactant_only:  # Assuming 'reactant_only' flag is defined
        #         apply_masks_and_reorder_with_coef(bond_batch_indices, bond_features, product_ids, mappings["bond_map"][1], coef, add_to_net=True)
        #     # At this point, net_ft_full contains the computed net effect of features.
        #     reation_feats["bond"] = net_ft_full

        # if nt == "atom":
        #     atom_features = feats[nt] #!(2618, 512)
        #     coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)
        #     len_feature_nt = atom_features.size(1)
        #     net_ft_full = torch.zeros(mappings["num_atoms_total"], len_feature_nt, device=bond_features.device)

        #     def apply_masks_and_reorder_with_coef(bond_batch_indices, bond_features, ids_list, mappings_list, coef, add_to_net=True):
        #         breakpoint()
        #         for ids, mappings in zip(ids_list, mappings_list):
        #             mask = torch.zeros_like(bond_batch_indices, dtype=torch.bool)
        #             mask |= (bond_batch_indices == ids)
                    
        #             selected_features = bond_features[mask]
        #             # Assuming mappings is a list of dictionaries for each reactant/product
        #             mol_idx = list(mappings.keys())
        #             react_idx = list(mappings.values())
        #             # Apply coef directly during feature selection and placement
        #             if add_to_net:  # Adding features to net_ft_full
        #                 net_ft_full[react_idx] += selected_features[mol_idx] * coef
        #             else:  # Subtracting features from net_ft_full
        #                 net_ft_full[react_idx] -= selected_features[mol_idx] * coef
            
        #     apply_masks_and_reorder_with_coef(atom_batch_indices, atom_features, reactant_ids, mappings["bond_map"][0], coef, add_to_net=False)
        #     if not reactant_only:  # Assuming 'reactant_only' flag is defined
        #         apply_masks_and_reorder_with_coef(bond_batch_indices, bond_features, product_ids, mappings["bond_map"][1], coef, add_to_net=True)
        #     # At this point, net_ft_full contains the computed net effect of features.
        #     reation_feats["bond"] = net_ft_full
        











        # reordered_reactant_features = apply_masks_and_reorder(bond_batch_indices, bond_features, reactant_ids, mappings[0])

        # reordered_product_features = apply_masks_and_reorder(bond_batch_indices, bond_features, product_ids, mappings[1])
        # reactant_mask = torch.zeros_like(bond_batch_indices, dtype=torch.bool, device=device)
        # product_mask = torch.zeros_like(bond_batch_indices, dtype=torch.bool, device=device)
        # for reactant_id in reactant_ids:
        #     reactant_mask |= (bond_batch_indices == reactant_id)   #!(44x512)
        # for product_id in product_ids:
        #     product_mask |= (bond_batch_indices == product_id) #!(44x512)

        # # breakpoint()
        # # reactant_bond_features = bond_features[reactant_mask]
        # # product_bond_features = bond_features[product_mask]


        # # reactant_bond_features_reordered = apply_reordering(reactant_bond_features, reactant_mapping)
        # # product_bond_features_reordered = apply_reordering(product_bond_features, product_mapping)

        # # reactants_ft_sum = reactant_bond_features.sum(dim=0, keepdim=True)
        # # products_ft_sum = product_bond_features.sum(dim=0, keepdim=True)

        # # len_features = bond_features.size(1)
        # # # Prepare tensor to hold the net effect of bond features
        # # net_bond_ft_full = torch.zeros(1, len_features, device=device)

        # # # Coefficients for aggregation
        # # coef = torch.tensor([1], device=device) if not reverse else torch.tensor([-1], device=device)

        # # # Calculate the net effect, considering reactant_only flag
        # # if not reactant_only:
        # #     net_bond_ft_full += coef * products_ft_sum
        # # net_bond_ft_full -= coef * reactants_ft_sum

        # # # Store the computed net effect in feats dictionary under 'bond' key
        # # feats["bond"] = net_bond_ft_full

def mol_graph_to_rxn_graph(
    graph,
    feats,
    reactions,
    device=None,
    reverse=False,
    reactant_only=False,
    atom_batch_indices = None,
    bond_batch_indices = None,
    global_batch_indices = None,
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

    with profiler.record_function("update features"):

        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})
        # unbatch molecule graph
        graphs = dgl.unbatch(graph)
        reaction_graphs, reaction_feats = [], []
        batched_feats = {}
        with profiler.record_function("iterative reactions"):
            for rxn in reactions:
                #breakpoint()
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

                        #!return g from empty and fts of g
                        g, fts = create_rxn_graph(
                            reactants=reactants,
                            products=products,
                            mappings=mappings,
                            device=device,
                            has_bonds=None,
                            reverse=reverse,
                            reactant_only=reactant_only,
                            #!put empty graph.
                            empty_graph_fts={
                                "empty_graph": rxn["reaction_graph"],
                                "zero_feats": rxn["reaction_feature"],
                                },
                        # empty_graph_fts=None
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
    with profiler.record_function("batch reaction graphs"):
        for i, g, ind in zip(reaction_feats, reaction_graphs, range(len(reaction_feats))):
            feat_len_dict = {}
            total_feats = 0

            for nt in i.keys():
                total_feats += i[nt].shape[0]
                feat_len_dict[nt] = i[nt].shape[0]

            assert total_feats == g.number_of_nodes(), "error in graph construction"

        #breakpoint()
        batched_graph = dgl.batch(reaction_graphs)

        #reaction_feats [20 (reactions)] [atom 44x512, bond 45x512, global 1x512]
        #batched empty graph, batched feats of reactions.
        for nt in feats:  # batch features
            batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])
        return batched_graph, batched_feats

        #reaction_graphs 0，的原子数 == reaction_feats[0]["atom"].shape [44x512]
        #所以返回的是 batched reaction_graph, and batched_feats. 
        #batched reaction_graph直接从empty拿到，batched_feats直接从其他拿到
        #batched molecule 是一个[graph[i]] loop拿到的，所以感觉差别不大。如果loop reactions. 那么reaction_feats如何得到。


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

    #breakpoint()
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

    for nt in ntypes:
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]
<<<<<<< HEAD
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
=======
>>>>>>> wenbin_dev2/wenbin_dev

        if nt == "bond":
            if has_bonds == None:
                has_bonds = {
                    "reactants": [True if len(mp) > 0 else False for mp in mappings["bond_map"][0]],
                    "products": [True if len(mp) > 0 else False for mp in mappings["bond_map"][1]],
                }

                
            if num_products > 1:
                products_ft = list(
                    itertools.compress(products_ft, has_bonds["products"])
                )
                filter_maps = list(
                    itertools.compress(mappings[nt + "_map"][1], has_bonds["products"])
                )
                mappings[nt + "_map"] = [mappings[nt + "_map"][0], filter_maps]
                

            if num_reactants > 1:
                reactants_ft = list(
                    itertools.compress(reactants_ft, has_bonds["reactants"])
                )
                filter_maps = list(
                    itertools.compress(mappings[nt + "_map"][0], has_bonds["reactants"])
                )
                mappings[nt + "_map"] = [filter_maps, mappings[nt + "_map"][1]]


        if nt == "global":
            #!reactants_ft[0] (1x512), reactants_ft[1] (1x512)
            #!sum dim
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
            #!(1x512)
            net_ft_full = torch.zeros(1, len_feature_nt).type_as(reactants_ft[0])
        elif nt == "bond":
            net_ft_full = torch.zeros(mappings["num_bonds_total"], len_feature_nt).type_as(reactants_ft[0])
        else:
            net_ft_full = torch.zeros(mappings["num_atoms_total"], len_feature_nt).type_as(reactants_ft[0])

        if not zero_fts:
            if nt == "global":
                coef = torch.tensor([1]).type_as(reactants_ft[0])
                if reverse == True:
                    coef = torch.tensor([-1]).type_as(reactants_ft[0])
                
                if reactant_only == False:
                    for product_ft in products_ft:
                
                        net_ft_full += coef * torch.sum(product_ft, dim=0, keepdim=True)

                for reactant_ft in reactants_ft:
                    #!(1x512)
                    net_ft_full -= coef * torch.sum(reactant_ft, dim=0, keepdim=True)

            else:
                net_ft_full_zeros = copy.deepcopy(net_ft_full)

                coef = torch.tensor([1]).type_as(reactants_ft[0])
                if reverse == True:
                    coef = torch.tensor([-1]).type_as(reactants_ft[0])

<<<<<<< HEAD
                #if device is not None:
                #    coef = coef.to(device)
                
                # reactants
                #print("maps full {}".format(mappings[nt + "_map"]))
                #print("nt {}".format(nt))
                #print("nt maps {}".format(mappings[nt + "_map"]))
                breakpoint()
=======

>>>>>>> wenbin_dev2/wenbin_dev
                for ind, reactant_ft in enumerate(reactants_ft):

                    net_ft_full_temp = copy.deepcopy(net_ft_full_zeros)
                    mappings_raw = mappings[nt + "_map"][0][ind]
                    mappings_react = list(mappings_raw.keys())
                    mappings_total = [
                        mappings_raw[mapping] for mapping in mappings_react
                    ]
                    assert np.max(np.array(mappings_total)) < len(
                        net_ft_full_temp
                    ), f"invalid index  {mappings}"

                    net_ft_full_temp[mappings_total] = reactant_ft[mappings_react]
                    net_ft_full[mappings_total] -= (
                        coef * net_ft_full_temp[mappings_total]
                    )

                if reactant_only == False:
                    for ind, product_ft in enumerate(products_ft):
                        net_ft_full_temp = copy.deepcopy(net_ft_full_zeros)

                        mappings_raw = mappings[nt + "_map"][1][ind]
                        mappings_prod = list(mappings_raw.keys())
                        mappings_total = [
                            mappings_raw[mapping] for mapping in mappings_prod
                        ]
                        assert np.max(np.array(mappings_total)) < len(
                            net_ft_full_temp
                        ), f"invalid index  {mappings}"
                        net_ft_full_temp[mappings_total] = product_ft[mappings_prod]
                        net_ft_full[mappings_total] += (
                            coef * net_ft_full_temp[mappings_total]
                        )

        feats[nt] = net_ft_full

    #print("graph device: ", graph.device, " feats device: ", feats["atom"].device)
    return graph, feats
