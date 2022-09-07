import torch
import itertools, copy, dgl
import numpy as np
from bondnet.model.gated_mol import GatedGCNMol


class GatedGCNReactionNetwork(GatedGCNMol):
    def forward(self, graph, feats, reactions, norm_atom=None, norm_bond=None, device=None, reverse = False):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            reactions (list): a sequence of :class:`bondnet.data.reaction_network.Reaction`,
                each representing a reaction.
            norm_atom (2D tensor or None): graph norm for atom
            norm_bond (2D tensor or None): graph norm for bond

        Returns:
            2D tensor: of shape(N, M), where `M = outdim`.
        """
        # embedding
        feats = self.embedding(feats)
        # gated layer
        
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)
        
        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        # graph is actually batch graphs, not just a graph
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions, device, reverse)
        
        # readout layer
        feats = self.readout_layer(graph, feats)
        
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats

    def feature_before_fc(self, graph, feats, reactions, norm_atom, norm_bond):
        """
        Get the features before the final fully-connected.

        This is used for feature visualization.
        """
        # embedding
        feats = self.embedding(feats)
        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)
        # readout layer
        feats = self.readout_layer(graph, feats)
        return feats

    def feature_at_each_layer(self, graph, feats, reactions, norm_atom, norm_bond):
        """
        Get the features at each layer before the final fully-connected layer.

        This is used for feature visualization to see how the model learns.

        Returns:
            dict: (layer_idx, feats), each feats is a list of
        """

        layer_idx = 0
        all_feats = dict()

        # embedding
        feats = self.embedding(feats)

        # store bond feature of each molecule
        fts = _split_batched_output(graph, feats["bond"])
        all_feats[layer_idx] = fts
        layer_idx += 1

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

            # store bond feature of each molecule
            fts = _split_batched_output(graph, feats["bond"])
            all_feats[layer_idx] = fts
            layer_idx += 1

        return all_feats


def _split_batched_output(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.

    Returns:
        list of tensor.

    """
    nbonds = graph.batch_num_nodes("bond")
    return torch.split(value, nbonds)


def mol_graph_to_rxn_graph(graph, feats, reactions, device=None, reverse = False):
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
    #updates node features on batches
    for nt, ft in feats.items():
        graph.nodes[nt].data.update({"ft": ft}) 
        #feats = {k: v.to(device) for k, v in feats.items()}
    # unbatch molecule graph
    graphs = dgl.unbatch(graph)
    reaction_graphs, reaction_feats = [], []
    batched_feats = {}

    for rxn in reactions:
        reactants = [graphs[i] for i in rxn.reactants]
        products = [graphs[i] for i in rxn.products]
        # might need to rewrite to create dynamics here
        mappings = {
                    "bond_map": rxn.bond_mapping, 
                    "atom_map": rxn.atom_mapping, 
                    "total_bonds": rxn.total_bonds,
                    "total_atoms": rxn.total_atoms,
                    "num_bonds_total":rxn.num_bonds_total,
                    "num_atoms_total":rxn.num_atoms_total
                    }
        has_bonds = {
            "reactants": [True if len(mp) > 0 else False 
                    for mp in rxn.bond_mapping[0]],
            "products": [True if len(mp) > 0 else False 
                    for mp in rxn.bond_mapping[1]],
        }
        if(len(has_bonds["reactants"])!=len(reactants) or 
            len(has_bonds["products"])!=len(products)): 
            print("unequal mapping & graph len")
        
        
        g, fts = create_rxn_graph(
            reactants, products, mappings, has_bonds, device, reverse = reverse
        )
        reaction_graphs.append(g)
        #if(device !=None):
        #    fts = {k: v.to(device) for k, v in fts.items()}
        reaction_feats.append(fts)
        ##################################################
    for i, g, ind in zip(reaction_feats, reaction_graphs, range(len(reaction_feats))): 
        feat_len_dict = {}
        total_feats = 0 
        
        for nt in i.keys():
            total_feats+=i[nt].shape[0]
            feat_len_dict[nt] = i[nt].shape[0]

        if(total_feats!=g.number_of_nodes()): # error checking
            print(g)
            print(feat_len_dict)
            print(reactions[ind].atom_mapping)
            print(reactions[ind].bond_mapping)
            print(reactions[ind].total_bonds)
            print(reactions[ind].total_atoms)
            print("--"*20)

    batched_graph = dgl.batch(reaction_graphs) # batch graphs
    for nt in feats: # batch features
        batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])
    return batched_graph, batched_feats


def construct_rxn_graph_empty(mappings, device=None, self_loop = True):

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
        
    rxn_graph_zeroed = dgl.heterograph(edges_dict)
    if(device is not None): rxn_graph_zeroed = rxn_graph_zeroed.to(device)
    return rxn_graph_zeroed


def create_rxn_graph(
    reactants,
    products,
    mappings,
    has_bonds,
    device=None, 
    ntypes=("global", "atom", "bond"),
    ft_name="ft",
    reverse = False
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

    Returns:
        graph (DGLHeteroGraph): a reaction graph with feats constructed from between
            reactant and products.
        feats (dict): features of reaction graph
    """
    verbose = False
    reactants_ft, products_ft = [], []
    feats = dict()
    
    # note, this assumes we have one reactant
    num_products = int(len(products))
    num_reactants = int(len(reactants))
    graph = construct_rxn_graph_empty(mappings, device)
    if(verbose):
        print("# reactions: {}, # products: {}".format(int(len(reactants)), int(len(products))))

    for nt in ntypes:
        
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]
        
        if(device is not None):
            reactants_ft = [r.to(device) for r in reactants_ft]
            products_ft = [p.to(device) for p in products_ft]

        if(nt=='bond'):
            if(num_products>1):
                products_ft = list(itertools.compress(products_ft, has_bonds["products"]))
                filter_maps = list(itertools.compress(mappings[nt+"_map"][1], has_bonds["products"]))
                mappings[nt+"_map"] = [mappings[nt+"_map"][0], filter_maps]
                
            if(num_reactants>1):      
                reactants_ft = list(itertools.compress(reactants_ft, has_bonds["reactants"]))
                filter_maps = list(itertools.compress(mappings[nt+"_map"][0], has_bonds["reactants"]))
                mappings[nt+"_map"] = [filter_maps, mappings[nt+"_map"][1]]

        if(nt=='global'):
            reactants_ft = [torch.sum(reactant_ft, dim=0, keepdim=True) for reactant_ft in reactants_ft]
            products_ft = [torch.sum(product_ft, dim=0, keepdim=True) for product_ft in products_ft]
            if(device is not None):
                reactants_ft = [r.to(device) for r in reactants_ft]
                products_ft = [p.to(device) for p in products_ft]
        
        len_feature_nt = reactants_ft[0].shape[1]
        #if(len_feature_nt!=64): print(mappings)

        if(nt == 'global'):
            net_ft_full = torch.zeros(1, len_feature_nt)
        elif(nt == 'bond'):
            net_ft_full = torch.zeros(mappings['num_bonds_total'],len_feature_nt) 
        else: 
            net_ft_full = torch.zeros(mappings['num_atoms_total'], len_feature_nt) 
        
        if(device is not None):
            net_ft_full = net_ft_full.to(device)
        
        if nt == "global": 
            coef = 1
            if(reverse==True): coef = -1

            for product_ft in products_ft:
                net_ft_full += coef * torch.sum(product_ft, dim=0, keepdim=True)                
            for reactant_ft in reactants_ft:
                net_ft_full -= coef * torch.sum(reactant_ft, dim=0, keepdim=True) 

        else:
            net_ft_full_temp = copy.deepcopy(net_ft_full) 
            coef = 1
            if(reverse==True): coef = -1
            for ind, reactant_ft in enumerate(reactants_ft):
                mappings_raw = mappings[nt+"_map"][0][ind]
                mappings_react = list(mappings_raw.keys())
                mappings_total = [mappings_raw[mapping] for mapping in mappings_react]
                assert np.max(np.array(mappings_total)) < len(net_ft_full_temp), f"invalid index  {mappings}"
                net_ft_full_temp[mappings_total] = reactant_ft[mappings_react]
                net_ft_full -= coef * net_ft_full_temp

            for ind, product_ft in enumerate(products_ft):
                mappings_raw = mappings[nt+"_map"][1][ind]
                mappings_prod = list(mappings_raw.keys())
                mappings_total = [mappings_raw[mapping] for mapping in mappings_prod]
                assert np.max(np.array(mappings_total)) < len(net_ft_full_temp), f"invalid index  {mappings}"
                net_ft_full_temp[mappings_total] = product_ft[mappings_prod]
                net_ft_full += coef * net_ft_full_temp

        feats[nt] = net_ft_full
    return graph, feats