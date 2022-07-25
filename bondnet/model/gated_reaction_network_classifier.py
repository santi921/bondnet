from copy import deepcopy
import torch
import itertools
import numpy as np
import dgl
from bondnet.model.gated_mol import GatedGCNMol
import torch.nn.functional as F
import torch.nn as nn
from bondnet.layer.gatedconv import GatedGCNConv, GatedGCNConv1, GatedGCNConv2
from bondnet.layer.readout import Set2SetThenCat
from bondnet.layer.utils import UnifySize


class GatedGCNReactionNetworkClassifier(GatedGCNMol):
    def __init__(
        self,
        in_feats,
        embedding_size=32,
        gated_num_layers=2,
        gated_hidden_size=[64, 64, 32],
        gated_num_fc_layers=1,
        gated_graph_norm=False,
        gated_batch_norm=True,
        gated_activation="ReLU",
        gated_residual=True,
        gated_dropout=0.0,
        num_lstm_iters=6,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        fc_num_layers=2,
        fc_hidden_size=[32, 16],
        fc_batch_norm=False,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=1,
        conv="GatedGCNConv",
    ):
        super(GatedGCNMol, self).__init__()

        if isinstance(gated_activation, str):
            gated_activation = getattr(nn, gated_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        # embedding layer
        self.embedding = UnifySize(in_feats, embedding_size)

        # gated layer
        if conv == "GatedGCNConv":
            conv_fn = GatedGCNConv
        elif conv == "GatedGCNConv1":
            conv_fn = GatedGCNConv1
        elif conv == "GatedGCNConv2":
            conv_fn = GatedGCNConv2
        else:
            raise ValueError()

        in_size = embedding_size
        self.gated_layers = nn.ModuleList()
        for i in range(gated_num_layers):
            self.gated_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=gated_hidden_size[i],
                    num_fc_layers=gated_num_fc_layers,
                    graph_norm=gated_graph_norm,
                    batch_norm=gated_batch_norm,
                    activation=gated_activation,
                    residual=gated_residual,
                    dropout=gated_dropout,
                )
            )
            in_size = gated_hidden_size[i]

        # set2set readout layer
        ntypes = ["atom", "bond"]
        in_size = [gated_hidden_size[-1]] * len(ntypes)

        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size  of in feature)
        readout_out_size = gated_hidden_size[-1] * 2 + gated_hidden_size[-1] * 2
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size
        for i in range(fc_num_layers):
            out_size = fc_hidden_size[i]
            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_dropout))

            in_size = out_size

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))
        self.fc_layers.append(nn.Softmax(dim=1))

    def forward(
        self, graph, feats, reactions, target, stdev, norm_atom=None, norm_bond=None
    ):

        pred_filtered_index = []
        pred_filtered = []
        stdev_filtered = []
        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        ###############################
        graph_copy = deepcopy(graph)
        # assign feats
        for nt, ft in feats.items():
            graph_copy.nodes[nt].data.update({"ft": ft})
        graphs_copy = dgl.unbatch(graph_copy)
        # create reaction graphs
        reaction_graphs, reaction_feats = [], []
        for rxn in reactions:
            reactants = [graphs_copy[i] for i in rxn.reactants]
            products = [graphs_copy[i] for i in rxn.products]
            has_bonds = {
                # we support only one reactant now, so no it is assumed always to have bond
                "reactants": [True for _ in reactants],
                "products": [True if len(mp) > 0 else False for mp in rxn.bond_mapping],
            }
            mappings = {
                "atom": rxn.atom_mapping_as_list,
                "bond": rxn.bond_mapping_as_list,
            }
            ft_name, nt = "ft", "bond"
            reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
            products_ft = [p.nodes[nt].data[ft_name] for p in products]
            products_ft = list(itertools.compress(products_ft, has_bonds["products"]))
            products_ft.append(reactants_ft[0].new_zeros((1, reactants_ft[0].shape[1])))
            reactants_ft = torch.cat(reactants_ft)
            products_ft = torch.cat(products_ft)
            products_ft = products_ft[mappings[nt]]
            if len(products_ft) == len(reactants_ft):
                pred_filtered_index.append(1)
            else:
                pred_filtered_index.append(0)
        ###############################

        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        # graph is actually batch graphs, not just a graph
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)

        # readout layer
        feats = self.readout_layer(graph, feats)

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        for ind, i in enumerate(pred_filtered_index):
            if i == 1:
                pred_filtered.append(target[ind])
                stdev_filtered.append(stdev[ind].tolist())

        # print(pred_filtered)
        # ret_tensor = torch.cat(pred_filtered)
        # try:
        #    ret_tensor = torch.Tensor(pred_filtered)
        # except:
        ret_tensor = torch.cat(pred_filtered)
        try:
            stdev_filtered = torch.Tensor(stdev_filtered)
        except:
            stdev_filtered = torch.cat(stdev_filtered)

        # print(F.softmax(feats))
        return feats, ret_tensor, stdev_filtered

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

        graphs = dgl.unbatch(graph)

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


def mol_graph_to_rxn_graph(graph, feats, reactions):
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
    # TODO add graph.local_var() since hetero and homo graphs are combined
    # should not use graph.local_var() to make a local copy, since it converts a
    # BatchedDGLHeteroGraph into a DGLHeteroGraph. Then unbatch_hetero(graph) below
    # will not work.
    # If you really want to, use copy.deepcopy() to make a local copy

    # assign feats
    for nt, ft in feats.items():
        graph.nodes[nt].data.update({"ft": ft})

    # unbatch molecule graph
    graphs = dgl.unbatch(graph)

    # create reaction graphs
    reaction_graphs = []
    reaction_feats = []
    for rxn in reactions:

        reactants = [graphs[i] for i in rxn.reactants]
        products = [graphs[i] for i in rxn.products]
        # print(len(torch.cat([p.nodes['bond'].data['ft'] for p in reactants])),
        # len(torch.cat([p.nodes['bond'].data['ft'] for p in products])))
        # whether a molecule has bonds?
        has_bonds = {
            # we support only one reactant now, so no it is assumed always to have bond
            "reactants": [True for _ in reactants],
            "products": [True if len(mp) > 0 else False for mp in rxn.bond_mapping],
        }
        mappings = {"atom": rxn.atom_mapping_as_list, "bond": rxn.bond_mapping_as_list}

        ##################################################
        graph = reactants[0]
        ft_name = "ft"
        nt = "bond"
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]
        products_ft = list(itertools.compress(products_ft, has_bonds["products"]))
        # add a feature with all zeros for the broken bond
        products_ft.append(reactants_ft[0].new_zeros((1, reactants_ft[0].shape[1])))
        reactants_ft = torch.cat(reactants_ft)
        products_ft = torch.cat(products_ft)
        # reorder products_ft such that atoms/bonds have the same order as reactants
        products_ft = products_ft[mappings[nt]]
        # adds padding if bond isn't mapped/doesn't exist
        g, fts = create_rxn_graph(
            reactants, products, mappings, has_bonds, tuple(feats.keys())
        )
        if len(products_ft) == len(reactants_ft):
            reaction_graphs.append(g)
            reaction_feats.append(fts)
        ##################################################

    # batched reaction graph and data
    batched_graph = dgl.batch(reaction_graphs)
    batched_feats = {}
    for nt in feats:
        batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])

    return batched_graph, batched_feats


def create_rxn_graph(
    reactants,
    products,
    mappings,
    has_bonds,
    ntypes=("atom", "bond", "global"),
    ft_name="ft",
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
    assert len(reactants) == 1, f"number of reactants ({len(reactants)}) not supported"

    # note, this assumes we have one reactant
    graph = reactants[0]
    feats = dict()
    for nt in ntypes:
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]

        # remove bond ft if the corresponding molecule has no bond
        # this is necessary because, to make heterogeneous graph work, we create
        # fictitious bond features for molecule without any bond (i.e. single atom
        # molecule, e.g. H+)

        if nt == "bond":
            products_ft = list(itertools.compress(products_ft, has_bonds["products"]))
            # add a feature with all zeros for the broken bond
            products_ft.append(reactants_ft[0].new_zeros((1, reactants_ft[0].shape[1])))

        reactants_ft = torch.cat(reactants_ft)
        products_ft = torch.cat(products_ft)

        if nt == "global":
            reactants_ft = torch.sum(reactants_ft, dim=0, keepdim=True)
            products_ft = torch.sum(products_ft, dim=0, keepdim=True)
        else:
            # reorder products_ft such that atoms/bonds have the same order as reactants
            assert len(products_ft) == len(mappings[nt]), (
                f"products_ft ({len(products_ft)}) and mappings[{nt}] "
                f"({len(mappings[nt])}) have different length"
            )
            products_ft = products_ft[mappings[nt]]

            ##########################################
            # adds padding if bond isn't mapped/doesn't exist
            if len(products_ft) != len(reactants_ft):
                # print(len(mappings[nt]), len(products_ft), len(reactants_ft))
                unequal = True
                while unequal:
                    if len(products_ft) < len(reactants_ft):
                        products_ft = torch.cat(
                            (products_ft, torch.zeros(1, products_ft.size()[1])), 0
                        )
                    else:
                        reactants_ft = torch.cat(
                            (reactants_ft, torch.zeros(1, reactants_ft.size()[1])), 0
                        )
                    if len(products_ft) == len(reactants_ft):
                        unequal = False
            ##########################################

        feats[nt] = products_ft - reactants_ft
    return graph, feats
