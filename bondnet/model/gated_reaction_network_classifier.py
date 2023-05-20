from copy import deepcopy
import torch
import itertools
import dgl
from bondnet.model.gated_mol import GatedGCNMol
import torch.nn as nn
from bondnet.layer.gatedconv import GatedGCNConv, GatedGCNConv1, GatedGCNConv2
from bondnet.layer.readout import Set2SetThenCat
from bondnet.layer.utils import UnifySize
from bondnet.data.utils import (
    _split_batched_output,
    mol_graph_to_rxn_graph,
)


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
