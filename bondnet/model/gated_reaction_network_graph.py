import torch
import itertools, copy, dgl
import numpy as np
from bondnet.model.gated_mol import GatedGCNMol
from bondnet.data.utils import (
    _split_batched_output,
    mol_graph_to_rxn_graph,
)


class GatedGCNReactionNetwork(GatedGCNMol):
    def forward(
        self,
        graph,
        feats,
        reactions,
        norm_atom=None,
        norm_bond=None,
        device=None,
        reverse=False,
    ):
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
