"""
Graph Attention Networks for heterograph.
"""

import torch.nn as nn
from gnn.layer.hgatconv import HGATConv
from gnn.layer.readout import ConcatenateMeanMax


class HGAT(nn.Module):
    """
    Heterograph attention network.


    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key,
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): The order to attend the node features. Order matters.
        in_feats (list): input feature size for the corresponding node in `attn_order`.
        num_gat_layers (int): number of graph attention layer
        gat_hidden_size (list): hidden size of graph attention layers
        gat_activation (torch activation): activation fn of gat layers
        num_heads (int): number of attention heads, the same for all nodes
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        residual (bool, optional): [description]. Defaults to False.
        num_fc_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_activation (torch activation): activation fn of fc layers
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        num_gat_layers=3,
        gat_hidden_size=[32, 64, 128],
        gat_activation=nn.ELU(),
        num_heads=4,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        num_fc_layers=3,
        fc_hidden_size=[128, 64, 32],
        fc_activation=nn.ELU(),
    ):
        super(HGAT, self).__init__()

        self.gat_layers = nn.ModuleList()

        # input projection (no dropout, no residual)
        self.gat_layers.append(
            HGATConv(
                attn_mechanism=attn_mechanism,
                attn_order=attn_order,
                in_feats=in_feats,
                out_feats=gat_hidden_size[0],
                num_heads=num_heads,
                feat_drop=0.0,
                attn_drop=0.0,
                negative_slope=negative_slope,
                residual=False,
                activation=gat_activation,
            )
        )

        # hidden gat layers
        for i in range(1, num_gat_layers):
            in_size = [gat_hidden_size[i - 1] * num_heads for _ in in_feats]
            self.gat_layers.append(
                HGATConv(
                    attn_mechanism=attn_mechanism,
                    attn_order=attn_order,
                    in_feats=in_size,
                    out_feats=gat_hidden_size[i],
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=gat_activation,
                )
            )

        # TODO this could be passed in as arguments
        etypes = [("atom", "a2b", "bond")]
        self.readout_layer = ConcatenateMeanMax(etypes=etypes)

        self.fc_layers = nn.ModuleList()
        # outsize of readout_layer; 3 because we concatenate atom feats to bond feats
        # in readout_layer
        in_size = gat_hidden_size[-1] * num_heads * 3
        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(in_size, fc_hidden_size[i]))
            self.fc_layers.append(fc_activation)
            in_size = fc_hidden_size[i]

        # final output layer, mapping feature to size 1
        self.fc_layers.append(nn.Linear(in_size, 1))

    def forward(self, graph, feats):
        h = feats

        # hgat layer
        for layer in self.gat_layers:
            h = layer(graph, h)

        # readout layer
        h = self.readout_layer(graph, h)

        # fc
        h = h["bond"]
        for layer in self.fc_layers:
            h = layer(h)
        h = h.view(-1)  # reshape to a 1D tensor to make each component a bond energy

        return h

    def feature_before_fc(self, graph, feats):

        h = feats

        # hgat layer
        for layer in self.gat_layers:
            h = layer(graph, h)

        # readout layer
        h = self.readout_layer(graph, h)
        h = h["bond"]

        return h
