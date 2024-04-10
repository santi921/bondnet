"""
Global readout (pooling) layers.
"""
import torch
from torch import nn
import dgl
from typing import List, Tuple, Dict, Optional
from dgl import function as fn
from dgl.readout import sum_nodes, softmax_nodes
import torch.nn.functional as F


class ConcatenateMeanMax(nn.Module):
    """
    Concatenate the mean and max of features of a node type to another node type.

    Args:
        etypes: canonical edge types of a graph of which the features of node
            `u` are concatenated to the features of node `v`.
            For example: if `etypes = [('atom', 'a2b', 'bond'), ('global','g2b', 'bond')]`
            then the mean and max of the features of `atom` as well as  `global` are
            concatenated to the features of `bond`.
    """

    def __init__(self, etypes: List[Tuple[str, str, str]]):
        super(ConcatenateMeanMax, self).__init__()
        self.etypes = etypes

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.

        Returns:
            updated node features. Each tensor is of shape (N, D) where N is the number
            of nodes of the corresponding node type, and D is the feature size.

        """
        graph = graph.local_var()

        # assign data
        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})

        for et in self.etypes:
            # option 1
            graph[et].update_all(fn.copy_u("ft", "m"), fn.mean("m", "mean"), etype=et)
            graph[et].update_all(fn.copy_u("ft", "m"), fn.max("m", "max"), etype=et)

            nt = et[2]
            graph.apply_nodes(self._concatenate_node_feat, ntype=nt)

            # copy update feature from new_ft to ft
            graph.nodes[nt].data.update({"ft": graph.nodes[nt].data["new_ft"]})

        return {nt: graph.nodes[nt].data["ft"] for nt in feats}

    @staticmethod
    def _concatenate_node_feat(nodes):
        data = nodes.data["ft"]
        mean = nodes.data["mean"]
        max = nodes.data["max"]
        concatenated = torch.cat((data, mean, max), dim=1)
        return {"new_ft": concatenated}


class ConcatenateMeanAbsDiff(nn.Module):
    """
    Concatenate the mean and max of features of a node type to another node type.

    This is very specific to the scheme that two atoms directed to bond. Others may fail.

    Args:
        etypes: canonical edge types of a graph of which the features of node `u`
            are concatenated to the features of node `v`.
            For example: if `etypes = [('atom', 'a2b', 'bond'), ('global','g2b', 'bond')]`
            then the mean and max of the features of `atom` and `global` are concatenated
            to the features of `bond`.
    """

    def __init__(self, etypes):
        super(ConcatenateMeanAbsDiff, self).__init__()
        self.etypes = etypes

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.

        Returns:
            updated node features. Each tensor is of shape (N, D) where N is the number
            of nodes of the corresponding node type, and D is the feature size.
        """
        graph = graph.local_var()

        # assign data
        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})

        for et in self.etypes:
            graph[et].update_all(fn.copy_u("ft", "m"), self._concatenate_data, etype=et)

        return {nt: graph.nodes[nt].data["ft"] for nt in feats}

    @staticmethod
    def _concatenate_data(nodes):
        message = nodes.mailbox["m"]
        mean_v = torch.mean(message, dim=1)
        # NOTE this is very specific to the atom -> bond case
        # there are two elements along dim=1, since for each bond we have two atoms
        # directed to it
        abs_diff = torch.stack([torch.abs(x[0] - x[1]) for x in message])
        data = nodes.data["ft"]

        concatenated = torch.cat((data, mean_v, abs_diff), dim=1)
        return {"ft": concatenated}


class Set2Set(nn.Module):
    r"""
    Compared to the Official dgl implementation, we allowed node type.

    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Args:
        input_dim: The size of each input sample.
        n_iters: The number of iterations.
        n_layers: The number of recurrent layers.
        ntype: Type of the node to apply Set2Set.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int, ntype: str):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.ntype = ntype
        self.lstm = torch.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute set2set pooling.

        Args:
            graph: the input graph
            feat: The input feature with shape :math:`(N, D)` where  :math:`N` is the
                number of nodes in the graph, and :math:`D` means the size of features.

        Returns:
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to
            the batch size, and :math:`D` means the size of features.
        """
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * dgl.broadcast_nodes(graph, q, ntype=self.ntype)).sum(
                    dim=-1, keepdim=True
                )
                graph.nodes[self.ntype].data["e"] = e
                alpha = dgl.softmax_nodes(graph, "e", ntype=self.ntype)
                graph.nodes[self.ntype].data["r"] = feat * alpha
                readout = dgl.sum_nodes(graph, "r", ntype=self.ntype)
                q_star = torch.cat([q, readout], dim=-1)

            return q_star

    def extra_repr(self):
        """
        Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = "n_iters={n_iters}"
        return summary.format(**self.__dict__)


class Set2SetThenCat(nn.Module):
    """
    Set2Set for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        n_iter: number of LSTM iteration
        n_layer: number of LSTM layers
        ntypes: node types to perform Set2Set, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform Set2Set, whose feature is
            directly concatenated. e.g. ['global']
            #!this is because a+b->c = c-a-b no need Set2Set
    """

    def __init__(
        self,
        n_iters: int,
        n_layer: int,
        ntypes: List[str],
        in_feats: List[int],
        ntypes_direct_cat: Optional[List[str]] = None,
    ):
        super(Set2SetThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat

        self.layers = nn.ModuleDict()
        for nt, sz in zip(ntypes, in_feats):
            self.layers[nt] = Set2Set(
                input_dim=sz, n_iters=n_iters, n_layers=n_layer, ntype=nt
            )

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.
        Returns:
            update features. Each tensor is of shape (B, D), where B is the batch size
                and D is the feature size. Note D could be different for different
                node type.

        """
        rst = []
        for nt in self.ntypes:
            ft = self.layers[nt](graph, feats[nt])
            rst.append(ft)

        if self.ntypes_direct_cat is not None:
            for nt in self.ntypes_direct_cat:
                rst.append(feats[nt])

        res = torch.cat(rst, dim=-1)  # dim=-1 to deal with batched graph

        return res


class SumPoolingThenCat(nn.Module):
    """
    SumPooling for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform SumPooling, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform SumPooling, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(SumPoolingThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        # self.layers = nn.ModuleDict()
        # for nt, sz in zip(ntypes, in_feats):
        #    if nt not in ntypes_direct_cat:
        #        self.layers[nt] = dgl.SumPooling(ntype=nt)

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sumpooling of each node type and each graph in the batch.
        """
        rst = []
        with graph.local_scope():
            graph.ndata["h"] = feats

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    rst.append(dgl.readout_nodes(graph, "h", ntype=ntype, op="sum"))

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)


class WeightAndSumThenCat(nn.Module):
    """
    WeightAndSum for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform WeightAndSum, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform WeightAndSum, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(WeightAndSumThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        self.atom_weighting = nn.ModuleDict()
        for ntype, size in zip(ntypes, in_feats):
            if ntype not in ntypes_direct_cat:
                self.atom_weighting[ntype] = nn.Linear(size, 1)

        # for ntype, size in zip(ntypes, in_feats):
        #    self.layers[ntype] = WeightAndSum(in_feats=size)

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sumpooling of each node type and each graph in the batch.
        """
        rst = []
        with graph.local_scope():
            weight_dict = {}
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    weight_dict[ntype] = self.atom_weighting[ntype](feats[ntype])

            graph.ndata["h"] = feats
            graph.ndata["w"] = weight_dict

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    rst.append(
                        dgl.readout_nodes(graph, "h", "w", ntype=ntype, op="sum")
                    )

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)


class GlobalAttentionPoolingThenCat(nn.Module):
    """
    GlobalAttentionPooling for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform GlobalAttentionPooling, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform GlobalAttentionPooling, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(GlobalAttentionPoolingThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        self.gate_nn = nn.ModuleDict()
        for ntype, in_feat in zip(ntypes, in_feats):
            self.gate_nn[ntype] = nn.Linear(in_feat, 1)

    def forward(self, graph, feats, get_attention=False):
        rst = []
        readout_dict = {}
        gate_dict = {}
        gated_feats = {}
        with graph.local_scope():
            # gather, assign gate to graph
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    gate_dict[ntype] = F.leaky_relu(self.gate_nn[ntype](feats[ntype]))

            graph.ndata["gate"] = gate_dict
            graph.nodes["atom"].data["gate"]
            graph.nodes["bond"].data["gate"]

            # gather, assign gated features to graph
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    gate = softmax_nodes(graph=graph, feat="gate", ntype=ntype)
                    gated_feats[ntype] = feats[ntype] * gate
            graph.ndata.pop("gate")

            # gather, assign readout features to graph
            graph.ndata["r"] = gated_feats
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    readout_dict[ntype] = sum_nodes(graph, "r", ntype=ntype)
                    rst.append(readout_dict[ntype])
            graph.ndata.pop("r")

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        rst = torch.cat(rst, dim=-1)

        if get_attention:
            return rst, gate
        else:
            return rst


class MeanPoolingThenCat(nn.Module):
    """
    SumPooling for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform SumPooling, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform SumPooling, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(MeanPoolingThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        # self.layers = nn.ModuleDict()
        # for nt, sz in zip(ntypes, in_feats):
        #    if nt not in ntypes_direct_cat:
        #        self.layers[nt] = dgl.SumPooling(ntype=nt)

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sumpooling of each node type and each graph in the batch.
        """
        rst = []
        with graph.local_scope():
            graph.ndata["h"] = feats

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    rst.append(dgl.readout_nodes(graph, "h", ntype=ntype, op="mean"))

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)


class WeightAndMeanThenCat(nn.Module):
    """
    WeightAndSum for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform WeightAndSum, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform WeightAndSum, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(WeightAndMeanThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        self.atom_weighting = nn.ModuleDict()
        for ntype, size in zip(ntypes, in_feats):
            if ntype not in ntypes_direct_cat:
                self.atom_weighting[ntype] = nn.Linear(size, 1)

        # for ntype, size in zip(ntypes, in_feats):
        #    self.layers[ntype] = WeightAndSum(in_feats=size)

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sumpooling of each node type and each graph in the batch.
        """
        rst = []
        with graph.local_scope():
            weight_dict = {}
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    weight_dict[ntype] = self.atom_weighting[ntype](feats[ntype])

            graph.ndata["h"] = feats
            graph.ndata["w"] = weight_dict

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    rst.append(
                        dgl.readout_nodes(graph, "h", "w", ntype=ntype, op="mean")
                    )

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)

