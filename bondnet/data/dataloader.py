import torch
import dgl
import itertools
from torch.utils.data import DataLoader

"""
class DataLoader(DataLoader):


    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graphs = dgl.batch(graphs)
            batched_labels = torch.utils.data.dataloader.default_collate(labels)

            return batched_graphs, batched_labels

        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)
"""


class DataLoaderGraphNorm(DataLoader):
    """
    This dataloader works for the case where the label of each data point are of the
    same shape. For example, regression on molecule energy.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in graphs]
            sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            batched_labels = torch.utils.data.dataloader.default_collate(labels)

            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderGraphNorm, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderBond(DataLoader):
    """
    This dataloader works for bond related dataset, where bond specific properties (
    e.g. bond energy) needs to be ber specified by an index.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graphs = dgl.batch(graphs)

            value = torch.cat([la["value"] for la in labels])
            # index_0 = labels[0]["bond_index"]
            # indices = [
            #     labels[i]["bond_index"] + labels[i - 1]["num_bonds_in_molecule"]
            #     for i in range(1, len(labels))
            # ]
            # indices = torch.stack(index_0 + indices)
            num_bonds = torch.stack([la["num_bonds_in_molecule"] for la in labels])
            staring_index = torch.cumsum(num_bonds, dim=0)
            index = torch.cat(
                [
                    la["bond_index"]
                    if i == 0
                    else la["bond_index"] + staring_index[i - 1]
                    for i, la in enumerate(labels)
                ]
            )
            batched_labels = {"value": value, "index": index}

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.cat(mean)
                batched_labels["scaler_stdev"] = torch.cat(stdev)
            except KeyError:
                pass

            return batched_graphs, batched_labels

        super(DataLoaderBond, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderReaction(DataLoader):
    """
    This dataloader works specifically for the reaction dataset where each reaction is
    represented by a list of the molecules (i.e. reactants and products).

    Also, the label value of each datapoint should be of the same shape.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            # note each element of graph is a list of mol graphs that constitute a rxn
            # flatten double list
            graphs = list(itertools.chain.from_iterable(graphs))
            batched_graphs = dgl.batch(graphs)

            target_class = torch.stack([la["value"] for la in labels])
            atom_mapping = [la["atom_mapping"] for la in labels]
            bond_mapping = [la["bond_mapping"] for la in labels]
            global_mapping = [la["global_mapping"] for la in labels]
            num_mols = [la["num_mols"] for la in labels]
            identifier = [la["id"] for la in labels]

            batched_labels = {
                "value": target_class,
                "atom_mapping": atom_mapping,
                "bond_mapping": bond_mapping,
                "global_mapping": global_mapping,
                "num_mols": num_mols,
                "id": identifier,
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            return batched_graphs, batched_labels

        super(DataLoaderReaction, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderReactionNetwork(DataLoader):
    """
    This dataloader works specifically for the reaction network where a the reactions
    are constructed from a list of reactions.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            rn, rxn_ids, labels = map(list, zip(*samples))

            # each element of `rn` is the same reaction network
            reactions, graphs = rn[0].subselect_reactions(rxn_ids)

            batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in graphs]
            sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            target = torch.stack([la["value"] for la in labels])
            value_rev = torch.stack([la["value_rev"] for la in labels])
            identifier = [la["id"] for la in labels]

            reaction_types = [la["reaction_type"] for la in labels]

            batched_labels = {
                "value": target,
                "value_rev": value_rev,
                "id": identifier,
                "reaction": reactions,
                "reaction_types": reaction_types,
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            # graph norm
            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderReactionNetwork, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


"""def collate(samples):
    reaction_graph, reaction_features, labels = map(list, zip(*samples))

    batched_graphs = dgl.batch(reaction_graph)
    sizes_atom = [g.number_of_nodes("atom") for g in reaction_graph]
    sizes_bond = [g.number_of_nodes("bond") for g in reaction_graph]

    target = torch.stack([la["value"] for la in labels])
    value_rev = torch.stack([la["value_rev"] for la in labels])
    identifier = [la["id"] for la in labels]

    reaction_types = [la["reaction_type"] for la in labels]

    batched_labels = {
        "value": target,
        "value_rev": value_rev,
        "id": identifier,
        "reaction_types": reaction_types,
    }

    # add label scaler if it is used

    try:
        mean = [la["scaler_mean"] for la in labels]
        stdev = [la["scaler_stdev"] for la in labels]
        batched_labels["scaler_mean"] = torch.stack(mean)
        batched_labels["scaler_stdev"] = torch.stack(stdev)
    except KeyError:
        pass
    # graph norm
    norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
    norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
    batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
    batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

    return batched_graphs, batched_labels
"""


class DataLoaderPrecomputedReactionGraphs(DataLoader):
    """
    This dataloader works specifically precomputed reaction networks
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            reaction_graph, reaction_features, labels = map(list, zip(*samples))

            # each element of `rn` is the same reaction network
            # reactions, graphs = rn[0].subselect_reactions(rxn_ids)

            batched_graphs = dgl.batch(reaction_graph)
            sizes_atom = [g.number_of_nodes("atom") for g in reaction_graph]
            sizes_bond = [g.number_of_nodes("bond") for g in reaction_graph]

            target = torch.stack([la["value"] for la in labels])
            value_rev = torch.stack([la["value_rev"] for la in labels])
            identifier = [la["id"] for la in labels]

            reaction_types = [la["reaction_type"] for la in labels]

            batched_labels = {
                "value": target,
                "value_rev": value_rev,
                "id": identifier,
                "reaction_types": reaction_types,
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            # graph norm
            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderPrecomputedReactionGraphs, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


class DataLoaderPrecomputedReactionGraphsParallel(DataLoader):
    """
    This dataloader works specifically precomputed reaction networks
    """

    def __init__(self, dataset, **kwargs):
        super(DataLoaderPrecomputedReactionGraphsParallel, self).__init__(
            dataset, **kwargs
        )


def collate_parallel(samples):
    reaction_graph, reaction_features, labels = map(list, zip(*samples))

    # each element of `rn` is the same reaction network
    # reactions, graphs = rn[0].subselect_reactions(rxn_ids)

    batched_graphs = dgl.batch(reaction_graph)
    sizes_atom = [g.number_of_nodes("atom") for g in reaction_graph]
    sizes_bond = [g.number_of_nodes("bond") for g in reaction_graph]

    target = torch.stack([la["value"] for la in labels])
    value_rev = torch.stack([la["value_rev"] for la in labels])
    identifier = [la["id"] for la in labels]

    reaction_types = [la["reaction_type"] for la in labels]

    batched_labels = {
        "value": target,
        "value_rev": value_rev,
        "id": identifier,
        "reaction_types": reaction_types,
    }

    # add label scaler if it is used
    try:
        mean = [la["scaler_mean"] for la in labels]
        stdev = [la["scaler_stdev"] for la in labels]
        batched_labels["scaler_mean"] = torch.stack(mean)
        batched_labels["scaler_stdev"] = torch.stack(stdev)
    except KeyError:
        pass

    # graph norm
    norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
    norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
    batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
    batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

    return batched_graphs, batched_labels
