import torch
import dgl
import itertools
from torch.utils.data import DataLoader


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
            reactions, graphs, labels = map(list, zip(*samples))
            #graphs = [i.graph for i in molecules]
            # note each element of graph is a list of mol graphs that constitute a rxn
            # flatten double list
            graphs = list(itertools.chain.from_iterable(graphs))
            
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


class DataLoaderReactionNetworkParallel(DataLoader):
    """
    This dataloader works specifically for the reaction network where a the reactions
    are constructed from a list of reactions.
    """

    def __init__(self, dataset, **kwargs):
        super(DataLoaderReactionNetworkParallel, self).__init__(dataset, **kwargs)



class DataLoaderReactionNetworkLMDB(DataLoader):
    """
    This dataloader works specifically for the reaction network where a the reactions
    are constructed from a list of reactions.
    """

    def __init__(self, dataset, **kwargs):
        super(DataLoaderReactionNetworkLMDB, self).__init__(dataset, **kwargs)


def collate_parallel_lmdb(samples):
    reaction_network, rxn_ids = map(list, zip(*samples))  # new
    reactions, graphs = reaction_network[0].subselect_reactions(rxn_ids)

    batched_graphs = dgl.batch(graphs)
    sizes_atom = [g.number_of_nodes("atom") for g in graphs]
    sizes_bond = [g.number_of_nodes("bond") for g in graphs]

    target = torch.stack([reaction["label"] for reaction in reactions])
    value_rev = torch.stack([reaction["reverse_label"] for reaction in reactions])
    identifier = [reaction["reaction_index"] for reaction in reactions]
    # reaction_types = [reaction["reaction_type"] for reaction in reactions]

    batched_labels = {
        "value": target,
        "value_rev": value_rev,
        "reaction": reactions,
        "id": identifier,
        # "reaction_types": reaction_types,
    }

    # add label scaler if it is used
    try:
        # TODO SORT OUT SCALER INFO

        batched_labels["scaler_mean"] = torch.zeros(target.shape[1])
        batched_labels["scaler_stdev"] = torch.ones(target.shape[1])

    except KeyError:
        pass

    # graph norm
    norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
    norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
    batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
    batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

    return batched_graphs, batched_labels



def collate_parallel(samples):
    # reaction_graph, reaction_features, labels = map(list, zip(*samples))  # old

    reaction_network, rxn_ids, labels = map(list, zip(*samples))  # new
    reactions, graphs = reaction_network[0].subselect_reactions(rxn_ids)

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
        "reaction": reactions,
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

