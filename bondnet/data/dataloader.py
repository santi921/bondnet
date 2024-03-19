import torch
import dgl
import itertools
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np 

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
            # reaction_graph, reaction_features, labels = map(list, zip(*samples))  # old

            reactions, labels = map(list, zip(*samples))
            molecules = self.dataset.graphs

            mol_ids = set()
            for rxn in reactions:
                mol_ids.update(rxn.init_reactants + rxn.init_products)
            mol_ids = sorted(mol_ids)
            global_to_subset_mapping = {g: s for s, g in enumerate(mol_ids)}

            for rxn in reactions:
                rxn.reactants = [global_to_subset_mapping[i] for i in rxn.init_reactants]
                rxn.products = [global_to_subset_mapping[i] for i in rxn.init_products]

            # molecules subset
            graphs = [molecules[i] for i in mol_ids]


            batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in graphs]
            sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            target = torch.stack([la["value"] for la in labels])
            value_rev = torch.stack([la["value_rev"] for la in labels])
            identifier = [la["id"] for la in labels]

            #reaction_types = [la["reaction_type"] for la in labels]

            batched_labels = {
                "value": target,
                "value_rev": value_rev,
                "reaction": reactions,
                "id": identifier,
                #"reaction_types": reaction_types,
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


class DataLoaderReactionLMDB(DataLoader):
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
            reactions, labels = map(list, zip(*samples))
            graphs = self.dataset.graphs

            mol_ids = set()
            for rxn in reactions:
                mol_ids.update(
                    rxn["reaction_molecule_info"]["reactants"]["init_reactants"]
                    + rxn["reaction_molecule_info"]["products"]["init_products"]
                )
            mol_ids = sorted(mol_ids)
            global_to_subset_mapping = {g: s for s, g in enumerate(mol_ids)}
            ################### from reasction network ######################

            #global_to_subset_mapping = {g: s for s, g in enumerate(mol_id_map)} 
            #graphs_unsorted = list(itertools.chain.from_iterable(graphs))
            #graphs_unsorted = []
            #for reaction_graphs in graphs:
            #    [graphs_unsorted.append(graph_temp) for graph_temp in reaction_graphs[0]]
            #    [graphs_unsorted.append(graph_temp) for graph_temp in reaction_graphs[1]]
            #print(len(graphs_unsorted), len(mol_id_map))
            #print(mol_id_map)
            #graphs_unsorted = []
            #graphs_sorted = []
            #test_sort = []

            for rxn in reactions:
                init_reactants = rxn["reaction_molecule_info"]["reactants"][
                    "init_reactants"
                ]
                init_products = rxn["reaction_molecule_info"]["products"]["init_products"]
                mapped_reactants = [global_to_subset_mapping[i] for i in init_reactants]
                mapped_products = [global_to_subset_mapping[i] for i in init_products]
                rxn["reaction_molecule_info"]["reactants"]["reactants"] = mapped_reactants
                rxn["reaction_molecule_info"]["products"]["products"] = mapped_products
            # molecules subset
            sub_molecules = [graphs[i] for i in mol_ids]
            
            #print(mol_id_map)
            #print(test_sort)
            #graphs = graphs_sorted    
            # get graphs, ordered by mol_ids
            
            
            # resort graphs to match the new mol_ids
            batched_graphs = dgl.batch(sub_molecules)
            ################### from reaction network ######################
            

            #batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in sub_molecules]
            sizes_bond = [g.number_of_nodes("bond") for g in sub_molecules]

            target = torch.stack([la["value"] for la in labels])
            value_rev = torch.stack([la["value_rev"] for la in labels])
            identifier = [la["id"] for la in labels]


            batched_labels = {
                "value": target,
                "value_rev": value_rev,
                "reaction": reactions,
                "id": identifier
            }


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
        
        super(DataLoaderReactionLMDB, self).__init__(dataset, collate_fn=collate, **kwargs)




