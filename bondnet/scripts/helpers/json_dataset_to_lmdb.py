import argparse
import torch
import json
import numpy as np
import argparse
import dgl
import torch
import torch.multiprocessing
import tempfile
from copy import deepcopy
from bondnet.data.lmdb import (
    write_reaction_lmdb,
    write_molecule_lmdb,
    serialize_dgl_graph,
    load_dgl_graph_from_serialized,
)
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.utils import seed_torch
from bondnet.data.utils import create_rxn_graph, find_rings
from bondnet.model.training_utils import get_grapher
from bondnet.dataset.utils import (
    clean,
    clean_op,
)


torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
seed_torch()
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def construct_lmdb_and_save(dataset, lmdb_dir, workers=8):
    # List of Molecules
    dgl_graphs = []
    dgl_graphs_serialized = []
    pmg_objects = []
    molecule_ind_list = []
    charge_set = set()
    ring_size_set = set()
    element_set = set()
    feature_size = dataset._feature_size
    feature_name = dataset._feature_name
    feature_scaler_mean = dataset._feature_scaler_mean
    feature_scaler_std = dataset._feature_scaler_std
    label_scaler_mean = dataset._label_scaler_mean
    label_scaler_std = dataset._label_scaler_std
    dtype = dataset.dtype

    # add these to global keys
    #print("label_scaler_mean: ", label_scaler_mean)
    #print("label_scaler_std: ", label_scaler_std)

    for ind, molecule_in_rxn_network in enumerate(
        dataset.reaction_network.molecule_wrapper
    ):
        #    pmg mol retrieval
        pmg_objects.append(molecule_in_rxn_network.pymatgen_mol)
        #    molecule index in rxn network
        # this would just be given by the index in HiPRGen anyways
        molecule_ind_list.append(ind)

        formula = molecule_in_rxn_network.pymatgen_mol.composition.formula.split()
        elements = [clean(x) for x in formula]
        atom_num = np.sum(np.array([int(clean_op(x)) for x in formula]))
        element_set.update(elements)

        charge = molecule_in_rxn_network.pymatgen_mol.charge
        charge_set.add(charge)
        bond_list = [
            [i[0], i[1]] for i in molecule_in_rxn_network.mol_graph.graph.edges
        ]
        cycles = find_rings(atom_num, bond_list, edges=False)
        ring_len_list = [len(i) for i in cycles]
        ring_size_set.update(ring_len_list)

    for ind, molecule_in_rxn_network in enumerate(dataset.reaction_network.molecules):
        # serialized dgl graph
        dgl_graphs_serialized.append(serialize_dgl_graph(molecule_in_rxn_network))
        dgl_graphs.append(molecule_in_rxn_network)
        #dgl_graph_non_serialized = load_dgl_graph_from_serialized(dgl_graphs[-1])
        # option 2: don't serialize the graph, just the features
        #dgl_graphs.append(molecule_in_rxn_network)

    batched_graph = dgl.batch(dgl_graphs)
    feats = batched_graph.ndata["feat"]
    for nt, ft in feats.items():
        batched_graph.nodes[nt].data.update({"ft": ft})
    graphs = dgl.unbatch(batched_graph)

    extra_info = []
    reaction_molecule_info = []
    label_list = []
    reverse_list = []
    has_bonds_list = []
    mappings_list = []
    empty_reaction_graphs = []
    empty_reaction_fts = []
    reaction_indicies = []

    global_dict = {
        "feature_size": feature_size,
        "mean": label_scaler_mean,
        "std": label_scaler_std,
        "feature_name": feature_name,
        "feature_scaler_mean": feature_scaler_mean,
        "feature_scaler_std": feature_scaler_std,
        "dtype": dtype
    }

    for ind, rxn in enumerate(dataset.reaction_network.reactions):
        rxn_copy = deepcopy(rxn)
        
        reactants = [graphs[i] for i in rxn_copy.reactants]
        products = [graphs[i] for i in rxn_copy.products]
        
        mappings = {
            "bond_map": rxn_copy.bond_mapping,
            "atom_map": rxn_copy.atom_mapping,
            "total_bonds": rxn_copy.total_bonds,
            "total_atoms": rxn_copy.total_atoms,
            "num_bonds_total": rxn_copy.num_bonds_total,
            "num_atoms_total": rxn_copy.num_atoms_total,
        }
        has_bonds = {
            "reactants": [
                True if len(mp) > 0 else False for mp in rxn_copy.bond_mapping[0]
            ],
            "products": [
                True if len(mp) > 0 else False for mp in rxn_copy.bond_mapping[1]
            ],
        }

        if len(has_bonds["reactants"]) != len(reactants) or len(
            has_bonds["products"]
        ) != len(products):
            print("unequal mapping & graph len")

        #assert len(has_bonds["reactants"]) == len(mappings["bond_map"][0]), "has_bond not the same length as mappings {} {}".format(has_bonds["reactants"], mappings["bond_map"][0])
        #assert len(has_bonds["products"]) == len(mappings["bond_map"][1]), "has_bond not the same length as mappings {} {}".format(has_bonds["products"], mappings["bond_map"][1])


        """empty_graph, empty_fts = create_rxn_graph(
            reactants=reactants,
            products=products,
            mappings=mappings,
            device=None,
            has_bonds=has_bonds,
            reverse=False,
            reactant_only=False, 
            zero_fts=True,
            empty_graph_fts=None
        )
"""

        molecule_info_temp = {
            "reactants": {
                #"reactants": rxn.reactants,
                "init_reactants": rxn_copy.init_reactants,
                "has_bonds": has_bonds["reactants"]
            },
            "products": {
                #"products": rxn.products,
                "init_products": rxn_copy.init_products,
                "has_bonds": has_bonds["products"]
            },
            #"has_bonds": has_bonds,
            "mappings": mappings,
        }

        extra_info.append([])
        # extra_info.append(reaction_in_rxn_network.extra_info)
        label_list.append(dataset.labels[ind]["value"])
        reverse_list.append(dataset.labels[ind]["value_rev"])
        reaction_molecule_info.append(molecule_info_temp)
        has_bonds_list.append(has_bonds) # don't need to save
        reaction_indicies.append(int(rxn.id[0])) # need 
        #empty_reaction_graphs.append(empty_graph) # need 
        #empty_reaction_fts.append(empty_fts) # potentially source of bugginess
        mappings_list.append(mappings)


    print("...> writing molecules to lmdb")
    write_molecule_lmdb(
        indices=molecule_ind_list,
        graphs=dgl_graphs_serialized,
        pmgs=pmg_objects,
        charges=charge_set,
        ring_sizes=ring_size_set,
        elements=element_set,
        feature_info={
            "feature_size": feature_size,
            "feature_scaler_mean": feature_scaler_mean,
            "feature_scaler_std": feature_scaler_std,
        },
        #num_workers=1,
        lmdb_dir=lmdb_dir,
        lmdb_name="/molecule.lmdb",
    )
    print("...> writing reactions to lmdb")

    
    write_reaction_lmdb(
        indices=reaction_indicies,
        empty_reaction_graphs=empty_reaction_graphs,
        empty_reaction_fts=empty_reaction_fts,
        reaction_molecule_info=reaction_molecule_info,
        labels=label_list,
        reverse_labels=reverse_list,
        extra_info=extra_info,
        lmdb_dir=lmdb_dir,
        lmdb_name="/reaction.lmdb",
        mappings=mappings_list, 
        has_bonds=has_bonds_list, 
        global_values=global_dict,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="write to molecule or reaction lmdb dataset with multiprocess"
    )
    parser.add_argument(
        "--dataset_loc",
        type=str,
        help="location of json file containing dataset",
    )

    parser.add_argument(
        "-config_loc",
        type=str,
        help="location of json file containing config",
    )

    parser.add_argument(
        "-lmdb_dir",
        type=str,
        help="location of lmdb directory",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )

    parser.add_argument(
        "-workers",
        type=str,
        default="4",
        help="number of workers to use in conversion",
    )

    args = parser.parse_args()
    config_loc = args.config_loc
    dataset_loc = args.dataset_loc
    lmdb_dir = args.lmdb_dir
    workers = int(args.workers)
    debug = bool(args.debug)

    # read json
    with open(config_loc) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extra_keys = config["extra_features"]
    precision = config["precision"]

    if precision == "16" or precision == "32":
        precision = int(precision)

    extra_keys = config["extra_features"]

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(extra_keys),
        file=dataset_loc,
        target=config["target_var"],
        classifier=config["classifier"],
        classif_categories=config["classif_categories"],
        filter_species=config["filter_species"],
        filter_outliers=config["filter_outliers"],
        filter_sparse_rxns=False,
        debug=debug,
        extra_keys=extra_keys,
        extra_info=config["extra_info"],
    )

    construct_lmdb_and_save(dataset, lmdb_dir, workers=workers)
