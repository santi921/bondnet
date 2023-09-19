import argparse
import torch
import json
import numpy as np
import argparse
import dgl

import torch
from torch.utils.data import random_split
import torch.multiprocessing

from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.utils import seed_torch
from bondnet.data.utils import find_rings
from bondnet.model.training_utils import get_grapher
from bondnet.dataset.utils import (
    clean,
    clean_op,
)

from bondnet.data.lmdb import (
    parallel2reactionlmdb,
    parallel2moleculelmdb,
)


torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
seed_torch()
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def construct_lmdb_and_save(dataset, lmdb_dir, workers=8):
    # List of Molecules
    dgl_graphs = []
    pmg_objects = []
    molecule_ind_list = []
    charge_set = set()
    ring_size_set = set()
    element_set = set()

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
        dgl_graphs.append(molecule_in_rxn_network)

    batched_graph = dgl.batch(dgl_graphs)
    feats = batched_graph.ndata["feat"]
    for nt, ft in feats.items():
        batched_graph.nodes[nt].data.update({"ft": ft})
    graphs = dgl.unbatch(batched_graph)
    from bondnet.data.utils import create_rxn_graph

    extra_info = []
    reaction_molecule_info = []
    label_list = []
    reverse_list = []
    reaction_molecule_info_list = []
    empty_reaction_graphs = []
    empty_reaction_fts = []
    reaction_indicies = []

    for ind, rxn in enumerate(dataset.reaction_network.reactions):
        extra_info.append([])
        # extra_info.append(reaction_in_rxn_network.extra_info)
        label_list.append(dataset.labels[ind]["value"])
        reverse_list.append(dataset.labels[ind]["value_rev"])

        mappings = {
            "bond_map": rxn.bond_mapping,
            "atom_map": rxn.atom_mapping,
            "total_bonds": rxn.total_bonds,
            "total_atoms": rxn.total_atoms,
            "num_bonds_total": rxn.num_bonds_total,
            "num_atoms_total": rxn.num_atoms_total,
        }

        molecule_info_temp = {
            "reactants": {
                "molecule_index": rxn.reactants,
                "atom_map": rxn.atom_mapping[0],
                "bond_map": rxn.bond_mapping[0],
            },
            "products": {
                "molecule_index": rxn.products,
                "atom_map": rxn.atom_mapping[1],
                "bond_map": rxn.bond_mapping[1],
            },
        }

        reaction_molecule_info.append(molecule_info_temp)

        #### taken from create_rxn_graph in bondnet
        reactants = [graphs[i] for i in rxn.reactants]
        products = [graphs[i] for i in rxn.products]

        has_bonds = {
            "reactants": [True if len(mp) > 0 else False for mp in rxn.bond_mapping[0]],
            "products": [True if len(mp) > 0 else False for mp in rxn.bond_mapping[1]],
        }
        if len(has_bonds["reactants"]) != len(reactants) or len(
            has_bonds["products"]
        ) != len(products):
            print("unequal mapping & graph len")

        empty_graph, empty_fts = create_rxn_graph(
            reactants=reactants,
            products=products,
            mappings=mappings,
            device=None,
            has_bonds=has_bonds,
            reverse=False,
        )

        reaction_indicies.append(rxn.id)
        empty_reaction_graphs.append(empty_graph)
        empty_reaction_fts.append(empty_fts)

    print("...> writing molecules to lmdb")
    parallel2moleculelmdb(
        molecule_ind_list,
        dgl_graphs,
        pmg_objects,
        charge_set,
        ring_size_set,
        element_set,
        num_workers=workers,
        lmdb_dir=lmdb_dir,
        lmdb_name="molcule.lmdb",
    )
    print("...> writing reactions to lmdb")
    parallel2reactionlmdb(
        reaction_indicies,
        empty_reaction_graphs,
        empty_reaction_fts,
        reaction_molecule_info,
        label_list,
        reverse_list,
        extra_info,
        num_workers=workers,
        lmdb_dir=lmdb_dir,
        lmdb_name="reaction.lmdb",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="write to molecule or reaction lmdb dataset with multiprocess"
    )
    parser.add_argument(
        "-json_loc",
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

    args = parser.parse_args()
    config_loc = args.config_loc
    dataset_loc = args.json_loc
    lmdb_dir = args.lmdb_dir
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

    construct_lmdb_and_save(dataset, lmdb_dir, workers=16)
