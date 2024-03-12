import os
import numpy as np
import multiprocessing as mp
import dgl
import unittest
import torch

from bondnet.data.utils import find_rings, create_rxn_graph
from bondnet.dataset.utils import (
    clean,
    clean_op
)
from bondnet.data.lmdb import (
    LmdbMoleculeDataset,
    LmdbReactionDataset
)

from bondnet.data.lmdb import write_molecule_lmdb, write_reaction_lmdb
from bondnet.data.reaction_network import ReactionNetworkLMDB, ReactionLMDB
from bondnet.data.dataset import ReactionNetworkLMDBDataset, ReactionLMDBDataset
from bondnet.test_utils import get_test_reaction_network_data

torch.multiprocessing.set_sharing_strategy("file_system")


class TestLMDB(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.dataset = get_test_reaction_network_data(allowed_charges=[0, 1, 2, -1, -2])

        # List of Molecules
        self.dgl_graphs = []
        self.pmg_objects = []
        self.molecule_ind_list = []
        self.charge_set = set()
        self.ring_size_set = set()
        self.element_set = set()

        feature_size = self.dataset._feature_size
        feature_name = self.dataset._feature_name
        feature_scaler_mean = self.dataset._feature_scaler_mean
        feature_scaler_std = self.dataset._feature_scaler_std
        label_scaler_mean = self.dataset._label_scaler_mean
        label_scaler_std = self.dataset._label_scaler_std
        dtype = self.dataset.dtype

        self.global_dict = {
            "feature_size": feature_size,
            "mean": label_scaler_mean,
            "std": label_scaler_std,
            "feature_name": feature_name,
            "feature_scaler_mean": feature_scaler_mean,
            "feature_scaler_std": feature_scaler_std,
            "dtype": dtype
        }
        for ind, molecule_in_rxn_network in enumerate(
            self.dataset.reaction_network.molecule_wrapper
        ):
            #    pmg mol retrieval
            self.pmg_objects.append(molecule_in_rxn_network.pymatgen_mol)
            #    molecule index in rxn network
            # this would just be given by the index in HiPRGen anyways
            self.molecule_ind_list.append(ind)

            formula = molecule_in_rxn_network.pymatgen_mol.composition.formula.split()
            elements = [clean(x) for x in formula]
            atom_num = np.sum(np.array([int(clean_op(x)) for x in formula]))
            self.element_set.update(elements)

            charge = molecule_in_rxn_network.pymatgen_mol.charge
            self.charge_set.add(charge)
            bond_list = [
                [i[0], i[1]] for i in molecule_in_rxn_network.mol_graph.graph.edges
            ]
            cycles = find_rings(atom_num, bond_list, edges=False)
            ring_len_list = [len(i) for i in cycles]
            self.ring_size_set.update(ring_len_list)

        for ind, molecule_in_rxn_network in enumerate(
            self.dataset.reaction_network.molecules
        ):
            self.dgl_graphs.append(molecule_in_rxn_network)

        batched_graph = dgl.batch(self.dgl_graphs)
        feats = batched_graph.ndata["ft"]
        for nt, ft in feats.items():
            batched_graph.nodes[nt].data.update({"ft": ft})

        self.reaction_molecule_info = []
        self.label_list = []
        self.reverse_list = []
        self.reaction_molecule_info_list = []
        self.empty_reaction_graphs = []
        self.empty_reaction_fts = []
        self.extra_info = []
        self.graphs = dgl.unbatch(batched_graph)
        self.mappings_list = []
        self.has_bonds_list = []

        # print(graphs)
        for ind, rxn in enumerate(self.dataset.reaction_network.reactions):
            # print(graphs[0])
            self.extra_info.append([])
            # extra_info.append(reaction_in_rxn_network.extra_info)
            self.label_list.append(self.dataset.labels[ind]["value"])
            self.reverse_list.append(self.dataset.labels[ind]["value_rev"])

            mappings = {
                "bond_map": rxn.bond_mapping,
                "atom_map": rxn.atom_mapping,
                "total_bonds": rxn.total_bonds,
                "total_atoms": rxn.total_atoms,
                "num_bonds_total": rxn.num_bonds_total,
                "num_atoms_total": rxn.num_atoms_total,
            }
            self.mappings_list.append(mappings)

            molecule_info_temp = {
                "reactants": {
                    #"reactants": rxn.reactants,
                    #"atom_map": rxn.atom_mapping[0],
                    #"bond_map": rxn.bond_mapping[0],
                    "init_reactants": rxn.init_reactants,
                },
                "products": {
                    #"products": rxn.products,
                    #"atom_map": rxn.atom_mapping[1],
                    #"bond_map": rxn.bond_mapping[1],
                    "init_products": rxn.init_products,
                },
            }

            self.reaction_molecule_info.append(molecule_info_temp)

            #### taken from create_rxn_graph in bondnet
            reactants = []
            products = []
            for i in rxn.reactants:
                reactants.append(self.graphs[i])
            for i in rxn.products:
                products.append(self.graphs[i])

            has_bonds = {
                "reactants": [
                    True if len(mp) > 0 else False for mp in rxn.bond_mapping[0]
                ],
                "products": [
                    True if len(mp) > 0 else False for mp in rxn.bond_mapping[1]
                ],
            }
            self.has_bonds_list.append(has_bonds)

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

            self.empty_reaction_graphs.append(empty_graph)
            self.empty_reaction_fts.append(empty_fts)

        self.reaction_indices = []
        for i in range(len(self.label_list)):
            self.reaction_indices.append(i)


    def test_write_molecule(self):
        write_molecule_lmdb(
            indices=self.molecule_ind_list,
            graphs=self.dgl_graphs,
            pmgs=self.pmg_objects,
            charges=self.charge_set,
            ring_sizes=self.ring_size_set,
            feature_info=self.dataset._feature_size,
            elements=self.element_set,
            lmdb_dir="./test_mol/",
            lmdb_name="molecule.lmdb",
        )


    def test_write_reaction(self):
        write_reaction_lmdb(
            indices=self.reaction_indices,
            empty_reaction_graphs=self.empty_reaction_graphs,
            empty_reaction_fts=self.empty_reaction_fts,
            reaction_molecule_info=self.reaction_molecule_info,
            labels=self.label_list,
            reverse_labels=self.reverse_list,
            extra_info=self.extra_info,
            lmdb_dir="./test_mol/",
            lmdb_name="reaction.lmdb",
            mappings=self.mappings_list,
            has_bonds=self.has_bonds_list,
            global_values=self.global_dict

        )


    def test_featurization_hiprgen(self): 
        config = {
            "src": "./testdata/lmdb_dev/mol.lmdb"
        }
        config_rxn = {
            "src": "./testdata/lmdb_dev/reaction.lmdb"
        }

        mol = LmdbMoleculeDataset(config=config)
        reaction = LmdbReactionDataset(config=config_rxn)
        rxn_ntwk = ReactionNetworkLMDB(mol, reaction)
        dataset = ReactionNetworkLMDBDataset(rxn_ntwk)
        features = rxn_ntwk.reactions.feature_name
        assert "charge one hot" in features["global"]


    def test_featurization_hiprgen_reaction(self): 
        config = {
            "src": "./testdata/lmdb_dev/mol.lmdb"
        }
        config_rxn = {
            "src": "./testdata/lmdb_dev/reaction.lmdb"
        }

        mol = LmdbMoleculeDataset(config=config)
        reaction = LmdbReactionDataset(config=config_rxn)
        rxn_ntwk = ReactionLMDB(mol, reaction)
        dataset = ReactionLMDBDataset(rxn_ntwk)
        features = rxn_ntwk.reactions.feature_name
        assert "charge one hot" in features["global"]



