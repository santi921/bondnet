import os
import numpy as np
import multiprocessing as mp
import dgl
import unittest
import torch

from bondnet.data.utils import find_rings, create_rxn_graph
from bondnet.dataset.utils import (
    clean,
    clean_op,
    divide_to_list,
)

from bondnet.dataset.lmdb import parallel2moleculelmdb, parallel2reactionlmdb

from bondnet.test_utils import get_test_reaction_network_data

torch.multiprocessing.set_sharing_strategy("file_system")


class TestLMDB(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.dataset = get_test_reaction_network_data()

        # List of Molecules
        self.dgl_graphs = []
        self.pmg_objects = []
        self.molecule_ind_list = []
        self.charge_set = set()
        self.ring_size_set = set()
        self.element_set = set()

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
        feats = batched_graph.ndata["feat"]
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
        parallel2moleculelmdb(
            self.molecule_ind_list,
            self.dgl_graphs,
            self.pmg_objects,
            self.charge_set,
            self.ring_size_set,
            self.element_set,
            num_workers=2,
            lmdb_dir="./test_mol/",
            lmdb_name="molcule.lmdb",
        )

    def test_write_reaction(self):
        parallel2reactionlmdb(
            self.reaction_indices,
            self.empty_reaction_graphs,
            self.empty_reaction_fts,
            self.reaction_molecule_info,
            self.label_list,
            self.reverse_list,
            self.extra_info,
            2,
            lmdb_dir="./test_mol/",
            lmdb_name="reaction.lmdb",
        )
