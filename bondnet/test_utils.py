import os
from rdkit import Chem
import torch
import dgl
import numpy as np
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from bondnet.core.molwrapper import create_wrapper_mol_from_atoms_and_bonds
from bondnet.core.molwrapper import create_rdkit_mol_from_mol_graph
from bondnet.core.reaction import Reaction
from bondnet.layer.utils import UnifySize
from bondnet.model.training_utils import get_grapher
from bondnet.data.dataset import ReactionNetworkDatasetGraphs

import torch


def test_unify_size():
    in_feats = {"a": 2, "b": 3}
    out_feats = 4
    us = UnifySize(in_feats, out_feats)

    feats = {"a": torch.zeros(2), "b": torch.zeros(3)}
    feats = us(feats)
    for _, v in feats.items():
        assert v.shape[0] == out_feats


def create_LiEC_pymatgen_mol():
    """
            O(3) -- Li(6)
            ||
             C(1)
           /   \
          O(0)  O(4)
          |     |
        C(2) --- C(5)



    """
    atoms = ["O", "C", "C", "O", "O", "C", "Li", "H", "H", "H", "H"]
    coords = [
        [0.3103, -1.1776, -0.3722],
        [-0.6822, -0.5086, 0.3490],
        [1.5289, -0.4938, -0.0925],
        [-1.9018, -0.6327, -0.0141],
        [-0.2475, 0.9112, 0.3711],
        [1.1084, 0.9722, -0.0814],
        [-2.0519, 1.1814, -0.2310],
        [2.2514, -0.7288, -0.8736],
        [1.9228, -0.8043, 0.8819],
        [1.1406, 1.4103, -1.0835],
        [1.7022, 1.5801, 0.6038],
    ]
    charge = 0

    m = Molecule(atoms, coords, charge)

    return m


def create_LiEC_mol_graph():
    bonds = [
        (0, 2),
        (0, 1),
        (2, 5),
        (2, 8),
        (4, 1),
        (5, 4),
        (5, 10),
        (7, 2),
        (9, 5),
        (3, 6),
        (3, 1),
    ]
    bonds = {b: None for b in bonds}

    mol = create_LiEC_pymatgen_mol()
    mol_graph = MoleculeGraph.with_edges(mol, bonds)

    return mol_graph


def create_LiEC_rdkit_mol():
    mol_graph = create_LiEC_mol_graph()
    mol, bond_type = create_rdkit_mol_from_mol_graph(mol_graph, force_sanitize=True)
    return mol


def create_C2H4O1():
    r"""
                O(0)
               / \
              /   \
      H(1)--C(2)--C(3)--H(4)
             |     |
            H(5)  H(6)
    """
    species = ["O", "H", "C", "C", "H", "H", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (2, 5), (3, 6)]
    m = create_wrapper_mol_from_atoms_and_bonds(
        species=species,
        coords=coords,
        bonds=bonds,
        charge=charge,
        free_energy=0.0,
        identifier="mol",
    )

    return m


def create_symmetric_molecules():
    r"""
    Create a list of molecules, which can form reactions where the reactant is symmetric.

    m0: charge 0
    H(0)---C(1)---H(2)
           / \
          /   \
       O(3)---O(4)

    m1: charge 0
    H(0)---C(1)---H(2)
           / \
          /   \
       O(3)   O(4)

    m2: charge 0
    H(0)---C(1)
           /  \
          /   \
       O(2)---O(3)

    m3: charge -1
    H(0)---C(1)
           /  \
          /   \
       O(2)---O(3)

    m4: charge 0
    H

    m5: charge 1
    H

    m6: charge -1
    H

    m7: charge 0
    H--H

    The below reactions exists (w.r.t. graph connectivity and charge):

    A -> B:
    m0 -> m1        C1H2O2 (0) -> C1H2O2 (0) + H (0)
    A -> B+C:
    m0 -> m2 + m4   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (1,2)
    m0 -> m3 + m5   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (1,2)
    m0 -> m2 + m4   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (0,1)
    m0 -> m3 + m5   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (0,1)
    m7 -> m4 + m4   H2 (0) -> H (0) + H (0)
    m7 -> m5 + m6   H2 (0) -> H (1) + H (-1)
    """

    mols = []

    # m0, charge 0
    species = ["H", "C", "H", "O", "O"]
    coords = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 4)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=0.0,
            identifier="m0",
        )
    )

    # m1, charge 0
    bonds = [(0, 1), (1, 2), (1, 3), (1, 4)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=1.0,
            identifier="m1",
        )
    )

    # m2, charge 0
    species = ["H", "C", "O", "O"]
    coords = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 1), (1, 2), (1, 3), (2, 3)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=2.0,
            identifier="m2",
        )
    )

    # m3, charge -1
    charge = -1
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=3.0,
            identifier="m3",
        )
    )

    # m4, H, charge 0
    species = ["H"]
    coords = [[1.0, 0.0, 0.0]]
    charge = 0
    bonds = []
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=4.0,
            identifier="m4",
        )
    )

    # m5, H, charge 1
    charge = 1
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=5.0,
            identifier="m5",
        )
    )

    # m6, H, charge -1
    charge = -1
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=6.0,
            identifier="m6",
        )
    )

    # m7, H2, charge 0
    species = ["H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    charge = 0
    bonds = [(0, 1)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=charge,
            free_energy=7.0,
            identifier="m7",
        )
    )

    return mols


def create_nonsymmetric_molecules():
    r"""
    Create a list of molecules, which can form reactions where the reactant is
    nonsymmetric.

    m0: charge 0
        C(0)
     0 /  \  1
      /____\     3
    O(1) 2 N(2)---H(3)

    m1: charge 0 (note the atom index order between this and m0)
        C(0)
          \ 0
       ____\      2
    O(2) 1  N(1)---H(3)

    m2: charge 0 (note the atom index order between this and m0)
        C(0)
    1  /  \ 0
      /____\
    O(2) 2  N(1)

    m3: charge 0
    H(0)


    The below reactions exists (w.r.t. graph connectivity and charge):

    m0 -> m1    CHNO (0) -> CHNO (0)
    m0 -> m2 + m3   CHNO (0) -> CNO (0) + H (0)
    """

    mols = []

    # m0
    species = ["C", "O", "N", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ]
    bonds = [(0, 1), (0, 2), (1, 2), (2, 3)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=0,
            free_energy=0.0,
            identifier="m0",
        )
    )

    # m1
    species = ["C", "N", "O", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ]
    bonds = [(0, 1), (1, 2), (1, 3)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=0,
            free_energy=1.0,
            identifier="m1",
        )
    )

    # m2, m0 without H
    species = ["C", "N", "O"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    bonds = [(0, 1), (0, 2), (1, 2)]
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=0,
            free_energy=2.0,
            identifier="m2",
        )
    )

    # m3, H
    species = ["H"]
    coords = [[1.0, 0.0, 1.0]]
    bonds = []
    mols.append(
        create_wrapper_mol_from_atoms_and_bonds(
            species=species,
            coords=coords,
            bonds=bonds,
            charge=0,
            free_energy=3.0,
            identifier="m3",
        )
    )

    return mols


def create_reactions_symmetric_reactant():
    """
    Create a list of reactions, using the mols returned by `create_symmetric_molecules`.
    """
    mols = create_symmetric_molecules()
    A2B = [Reaction(reactants=[mols[0]], products=[mols[1]], broken_bond=(3, 4))]
    A2BC = [
        Reaction(reactants=[mols[0]], products=[mols[2], mols[4]], broken_bond=(1, 2)),
        Reaction(reactants=[mols[0]], products=[mols[3], mols[5]], broken_bond=(1, 2)),
        Reaction(reactants=[mols[0]], products=[mols[2], mols[4]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[0]], products=[mols[3], mols[5]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[7]], products=[mols[4], mols[4]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[7]], products=[mols[5], mols[6]], broken_bond=(0, 1)),
    ]

    return A2B, A2BC


def create_reactions_nonsymmetric_reactant():
    """
    Create a list of reactions, using the mols returned by
    `create_nonsymmetric_molecules`.
    """
    mols = create_nonsymmetric_molecules()
    A2B = [Reaction(reactants=[mols[0]], products=[mols[1]], broken_bond=(0, 1))]
    A2BC = [
        Reaction(reactants=[mols[0]], products=[mols[2], mols[3]], broken_bond=(2, 3))
    ]

    return A2B, A2BC


def make_a_mol():
    sdf = """5d1a79e59ab9e0c05b1de572
 OpenBabel11151914373D

  7  7  0  0  0  0  0  0  0  0999 V2000
    0.0852   -0.2958   -0.5026 O   0  3  0  0  0  0  0  0  0  0  0  0
    1.4391   -0.0921   -0.0140 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1032   -1.4653    0.0152 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9476   -2.1383   -1.1784 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.3604   -1.7027   -1.7840 Li  0  5  0  0  0  0  0  0  0  0  0  0
   -0.3721    0.5500   -0.5368 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4039    0.3690    0.9792 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  2  7  1  0  0  0  0
  4  3  1  0  0  0  0
  5  4  1  0  0  0  0
  6  1  1  0  0  0  0
M  CHG  2   1   1   5  -1
M  ZCH  2   1   0   5   0
M  ZBO  1   2   0
M  END
$$$$
    """
    return Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=False)


def make_hetero(num_atoms, num_bonds, a2b, b2a, self_loop=False):
    """
    Create a hetero graph and create features.
    A global node is connected to all atoms and bonds.

    Atom features are:
    [[0,1],
     [2,3],
     .....]

    Bond features are:
    [[0,1,2],
     [3,4,5],
     .....]

    Global features are:
    [[0,1,2,3]]
    """
    if num_bonds == 0:
        # create a fake bond and create an edge atom->bond
        num_bonds = 1
        a2b = [(0, 0)]
        b2a = [(0, 0)]

    edge_dict = {
        ("atom", "a2b", "bond"): a2b,
        ("bond", "b2a", "atom"): b2a,
        ("atom", "a2g", "global"): [(i, 0) for i in range(num_atoms)],
        ("global", "g2a", "atom"): [(0, i) for i in range(num_atoms)],
        ("bond", "b2g", "global"): [(i, 0) for i in range(num_bonds)],
        ("global", "g2b", "bond"): [(0, i) for i in range(num_bonds)],
    }

    if self_loop:
        a2a = [(i, i) for i in range(num_atoms)]
        b2b = [(i, i) for i in range(num_bonds)]
        g2g = [(0, 0)]
        edge_dict.update(
            {
                ("atom", "a2a", "atom"): a2a,
                ("bond", "b2b", "bond"): b2b,
                ("global", "g2g", "global"): g2g,
            }
        )
    g = dgl.heterograph(edge_dict)

    feats_size = {"atom": 2, "bond": 3, "global": 4}
    feats = {}
    for ntype, size in feats_size.items():
        num_node = g.number_of_nodes(ntype)
        ft = torch.tensor(
            np.arange(num_node * size).reshape(num_node, size), dtype=torch.float32
        )
        g.nodes[ntype].data.update({"feat": ft})
        feats[ntype] = ft

    return g, feats


def make_hetero_CH2O(self_loop=False):
    r"""
            O (0)
            || (0)
            C (1)
        /(1)  \ (2)
        H (2)  H (3)

    atom features:
    [[0,1],
     [2,3],
     [4,5],
     [6,7],]

    bond features:
    [[0,1,2],
     [3,4,5],
     [6,7,8]]

    global features:
    [[0,1,2,3]]
    """

    return make_hetero(
        num_atoms=4,
        num_bonds=3,
        a2b=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
        b2a=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)],
        self_loop=self_loop,
    )


def make_hetero_CHO():
    r"""
            O (0)
            || (0)
            C (1)
        /(1)
        H (2)

    atom features:
    [[0,1],
     [2,3],
     [4,5],]

    bond features:
    [[0,1,2],
     [3,4,5]]

    global features:
    [[0,1,2,3]]
    """

    return make_hetero(
        num_atoms=3,
        num_bonds=2,
        a2b=[(0, 0), (1, 0), (1, 1), (2, 1)],
        b2a=[(0, 0), (0, 1), (1, 1), (1, 2)],
    )


def make_hetero_H():
    r"""
    H

    atom features:
    [[0,1]]

    bond features (a factitious bond is created):
    [[0,1,2]]

    global features:
    [[0,1,2,3]]
    """

    return make_hetero(num_atoms=1, num_bonds=0, a2b=[], b2a=[])


def make_batched_hetero_CH2O(size=3):
    graphs = [make_hetero_CH2O()[0] for i in range(size)]
    g = dgl.batch(graphs)
    feats = {t: g.nodes[t].data["feat"] for t in ["atom", "bond", "global"]}
    return g, feats


def make_batched_hetero_forming_reaction():
    graphs = [
        make_hetero_CH2O()[0],
        make_hetero_CHO()[0],
        make_hetero_H()[0],
        make_hetero_CH2O()[0],
        make_hetero_CHO()[0],
        make_hetero_H()[0],
    ]
    g = dgl.batch(graphs)
    feats = {t: g.nodes[t].data["feat"] for t in ["atom", "bond", "global"]}
    return g, feats


def make_homo_CH2O():
    r"""
    Make a bidirected homograph for COHH and featurize it
            O (0)
            || (0)
            C (1)
        /(1)  \ (2)
        H (2)  H (3)
    A global node u is attached to all atoms and bonds.
    """

    src = [0, 1, 1, 1, 2, 3]
    dst = [1, 0, 2, 3, 1, 1]
    g = dgl.graph((src, dst))

    feats = {}
    N = 4
    size = 2
    ft = torch.tensor(np.arange(N * size).reshape(N, size), dtype=torch.float32)
    g.ndata.update({"feat": ft})
    feats["node"] = ft

    N = 6
    size = 3
    ft = torch.tensor(np.arange(N * size).reshape(N, size), dtype=torch.float32)
    g.edata.update({"feat": ft})
    feats["edge"] = ft

    return g, feats


def get_test_reaction_network_data(dir=None):
    config = {
        "debug": False,
        "classifier": False,
        "classif_categories": 3,
        "cat_weights": [1.0, 1.0, 1.0],
        "extra_features": ["bond_length"],
        "extra_info": [],
        "filter_species": [3, 5],
        "precision": "bf16",
        "on_gpu": True,
        "target_var": "ts",
        "target_var_transfer": "diff",
        "transfer": False,
        "filter_outliers": True,
    }
    # get current directory
    if dir is None:
        dataset_loc = "./testdata/barrier_100.json"
    else:
        dataset_loc = dir

    on_gpu = config["on_gpu"]
    extra_keys = config["extra_features"]
    debug = config["debug"]
    precision = config["precision"]
    if precision == "16" or precision == "32":
        precision = int(precision)
    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

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
    return dataset


def get_defaults():
    config = {
        "model": {
            "conv": "GatedGCNConv",
            "augment": False,
            "classifier": False,
            "classif_categories": 3,
            "cat_weights": [1.0, 1.0, 1.0],
            "embedding_size": 24,
            "epochs": 100,
            "extra_features": ["bond_length"],
            "extra_info": [],
            "filter_species": [3, 5],
            "fc_activation": "ReLU",
            "fc_batch_norm": True,
            "fc_dropout": 0.2,
            "fc_hidden_size_1": 256,
            "fc_hidden_size_shape": "flat",
            "fc_num_layers": 1,
            "gated_activation": "ReLU",
            "gated_batch_norm": False,
            "gated_dropout": 0.1,
            "gated_graph_norm": False,
            "gated_hidden_size_1": 512,
            "gated_hidden_size_shape": "flat",
            "gated_num_fc_layers": 1,
            "gated_num_layers": 2,
            "gated_residual": True,
            "learning_rate": 0.003,
            "precision": "bf16",
            "loss": "mse",
            "num_lstm_iters": 3,
            "num_lstm_layers": 1,
            "restore": False,
            "weight_decay": 0.0,
            "max_epochs": 1000,
            "max_epochs_transfer": 10,
            "transfer": False,
            "filter_outliers": True,
            "freeze": True,
            "reactant_only": False,
        }
    }
    # config = "./settings.json"
    # config = json.load(open(config, "r"))
    return config
