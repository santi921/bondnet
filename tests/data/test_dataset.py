import numpy as np
from pathlib import Path
from bondnet.data.dataset import (
    BondDataset,
    BondDatasetClassification,
    MoleculeDataset,
    ReactionDataset,
    ReactionNetworkDataset,
)
from bondnet.data.qm9 import QM9Dataset
from bondnet.data.grapher import HeteroMoleculeGraph, HomoCompleteGraph
from bondnet.data.featurizer import (
    AtomFeaturizerFull,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
import torch


test_files = Path(__file__).parent.joinpath("testdata")


def get_grapher_hetero():
    return HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizerFull(),
        bond_featurizer=BondAsNodeFeaturizerFull(),
        global_featurizer=GlobalFeaturizer(),
        self_loop=True,
    )


def test_electrolyte_bond_label():
    def assert_label(lt):
        ref_label_energy = [[0.1, 0.2, 0.3], [0.4, 0.5]]
        ref_label_index = [[1, 3, 5], [0, 3]]

        if lt:
            energies = torch.tensor(np.concatenate(ref_label_energy))
            mean = float(torch.mean(energies))
            std = float(torch.std(energies))
            ref_label_energy = [(np.asarray(a) - mean) / std for a in ref_label_energy]
            ref_std = [[std] * len(x) for x in ref_label_energy]
            ref_mean = [[mean] * len(x) for x in ref_label_energy]

        dataset = BondDataset(
            grapher=get_grapher_hetero(),
            molecules=test_files.joinpath("electrolyte_struct_bond.sdf"),
            labels=test_files.joinpath("electrolyte_label_bond.yaml"),
            extra_features=test_files.joinpath("electrolyte_feature_bond.yaml"),
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label = dataset[i]
            assert np.allclose(label["value"], ref_label_energy[i])
            assert np.array_equal(label["bond_index"], ref_label_index[i])
            if lt:
                assert np.allclose(label["scaler_mean"], ref_mean[i])
                assert np.allclose(label["scaler_stdev"], ref_std[i])
            else:
                assert "scaler_stedv" not in label

    assert_label(False)
    assert_label(True)


def test_electrolyte_bond_label_classification():
    ref_label_class = [0, 1]
    ref_label_indicators = [1, 2]

    dataset = BondDatasetClassification(
        grapher=get_grapher_hetero(),
        molecules=test_files.joinpath("electrolyte_struct_bond.sdf"),
        labels=test_files.joinpath("electrolyte_label_bond_clfn.txt"),
        extra_features=test_files.joinpath("electrolyte_feature_bond.yaml"),
        feature_transformer=True,
    )

    size = len(dataset)
    assert size == 2

    for i in range(size):
        _, label = dataset[i]
        assert label["value"] == ref_label_class[i]
        assert label["indicator"] == ref_label_indicators[i]


def test_electrolyte_molecule_label():
    def assert_label(lt):
        ref_labels = np.asarray([[-0.941530613939904], [-8.91357537335352]])
        natoms = np.asarray([[2], [5]])

        if lt:
            ref_labels /= natoms
            ref_ts = natoms

        dataset = MoleculeDataset(
            grapher=get_grapher_homo(),
            molecules=test_files.joinpath("electrolyte_struct_mol.sdf"),
            labels=test_files.joinpath("electrolyte_label_mol.csv"),
            properties=["atomization_energy"],
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label = dataset[i]
            assert np.allclose(label["value"], ref_labels[i])
            if lt:
                assert np.allclose(label["scaler_stdev"], ref_ts[i])
            else:
                assert "scaler_stdev" not in label

    assert_label(False)
    assert_label(True)


def test_hydro_reg(): # TODO
    pass


def test_mg_class():# TODO
    pass


def test_augment():# TODO
    pass
