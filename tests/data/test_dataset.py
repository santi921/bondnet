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



def test_hydro_reg(): # TODO
    pass


def test_mg_class():# TODO
    pass


def test_augment():# TODO
    pass
