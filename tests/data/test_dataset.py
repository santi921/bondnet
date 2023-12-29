import numpy as np
from pathlib import Path
from bondnet.data.dataset import ReactionNetworkDataset
from bondnet.data.grapher import HeteroCompleteGraphFromMolWrapper
from bondnet.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
)
import torch


test_files = Path(__file__).parent.joinpath("testdata")



def test_hydro_reg(): # TODO
    pass


def test_mg_class():# TODO
    pass


def test_augment():# TODO
    pass
