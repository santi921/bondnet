"""
Do not assert feature and graph struct, which is handled by dgl.
Here we mainly test the correctness of batch.
"""

from pathlib import Path
import torch
import numpy as np
from bondnet.data.dataloader import DataLoaderReaction
from bondnet.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
)
from bondnet.test_utils import get_test_reaction_data

test_files = Path(__file__).parent.joinpath("testdata")




def test_dataloader_reaction():
    
    rxn_data = get_test_reaction_data(dir = "./testdata/barrier_2.json")
    
    ref_label_class = [-0.7071, 0.7071]

    # batch size 1 case (exactly the same as test_dataset)
    data_loader = DataLoaderReaction(rxn_data, batch_size=1, shuffle=False)
    for i, (graph, labels, batch_data) in enumerate(data_loader):
        
        assert np.allclose(labels["value"], ref_label_class[i])

    ref_label_class = [[-0.7071], [0.7071]]
    # batch size 2 case
    data_loader = DataLoaderReaction(rxn_data, batch_size=2, shuffle=False)
    for (graph, labels, batch_data) in data_loader:
        assert np.allclose(labels["value"].tolist(), [ref_label_class], atol=1e-3)
    


