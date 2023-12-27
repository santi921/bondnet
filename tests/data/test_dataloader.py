"""
Do not assert feature and graph struct, which is handled by dgl.
Here we mainly test the correctness of batch.
"""

from pathlib import Path
import torch
import numpy as np
from bondnet.data.dataset import BondDataset, ReactionDataset, ReactionNetworkDataset
from bondnet.data.qm9 import QM9Dataset
from bondnet.data.dataloader import (
    DataLoaderBond,
    DataLoader,
    DataLoaderReaction,
    DataLoaderReactionNetwork,
)
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizerFull,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,

)

test_files = Path(__file__).parent.joinpath("testdata")


def test_dataloader_reaction_network():
    # TODO: use new grapher, reaction networks
    """ref_label_class = [0, 1]

    dataset = ReactionNetworkDataset(
        grapher=get_grapher_hetero(),
        molecules=test_files.joinpath("electrolyte_struct_rxn_ntwk_clfn.sdf"),
        labels=test_files.joinpath("electrolyte_label_rxn_ntwk_clfn.yaml"),
        extra_features=test_files.joinpath("electrolyte_feature_rxn_ntwk_clfn.yaml"),
        feature_transformer=False,
        label_transformer=False,
    )

    # batch size 1 case (exactly the same as test_dataset)
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=1, shuffle=False)
    for i, (graph, labels) in enumerate(data_loader):
        assert np.allclose(labels["value"], ref_label_class[i])

    # batch size 2 case
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=2, shuffle=False)
    for graph, labels in data_loader:
        assert np.allclose(labels["value"], ref_label_class)
    """

