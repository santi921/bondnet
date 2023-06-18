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
)

test_files = Path(__file__).parent.joinpath("testdata")


def get_grapher_hetero():
    # TODO: use new grapher
    return HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizerFull(),
        bond_featurizer=BondAsNodeFeaturizerFull(),
        global_featurizer=GlobalFeaturizer(),
        self_loop=True,
    )


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


def test_dataloader_reaction_network_precompute():
    # TODO: test precompute variant
    pass
    extra_keys = []

    """dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher(extra_keys),
        file=dataset_loc,
        target=config["target_var"],
        classifier=config["classifier"],
        classif_categories=config["classif_categories"],
        filter_species=config["filter_species"],
        filter_outliers=config["filter_outliers"],
        filter_sparse_rxns=False,
        debug=debug,
        device=device,
        extra_keys=extra_keys,
        extra_info=config["extra_info"],
    )"""
