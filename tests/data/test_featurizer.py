import numpy as np
import pandas as pd
from bondnet.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
    DistanceBins,
    RBF,
)
from bondnet.dataset.generalized import create_reaction_network_files_and_valid_rows
from bondnet.model.training_utils import get_grapher
from bondnet.data.utils import get_dataset_species


def get_data(
    extra_keys=[],
    species=["C", "F", "H", "N", "O", "Mg", "Li", "S", "Cl", "P", "O", "Br"],
):
    test_df_file = "./testdata/barrier_2.json"
    df = pd.read_json(test_df_file)

    grapher = get_grapher(features=extra_keys)
    # store feature name and size
    (
        molecules,
        raw_labels,
        extra_features,
    ) = create_reaction_network_files_and_valid_rows(
        test_df_file,
        bond_map_filter=False,
        target="ts",
        filter_species=[4, 6],
        classifier=False,
        debug=False,
        filter_outliers=False,
        categories=None,
        filter_sparse_rxn=False,
        feature_filter=True,
        extra_keys=extra_keys,
        extra_info=[],
    )
    # print("extra_features: ", extra_features)

    count = 0

    # feats_global = glob_feats
    if extra_keys is not None:
        extra_features = extra_features
    else:
        extra_features = [None] * len(molecules)

    graphs = []
    feat_name_list = []
    for ind, mol in enumerate(molecules):
        feats = extra_features[count]

        if mol is not None:
            g, feat_names = grapher.build_graph_and_featurize(
                mol,
                extra_feats_info=feats,
                dataset_species=species,
                ret_feat_names=True,
            )
            feat_name_list.append(feat_names)
            # add this for check purpose; some entries in the sdf file may fail
            g.graph_id = ind
        else:
            g = None
        graphs.append(g)
        count += 1
    return graphs, feat_name_list


def test_extra_atom_featurizer():
    extra_feats = [
        "esp_nuc",
        "esp_e",
        "esp_total",
    ]
    graphs, feats = get_data(extra_keys=extra_feats)
    [print(k, len(v[0])) for k, v in feats[0].items()]
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 23
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 7


def test_extra_bond_featurizer():
    extra_feats = [
        "esp_total",  # the atom esp is also added
        "bond_esp_total",  # this turns on the esp added to graphs
        "bond_length",  # this is a feature
        "indices_qtaim",  # this maps features to bonds
    ]

    graphs, feats = get_data(extra_keys=extra_feats)
    [print(k, len(v[0])) for k, v in feats[0].items()]
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 21
        elif k == "bond":
            assert len(v[0]) == 9
        elif k == "global":
            assert len(v[0]) == 7


