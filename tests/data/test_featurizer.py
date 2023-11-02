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
    global_feats=["charge"],
    test_df_file="./testdata/barrier_2.json",
    target="ts",
):
    grapher = get_grapher(
        features=extra_keys, allowed_charges=None, global_feats=global_feats
    )
    # store feature name and size
    (
        molecules,
        raw_labels,
        extra_features,
        reactions,
    ) = create_reaction_network_files_and_valid_rows(
        test_df_file,
        bond_map_filter=False,
        target=target,
        filter_species=[4, 6],
        classifier=False,
        debug=False,
        filter_outliers=False,
        categories=None,
        filter_sparse_rxn=False,
        feature_filter=True,
        extra_keys=extra_keys,
        extra_info=[],
        return_reactions=True,
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
                element_set=species,
                ret_feat_names=True,
            )
            feat_name_list.append(feat_names)
            # add this for check purpose; some entries in the sdf file may fail
            g.graph_id = ind
        else:
            g = None
        graphs.append(g)
        count += 1
    # print("feature names: ", feat_name_list)
    return graphs, feat_name_list, reactions


def test_extra_atom_featurizer():
    extra_feats = [
        "esp_nuc",
        "esp_e",
        "esp_total",
    ]
    _, feats, _ = get_data(extra_keys=extra_feats)
    [print(k, len(v[0])) for k, v in feats[0].items()]
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 23
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 3


def test_extra_bond_featurizer():
    extra_feats = [
        "esp_total",  # the atom esp is also added
        "bond_esp_total",  # this turns on the esp added to graphs
        "bond_length",  # this is a feature
        "indices_qtaim",  # this maps features to bonds
    ]

    _, feats, _ = get_data(extra_keys=extra_feats)
    [print(k, len(v[0])) for k, v in feats[0].items()]
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 21
        elif k == "bond":
            assert len(v[0]) == 9
        elif k == "global":
            assert len(v[0]) == 3


def test_extra_global_featurizer():
    # test rishabh's func group
    extra_feats = [
        "functional_group_reacted",
    ]
    _, feats, _ = get_data(
        test_df_file="./testdata/hydro_funct.json",
        extra_keys=extra_feats,
        global_feats=["functional_group_reacted", "charge"],
        target="dG_sp",
    )
    # [print(k, len(v[0])) for k, v in feats[0].items()]
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 20
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 14

    # test manual global feats
    # test rishabh's func group
    extra_feats = [
        "func_group",
    ]
    _, feats, _ = get_data(
        test_df_file="./testdata/hydro_funct.json",
        extra_keys=extra_feats,
        global_feats=["func_group", "charge"],
        target="dG_sp",
    )
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 20
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 4


def test_atom_feat_mapping():
    extra_feats = [
        "esp_total",  # the atom esp is also added
        "indices_qtaim",  # this maps features to bonds
    ]

    df = pd.read_json("./testdata/barrier_2.json")
    reactant_qtaim_inds = df["extra_feat_bond_reactant_indices_qtaim"].tolist()
    product_qtaim_inds = df["extra_feat_bond_product_indices_qtaim"].tolist()
    reactant_bond_qtaim_esp = df["extra_feat_bond_reactant_esp_total"].tolist()
    product_bond_qtaim_esp = df["extra_feat_bond_product_esp_total"].tolist()
    reactant_atom_qtaim_esp = df["extra_feat_atom_reactant_esp_total"].tolist()
    product_atom_qtaim_esp = df["extra_feat_atom_product_esp_total"].tolist()
    graphs, feats, reactions = get_data(
        extra_keys=extra_feats, test_df_file="./testdata/barrier_2.json"
    )

    for i in range(len(reactions)):
        esp_tensor_reactant = graphs[2 * i].ndata["feat"]["atom"][:, -1]
        esp_tensor_prod = graphs[2 * i + 1].ndata["feat"]["atom"][:, -1]
        reactant_atom_map = reactions[i]._atom_mapping[0][0]
        product_atom_map = reactions[i]._atom_mapping[1][0]
        for graph_ind, sub_graph_ind in reactant_atom_map.items():
            assert np.allclose(
                esp_tensor_reactant[sub_graph_ind].tolist(),
                reactant_atom_qtaim_esp[i][graph_ind],
            ), "atom mapping is wrong: {}, {}".format(
                esp_tensor_reactant, reactant_atom_qtaim_esp[i]
            )

        for j in range(len(product_atom_map)):
            assert np.allclose(
                esp_tensor_prod[product_atom_map[j]].tolist(),
                product_atom_qtaim_esp[i][j],
            ), "atom mapping is wrong: {}".format(product_atom_map)


def test_bond_feat_mapping():
    extra_feats = [
        "esp_total",  # the atom esp is also added
        "bond_esp_total",  # this turns on the esp added to graphs # this is a feature
        "indices_qtaim",  # this maps features to bonds
    ]

    df = pd.read_json("./testdata/barrier_2.json")
    reactant_bond_qtaim_esp = df["extra_feat_bond_reactant_esp_total"].tolist()
    product_bond_qtaim_esp = df["extra_feat_bond_product_esp_total"].tolist()
    graphs, feats, reactions = get_data(
        extra_keys=extra_feats, test_df_file="./testdata/barrier_2.json"
    )

    for i in range(len(reactions)):
        esp_tensor_reactant = (graphs[2 * i].ndata["feat"]["bond"][:, -1]).tolist()
        esp_tensor_reactant = [round(x, 3) for x in esp_tensor_reactant]
        esp_tensor_prod = (graphs[2 * i + 1].ndata["feat"]["bond"][:, -1]).tolist()
        esp_tensor_prod = [round(x, 3) for x in esp_tensor_prod]

        reactant_bond_map = reactions[i]._bond_mapping_by_int_index[0]
        product_bond_map = reactions[i]._bond_mapping_by_int_index[1]
        # find mapping betweren values in product_bond_qtaim_esp and esp_tensor_prod
        dict_react_map, dict_prod_map = {}, {}
        for j in range(len(esp_tensor_prod)):
            val = product_bond_qtaim_esp[i][0][j]
            val_rounded = round(val, 3)
            ind_val_in_prod = esp_tensor_prod.index(val_rounded)
            dict_prod_map[j] = ind_val_in_prod

        for j in range(len(esp_tensor_reactant)):
            val = reactant_bond_qtaim_esp[i][0][j]
            val_rounded = round(val, 3)
            ind_val_in_react = esp_tensor_reactant.index(val_rounded)
            dict_react_map[j] = ind_val_in_react

        assert len(dict_prod_map) == len(product_bond_map[0])
        assert len(dict_react_map) == len(reactant_bond_map[0])
