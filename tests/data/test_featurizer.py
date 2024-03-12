import numpy as np
import pandas as pd
from bondnet.test_utils import get_data_test

def test_extra_atom_featurizer():
    extra_feats = {
        "atom": ["esp_nuc","esp_e","esp_total"],
    }

    _, feats, _ = get_data_test(
        extra_keys=extra_feats,
        test_df_file="./testdata/barrier_2.json"
    )
    #print("feats", feats)
    #[print(k, len(v[0])) for k, v in feats.items()]
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 23
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 3


def test_extra_bond_featurizer():

    extra_feats = {
        "atom": ["esp_total"],
        "bond": ["esp_total", "bond_length"],
        "mappings": ["indices_qtaim"],
    }


    _, feats, _ = get_data_test(
        test_df_file="./testdata/barrier_100.json",
        extra_keys=extra_feats
    )
    #print("feats", feats)
    #[print(k, len(v[0])) for k, v in feats.items()]
    #print(feats)
    for k, v in feats[0].items():
        if k == "atom":
            assert len(v[0]) == 21
        elif k == "bond":
            assert len(v[0]) == 9
        elif k == "global":
            assert len(v[0]) == 3


def test_extra_global_featurizer():
    # test rishabh's func group
    extra_feats = {
        "global": ["functional_group_reacted"]
    }
    _, feats, _ = get_data_test(
        test_df_file="./testdata/hydro_funct.json",
        extra_keys=extra_feats,
        allowed_charges=[-1, 0, 1],
        #global_feats=["functional_group_reacted", "charge"],
        target="dG_sp",
    )
    #print(feats)
    # [print(k, len(v[0])) for k, v in feats[0].items()]
    for k, v in feats[0].items():
        #print(k, len(v[0]))
        if k == "atom":
            assert len(v[0]) == 20
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 17

    # test manual global feats
    _, feats, _ = get_data_test(
        test_df_file="./testdata/hydro_funct.json",
        extra_keys={
            "global": []
        },
        allowed_charges=[-1, 0, 1],
        target="dG_sp",
    )

    for k, v in feats[0].items():
        #print(k, len(v[0]))
        if k == "atom":
            assert len(v[0]) == 20
        elif k == "bond":
            assert len(v[0]) == 7
        elif k == "global":
            assert len(v[0]) == 6



def test_atom_feat_mapping():
    extra_feats = {
        "atom": ["esp_total"],
        "bond": ["esp_total"],
        "mappings": ["indices_qtaim"],
    }

    df = pd.read_json("./testdata/green_2.json")
    reactant_atom_qtaim_esp = df["extra_feat_atom_reactant_esp_total"].tolist()
    product_atom_qtaim_esp = df["extra_feat_atom_product_esp_total"].tolist()

    graphs, feats, reactions = get_data_test(
        extra_keys=extra_feats, test_df_file="./testdata/green_2.json",
        target="dHrxn298"
    )
    
    for i in range(len(reactions)):
        esp_tensor_reactant = graphs[2 * i].ndata["ft"]["atom"][:, -1]
        esp_tensor_prod = graphs[2 * i + 1].ndata["ft"]["atom"][:, -1]
        reactant_atom_map = reactions[i]._atom_mapping[0][0]
        product_atom_map = reactions[i]._atom_mapping[1][0]
        
        for graph_ind, sub_graph_ind in reactant_atom_map.items():
            assert np.allclose(
                esp_tensor_reactant[sub_graph_ind].tolist(),
                reactant_atom_qtaim_esp[i][graph_ind],
                atol=1e-2
            ), "atom mapping is wrong, gap: {}".format(
                esp_tensor_reactant[sub_graph_ind] - reactant_atom_qtaim_esp[i][graph_ind]
            )

        for graph_ind, sub_graph_ind in product_atom_map.items():
            assert np.allclose(
                esp_tensor_prod[sub_graph_ind].tolist(),
                product_atom_qtaim_esp[i][graph_ind],
                atol=1e-2
            ), "atom mapping is wrong, gap: {}".format(
                esp_tensor_prod[sub_graph_ind] - product_atom_qtaim_esp[i][graph_ind]
            )


def test_bond_feat_mapping():
    extra_feats = {
        "atom": ["esp_total"],
        "bond": ["esp_total"],
        "mappings": ["indices_qtaim"],
    }

    df = pd.read_json("./testdata/green_2.json")
    reactant_bond_qtaim_esp = df["extra_feat_bond_reactant_esp_total"].tolist()
    product_bond_qtaim_esp = df["extra_feat_bond_product_esp_total"].tolist()

    graphs, feats, reactions = get_data_test(
        extra_keys=extra_feats, test_df_file="./testdata/green_2.json",
        target="dHrxn298"
    )

    for i in range(len(reactions)):
        dict_react_map, dict_prod_map = {}, {}
        esp_tensor_reactant = (graphs[2 * i].ndata["ft"]["bond"][:, -1]).tolist()
        esp_tensor_reactant = [round(x, 3) for x in esp_tensor_reactant]
        esp_tensor_prod = (graphs[2 * i + 1].ndata["ft"]["bond"][:, -1]).tolist()
        esp_tensor_prod = [round(x, 3) for x in esp_tensor_prod]
        prod_qtaim_esp = product_bond_qtaim_esp[i][0]
        react_qtaim_esp = reactant_bond_qtaim_esp[i][0]
        reactant_bond_map = reactions[i]._bond_mapping_by_int_index[0]
        product_bond_map = reactions[i]._bond_mapping_by_int_index[1]

        
        # find mapping betweren values in product_bond_qtaim_esp and esp_tensor_prod
        for j in range(len(esp_tensor_prod)):
            val = prod_qtaim_esp[j]
            val_rounded = round(val, 3)
            ind_val_in_prod = esp_tensor_prod.index(val_rounded)
            dict_prod_map[j] = ind_val_in_prod

        for j in range(len(esp_tensor_reactant)):
            val = react_qtaim_esp[j]
            val_rounded = round(val, 3)
            ind_val_in_react = esp_tensor_reactant.index(val_rounded)
            dict_react_map[j] = ind_val_in_react

        assert len(dict_prod_map) == len(product_bond_map[0])
        assert len(dict_react_map) == len(reactant_bond_map[0])


