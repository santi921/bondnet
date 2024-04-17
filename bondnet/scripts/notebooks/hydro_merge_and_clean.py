import os 
import torch
import pandas as pd 
from bondnet.data.dataset import ReactionDatasetGraphs
from bondnet.model.training_utils import get_grapher
from bondnet.model.training_utils import load_model_lightning
from bondnet.data.dataloader import DataLoaderReaction

def id_converter(df):
    ids_raw, ids_converted = [], []
    for i, row in df.iterrows():
        try:
            id = [i for i in row["reaction_id"].split("-")]
            id = str(id[0] + id[1] + id[2])
        except:
            id = str(row["reactant_id"])
            if type(row["product_id"]) == list:
                for i in row["product_id"]:
                    id += str(i)
            else:
                id += str(row["product_id"])
            id = str(id)
        ids_converted.append(id)
        ids_raw.append(row["reaction_id"])

    return ids_converted, ids_raw


def count_na_in_each_column(df): 
    return df.isna().sum()


def pmg_to_list_of_bonds(pmg_obj):
    adj = pmg_obj["graphs"]["adjacency"]
    list_bonds = []

    for i, list_connections in enumerate(adj):
        if len(list_connections) != 0:
            for j, bond in enumerate(list_connections):
                #print(i, bond["id"])
                list_bonds.append([i, bond["id"]])
    
    return list_bonds 


def check_bonds_on_pmgs(df, key_check_reactants_pmg, key_check_products_pmg, key_check_reactants, key_check_products):
    counts_equal_products = 0
    counts_equal_reactants = 0

    for i in range(len(df)):
        test_pmg_products = df[key_check_products_pmg].tolist()[i]
        test_pmg_reactants = df[key_check_reactants_pmg].tolist()[i]
        pmg_bonds_reactants = pmg_to_list_of_bonds(test_pmg_reactants)
        pmg_bonds_products = pmg_to_list_of_bonds(test_pmg_products)
        bonds_reactants = df[key_check_reactants].tolist()[i]
        bonds_products = df[key_check_products].tolist()[i]
        if bonds_reactants == None: 
            bonds_reactants = []
        if bonds_products == None:
            bonds_products = []
        # sort list of lists 
        bonds_reactants = sorted([sorted(i) for i in bonds_reactants])
        bonds_products = sorted([sorted(i) for i in bonds_products])
        pmg_bonds_reactants = sorted([sorted(i) for i in pmg_bonds_reactants])
        pmg_bonds_products = sorted([sorted(i) for i in pmg_bonds_products])

        if pmg_bonds_products == bonds_products:
            counts_equal_products += 1
        else: 
            print(pmg_bonds_products)
            print(bonds_products)
            print("---"*20)
        if pmg_bonds_reactants == bonds_reactants:
            counts_equal_reactants += 1
        else: 
            print(pmg_bonds_reactants)
            print(bonds_reactants)
            print("---"*20)
    print("equal reactants: {} / {}".format(counts_equal_reactants, len(df)))
    print("equal products: {} / {}".format(counts_equal_products, len(df)))
    

def pmg_to_comp_dict(pmg_obj):
    comp_dict = {}
    for i, site in enumerate(pmg_obj["molecule"]["sites"]):

        if site["species"][0]["element"] not in comp_dict:
            comp_dict[site["species"][0]["element"]] = 1
        else:
            comp_dict[site["species"][0]["element"]] += 1
    # sort dictionary
    comp_dict = dict(sorted(comp_dict.items()))
    return comp_dict


def bonds_on_pmgs(df, key_check_reactants_pmg, key_check_products_pmg):
    reactant_bonds_pmg_list = []
    product_bonds_pmg_list = []
    bonds_broken_list = []
    bonds_formed_list = []
    comp_dict_list = []

    for i in range(len(df)):
        test_pmg_products = df[key_check_products_pmg].tolist()[i]
        test_pmg_reactants = df[key_check_reactants_pmg].tolist()[i]
        pmg_bonds_reactants = pmg_to_list_of_bonds(test_pmg_reactants)
        pmg_bonds_products = pmg_to_list_of_bonds(test_pmg_products)
        # sort list of lists 
        pmg_bonds_reactants = sorted([sorted(i) for i in pmg_bonds_reactants])
        pmg_bonds_products = sorted([sorted(i) for i in pmg_bonds_products])
        bonds_broken = []
        bonds_formed = []

        # find bonds in reactants that are not in products
        for bond in pmg_bonds_reactants:
            if bond not in pmg_bonds_products:
                bonds_broken.append(bond)

        # find bonds in products that are not in reactants
        for bond in pmg_bonds_products:
            if bond not in pmg_bonds_reactants:
                bonds_formed.append(bond)
        
        comp_dict = pmg_to_comp_dict(test_pmg_products)
        comp_dict_list.append(comp_dict)
        bonds_broken_list.append(bonds_broken)
        bonds_formed_list.append(bonds_formed)

        reactant_bonds_pmg_list.append(pmg_bonds_reactants)
        product_bonds_pmg_list.append(pmg_bonds_products)

    return reactant_bonds_pmg_list, product_bonds_pmg_list, bonds_broken_list, bonds_formed_list, comp_dict_list


def get_clean_df(df):

    df_clean = pd.DataFrame()
    df_clean["reactant_id"] = df["reactant_id"]
    df_clean["reactant_nelements"] = df["reactant_nelements"]
    df_clean["product_id"] = df["product_id"]
    df_clean["reaction_id"] = df["reaction_id"]
    df_clean["functional_group_reacted"] = df["functional_group_reacted"]
    df_clean["combined_reactants_graph"] = df["combined_reactants_graph"]
    df_clean["combined_products_graph"] = df["combined_products_graph"]
    df_clean["composition"] = df["composition"]
    df_clean["charge"] = df["charge"]
    df_clean["dataset"] = df["dataset"]
    df_clean["product_bonds"] = df["product_bonds"]
    df_clean["reactant_bonds"] = df["reactant_bonds"]
    df_clean["product_bonds_nometal"] = df["product_bonds_nometal"]
    df_clean["reactant_bonds_nometal"] = df["reactant_bonds_nometal"]
    df_clean["bonds_broken"] = df["bonds_broken"]
    df_clean["bonds_formed"] = df["bonds_formed"]
    df_clean["dG_sp"] = df["dG_sp"]

    return df_clean


def shared_step_min(model, batch):
    # ========== compute predictions ==========
    batched_graph, label = batch
    nodes = ["atom", "bond", "global"]
    feats = {nt: batched_graph.nodes[nt].data["ft"] for nt in nodes}
    target = label["value"].view(-1)
    target_aug = label["value_rev"].view(-1)
    empty_aug = torch.isnan(target_aug).tolist()
    empty_aug = True in empty_aug
    norm_atom = label["norm_atom"]
    norm_bond = label["norm_bond"]
    stdev = label["scaler_stdev"]
    mean = label["scaler_mean"]
    reactions = label["reaction"]

    if model.stdev is None:
        model.stdev = stdev[0]

    pred = model(
        graph=batched_graph,
        feats=feats,
        reactions=reactions,
        reverse=False,
        norm_bond=norm_bond,
        norm_atom=norm_atom,
    )

def main():
    
    training_list = [
        'Filtered_train_set_alchemy_qm9_after_r1_training.json',
        'Training_set_protonated_HEPOM_after_filtering.json',
        'Training_set_hydroxylated_HEPOM_after_filtering.json'
    ]

    test_list = [
        'Filtered_test_set_alchemy_qm9_after_r1_training.json',
        'Holdout_test_set_HEPOM_hydroxylated_cons_with_neutral_cleaned.json',
        'Holdout_test_set_HEPOM_protonated_683_cons_with_neutral_cleaned.json',
    ]

    dfs_training = {}
    dfs_test = {}
    root = "/data/santiago/bondnet_training/data/hepom_cleaned/"
    root_testing = "/data/santiago/bondnet_training/merge/"


    for file in training_list:
        if "protonated" in file: 
            name = "protonated"
        elif "hydroxylated" in file:
            name = "hydroxylated"
        else:
            name = "neutral"
        df = pd.read_json(root + file)
        # add column with the name of the dataset
        df["dataset"] = name

        dfs_training[name] = df 

    for file in test_list:
        if "protonated" in file: 
            name = "protonated"
        elif "hydroxylated" in file:
            name = "hydroxylated"
        else:
            name = "neutral"
        df = pd.read_json(root + file)
        # add column with the name of the dataset
        df["dataset"] = name
        #cols_drop = [col for col in df.columns if "extra" in col]
        # add nonmetal column

        dfs_test[name] = df 


    df_training = pd.concat(dfs_training.values(), ignore_index=True, keys = [
        'reactant_id', 'reactant_charge', 'reactant_spin_multiplicity',
        'reactant_natoms', 'reactant_elements', 'reactant_nelements',
        'reactant_composition', 'reactant_formula_alphabetical',
        'reactant_chemsys', 'reactant_symmetry', 'deprecated',
        'reactant_smiles', 'reactant_structure', 'reactant_molecule_graph',
        'water_structure', 'water_molecule_graph', 'reactant_opt_energy',
        'reactant_sp_energy', 'reactant_enthalpy', 'reactant_entropy',
        'product_id', 'product_smiles', 'reaction_id',
        'functional_group_reacted', 'product_structure',
        'product_molecule_graph', 'product_opt_energy', 'product_sp_energy',
        'product_enthalpy', 'product_entropy', 'dE_opt', 'dE_sp', 'dH', 'dS',
        'dG_opt', 'dG_sp', 'reaction_atom_map', 'mapping_dict',
        'combined_reactants_graph', 'combined_products_graph', 'composition',
        'charge', 'dataset', 'product_bonds', 'reactant_bonds',
        'product_bonds_nometal', 'reactant_bonds_nometal', 'products_graph',
        'reactants_graph', 'bonds_broken', 'bonds_formed'], join='inner')
    

    df_testing = pd.concat(dfs_test.values(), ignore_index=True, keys = [
       'reactant_id', 'reactant_charge', 'reactant_spin_multiplicity',
       'reactant_natoms', 'reactant_elements', 'reactant_nelements',
       'reactant_composition', 'reactant_formula_alphabetical',
       'reactant_chemsys', 'reactant_symmetry', 'deprecated',
       'reactant_smiles', 'reactant_structure', 'reactant_molecule_graph',
       'water_structure', 'water_molecule_graph', 'reactant_opt_energy',
       'reactant_sp_energy', 'reactant_enthalpy', 'reactant_entropy',
       'product_id', 'product_smiles', 'reaction_id',
       'functional_group_reacted', 'product_structure',
       'product_molecule_graph', 'product_opt_energy', 'product_sp_energy',
       'product_enthalpy', 'product_entropy', 'dE_opt', 'dE_sp', 'dH', 'dS',
       'dG_opt', 'dG_sp', 'reaction_atom_map', 'mapping_dict',
       'combined_reactants_graph', 'combined_products_graph', 'composition',
       'charge', 'dataset', 'product_bonds', 'reactant_bonds',
       'product_bonds_nometal', 'reactant_bonds_nometal', 'products_graph',
       'reactants_graph', 'bonds_broken', 'bonds_formed'], join='inner')


    reactant_bonds_pmg_list, product_bonds_pmg_list, bonds_broken, bonds_formed, composition_list = bonds_on_pmgs(df = df_training, 
        key_check_reactants_pmg="combined_reactants_graph", 
        key_check_products_pmg="combined_products_graph", )

    reactant_bonds_pmg_list_test, product_bonds_pmg_list_test, bonds_broken_test, bonds_formed_test, composition_list_test = bonds_on_pmgs(df = df_testing, 
        key_check_reactants_pmg="combined_reactants_graph", 
        key_check_products_pmg="combined_products_graph", )

    charge_list_training = []
    charge_list_testing = []

    for i in range(len(df_testing)):
        charge = df_testing["reactant_charge"].tolist()[i]
        charge_list_testing.append(charge)
    
    for i in range(len(df_training)):
        charge = df_training["reactant_charge"].tolist()[i]
        charge_list_training.append(charge)

    df_testing = df_testing.drop(columns=[col for col in df_testing.columns if "bond" in col])
    df_training = df_training.drop(columns=[col for col in df_training.columns if "bond" in col])

    # overwrite columns
    df_testing["product_bonds"] = product_bonds_pmg_list_test
    df_testing["reactant_bonds"] = reactant_bonds_pmg_list_test
    df_testing["product_bonds_nometal"] = product_bonds_pmg_list_test
    df_testing["reactant_bonds_nometal"] = reactant_bonds_pmg_list_test
    df_testing["bonds_broken"] = bonds_broken_test
    df_testing["bonds_formed"] = bonds_formed_test
    df_testing["composition"] = composition_list_test
    df_testing["reactant_natoms"] = [sum(i.values()) for i in composition_list_test]
    df_testing["products_graph"] = df_testing["combined_products_graph"]
    df_testing["reactants_graph"] = df_testing["combined_reactants_graph"]
    df_testing["product_molecule_graph"] = df_testing["combined_products_graph"]
    df_testing["reactant_molecule_graph"] = df_testing["combined_reactants_graph"]
    df_testing["charge"] = charge_list_testing

    df_training["product_bonds"] = product_bonds_pmg_list
    df_training["reactant_bonds"] = reactant_bonds_pmg_list
    df_training["product_bonds_nometal"] = product_bonds_pmg_list
    df_training["reactant_bonds_nometal"] = reactant_bonds_pmg_list
    df_training["bonds_broken"] = bonds_broken
    df_training["bonds_formed"] = bonds_formed
    df_training["composition"] = composition_list
    df_training["reactant_natoms"] = [sum(i.values()) for i in composition_list]
    df_training["products_graph"] = df_training["combined_products_graph"]
    df_training["reactants_graph"] = df_training["combined_reactants_graph"]
    df_training["product_molecule_graph"] = df_training["combined_products_graph"]
    df_training["reactant_molecule_graph"] = df_training["combined_reactants_graph"]
    df_training["charge"] = charge_list_training


    df_testing_clean = get_clean_df(df_testing)
    df_testing_clean.to_json(root_testing + "merged_testing_raw.json")

    df_training_clean = get_clean_df(df_training)
    df_training_clean.to_json(root_testing + "merged_training_raw.json")


    file_loc_testing = root_testing + "merged_testing_raw.json"
    file_loc_training = root_testing + "merged_training_raw.json"

    dataset_testing = ReactionDatasetGraphs(
        get_grapher(
        {
        "bond": ["bond_length"], 
        "global":["functional_group_reacted"]
        }
        ),
        file=file_loc_testing,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        target="dG_sp",
        filter_species=[30, 30],
        filter_outliers=False,
        filter_sparse_rxns=False,
        feature_filter=False,
        classifier=False,
        debug=False,
        classif_categories=None,
        extra_keys={
        "bond": ["bond_length"], 
        "global":["functional_group_reacted"]
        },
        dataset_atom_types=None,
        extra_info=None,
        species=["C", "F", "H", "N", "O", "Mg", "Li", "S", "Cl", "P", "O", "Br"],
    )


    dataset_training = ReactionDatasetGraphs(
        get_grapher(
        {
        "bond": ["bond_length"], 
        "global":["functional_group_reacted"]
        }
        ),
        file=file_loc_training,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        target="dG_sp",
        filter_species=[30, 30],
        filter_outliers=False,
        filter_sparse_rxns=False,
        feature_filter=False,
        classifier=False,
        debug=False,
        classif_categories=None,
        extra_keys={
        "bond": ["bond_length"], 
        "global":["functional_group_reacted"]
        },
        dataset_atom_types=None,
        extra_info=None,
        species=["C", "F", "H", "N", "O", "Mg", "Li", "S", "Cl", "P", "O", "Br"],
    )


    config = {
    "model": {
        "conv": "GatedGCNConv",
        "augment": False,
        "classifier": False,
        "initializer": "xavier", 
        "readout": "Attention",
        "classif_categories": 3,
        "cat_weights": [
        1.0,
        1.0,
        1.0
        ],
        "embedding_size": 12,
        "epochs": 1000,
        "extra_features": {
        "bond": ["bond_length"], 
        "global":[]
        },
        "extra_info": [],
        "filter_species": [
        100,
        100
        ],
        "fc_activation": "ReLU",
        "fc_batch_norm": False,
        "fc_dropout": 0.1,
        "fc_hidden_size_1": 512,
        "fc_hidden_size_shape": "cone",
        "fc_num_layers": 2,
        "gated_activation": "ReLU",
        "gated_batch_norm": True,
        "gated_dropout": 0.2,
        "gated_graph_norm": False,
        "gated_hidden_size_1": 256,
        "gated_hidden_size_shape": "flat",
        "gated_num_fc_layers": 1,
        "gated_num_layers": 2,
        "gated_residual": False,
        "learning_rate": 0.03810041122614165,
        "precision": "bf16",
        "loss": "mae",
        "num_lstm_iters": 15,
        "num_lstm_layers": 3,
        "restore": False,
        "weight_decay": 0.00005,
        "max_epochs": 1000,
        "max_epochs_transfer": 1000,
        "transfer": False,
        "filter_outliers": False,
        "freeze": False,
        "reactant_only": False
    },
    "optim": {
        "batch_size": 1024,
        "num_devices": 1,
        "num_nodes": 1,
        "num_workers": 1,
        "val_size": 0.1,
        "test_size": 0.1,
        "strategy": "auto",
        "gradient_clip_val": 5.0,
        "accumulate_grad_batches": 3,
        "pin_memory": True,
        "persistent_workers": True
    },
    "dataset": {
        "log_save_dir": "./retrain_best_1/",
        "lmdb_dir": "./lmdb_no_vibes/",
        "target_var": "dG_sp",
        "overwrite": True
    },
    "dataset_transfer": {
        "log_save_dir": "./retrain_best_transfer/",
        "lmdb_dir": "./lmdb_no_vibes_transfer/",
        "target_var": "diff",
        "overwrite": True
    }
    }


    data_loader_training = DataLoaderReaction(
        dataset=dataset_training,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    data_loader_testing = DataLoaderReaction(
        dataset=dataset_testing,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )


    config["model"]["in_feats"] = dataset_training.feature_size
    model = load_model_lightning(config["model"], load_dir="./deez/")


    ids_fucked_training = []
    ids_fucked_testing = []

    for batch in data_loader_training:
        try: 
            shared_step_min(model, batch)
            
        except: 
            id_fucked = batch[1]["id"][0][0]
            ids_fucked_training.append(id_fucked)


    for batch in data_loader_testing:
        try: 
            shared_step_min(model, batch)
            
        except: 
            id_fucked = batch[1]["id"][0][0]
            ids_fucked_testing.append(id_fucked)    


    with open("ids_faulty_test.txt", 'w') as f:
        for s in ids_fucked_testing:
            f.write(str(s) + '\n')

    with open("ids_faulty_train.txt", 'w') as f:
        for s in ids_fucked_training:
            f.write(str(s) + '\n')

    # read from the files 
    ids_fucked_training = open("ids_faulty_train.txt").readlines()
    ids_fucked_testing = open("ids_faulty_test.txt").readlines()
    ids_fucked_training = [i.strip() for i in ids_fucked_training]
    ids_fucked_testing = [i.strip() for i in ids_fucked_testing]
    

    rxn_id_raw_test = df_testing_clean.reaction_id.tolist()
    rxn_id_raw_train = df_training_clean.reaction_id.tolist()

    ids_converted_testing, ids_raw_testing = id_converter(df_testing_clean)
    ids_converted_training, ids_raw_training = id_converter(df_training_clean)

    
    #ids_raw_filtered_test = [ids_raw_testing[i] for i,j in enumerate(ids_converted_testing) if j in ids_fucked_testing]
    #ids_raw_filtered_train = [ids_raw_training[i] for i,j in enumerate(ids_converted_training) if j in ids_fucked_training]
    
    ids_raw_filtered_bool_test = [j not in ids_fucked_testing for i,j in enumerate(ids_converted_testing)]
    ids_raw_filtered_bool_train = [j not in ids_fucked_training for i,j in enumerate(ids_converted_training)]

    df_cleanest_train = df_testing_clean[ids_raw_filtered_bool_train]
    df_cleanest_test = df_testing_clean[ids_raw_filtered_bool_test]
    
    print(df_cleanest_train.shape)
    print(df_cleanest_test.shape)
    
    df_cleanest_test.to_json(root + "merged_clean_test.json")
    df_cleanest_train.to_json(root + "merged_clean_train.json")

main()