import torch
import pytorch_lightning as pl

from bondnet.model.training_utils import (
    load_model_lightning,
)
from bondnet.data.datamodule import BondNetLightningDataModule, BondNetLightningDataModuleLMDB
from bondnet.data.utils import process_batch_mol_rxn

from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)
from bondnet.test_utils import get_defaults

# suppress warnings
import warnings
import time 


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.trainer.data_loading",
    lineno=102,
)
torch.set_float32_matmul_precision("medium")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True


def test_model_construction():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)
    feat_size, feat_name = dm.prepare_data()
    # config = get_defaults()
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_checkpoints/")
    assert type(model) == GatedGCNReactionNetworkLightning


def test_model_save_load():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [10, 10],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)

    feat_size, feat_name = dm.prepare_data()
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=32,
        strategy="ddp",
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        default_root_dir="./test_save_load/",
    )

    trainer.fit(model, dm)
    print("done training")
    trainer.save_checkpoint("./test_save_load/test.ckpt")
    print("done saving")
    config["restore_path"] = "./test_save_load/test.ckpt"
    config["restore"] = True
    model_restart = load_model_lightning(config["model"], load_dir="./test_save_load/")
    print("done loading")

    trainer_restart = pl.Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices=1,
        precision=32,
        strategy="ddp",
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        default_root_dir="./test_save_load/",
    )

    trainer_restart.fit(model_restart, dm, ckpt_path="./test_save_load/test.ckpt")
    print("done training pt2 ")
    trainer_restart.test(model, dm)
    print("done testing")
    assert type(model_restart) == GatedGCNReactionNetworkLightning
    


def test_model_set2set():
    dataset_loc = "../data/testdata/barrier_100.json"
    #dataset_loc = "/home/santiagovargas/dev/bondnet/bondnet/dataset/mg_dataset/rapter_clean/train_inorganic_mg_05132023.pkl"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    config_model["model"]["gated_hidden_size_1"] = 32
    config_model["model"]["gated_num_fc_layers"] = 1
    config_model["model"]["fc_hidden_size_1"] = 128
    config_model["model"]["embedding_size"] = 12
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)

    feat_size, feat_name = dm.prepare_data()
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        default_root_dir="./test_save_load/",
    )
    start_time = time.time()
    trainer.fit(model, dm)
    end_time = time.time()
    delta_time = end_time - start_time
    print("time for set2set", delta_time)


def test_model_attention():
    dataset_loc = "../data/testdata/barrier_100.json"
    #dataset_loc = "/home/santiagovargas/dev/bondnet/bondnet/dataset/mg_dataset/rapter_clean/train_inorganic_mg_05132023.pkl"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    config_model["model"]["readout"] = "Attention"
    config_model["model"]["gated_hidden_size_1"] = 32
    config_model["model"]["gated_num_fc_layers"] = 1
    config_model["model"]["fc_hidden_size_1"] = 128
    config_model["model"]["embedding_size"] = 12
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)

    feat_size, feat_name = dm.prepare_data()
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        default_root_dir="./test_save_load/",
    )

    start = time.time()
    trainer.fit(model, dm)
    end = time.time()
    delta_time = end - start
    print("time for attention", delta_time)



def test_transfer_learning():
    dataset_loc = "../data/testdata/barrier_100.json"

    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "augment": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }

    config_transfer = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG",
        },
        "model": {
            "extra_features": [],
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }

    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value
    for key, value in config_model["model"].items():
        config_transfer["model"][key] = value

    dm = BondNetLightningDataModule(config)
    dm_transfer = BondNetLightningDataModule(config_transfer)
    feat_size, feat_name = dm.prepare_data()
    _, _ = dm_transfer.prepare_data()

    config["model"]["in_feats"] = feat_size
    config_transfer["model"]["in_feats"] = feat_size
    model = load_model_lightning(
        config_transfer["model"], load_dir="./test_checkpoints/"
    )

    trainer_transfer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_checkpoints/",
        log_every_n_steps=1,
    )

    trainer_transfer.fit(model, dm_transfer)

    # trainer.fit(model, dm)
    # trainer.test(model, dm)
    

def test_augmentation():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)
    feat_size, feat_name = dm.prepare_data()
    dm.setup(stage="predict")

    config = get_defaults()
    config["model"]["in_feats"] = feat_size
    config["model"]["augment"] = True
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")


    nodes = ["atom", "bond", "global"]

    for it, (batched_graph, label) in enumerate(dm.test_dataloader()):
        feats = {nt: batched_graph.nodes[nt].data["ft"] for nt in nodes}
        target = label["value"].view(-1)
        target_aug = label["value_rev"].view(-1)

        reactions = label["reaction"]
        
        # graphs and reverse_graphs are the same
        rxn_graph, rxn_feats = mol_graph_to_rxn_graph(
            batched_graph,
            feats,
            reactions,
            reactant_only=False,
        )

        rxn_graph_rev, rxn_feats_rev = mol_graph_to_rxn_graph(
            batched_graph,
            feats,
            reactions,
            reactant_only=False,
            reverse=True,
        )

        for node_type in nodes:
            assert torch.allclose(rxn_feats[node_type], -rxn_feats_rev[node_type], atol=1e-3, rtol=0)
        
        assert not torch.allclose(target, target_aug, atol=1e-3, rtol=0)

def test_lmdb():

    config = {
        "dataset": {
            "log_save_dir": "./model_log/",
            "train_lmdb": "../data/testdata/lmdb/",
            "val_lmdb": "../data/testdata/lmdb/",
            "test_lmdb": "../data/testdata/lmdb/",
            "target_var": "value",
            "overwrite": False,
            "no_splits": False
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModuleLMDB(config)
    feat_size, feat_name = dm.prepare_data()

    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        default_root_dir="./test_save_load/",
    )

    start_time = time.time()
    trainer.fit(model, dm)
    end_time = time.time()
    delta_time = end_time - start_time
    print("time for mean", delta_time)


def test_reactant_only_construction():
    dataset_loc = "../data/testdata/barrier_100.json"

    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "augment": False,
            "restore": False,
            "initializer": "kaiming"
        },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }


    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)
    feat_size, feat_name = dm.prepare_data()
    dm.setup(stage="predict")
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(
        config["model"], load_dir="./test_checkpoints/"
    )

    
    for it, (batched_graph, label, batch_data) in enumerate(dm.test_dataloader()):
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
        reactions = len(target)
        device = feats["bond"].device
        

        if model.stdev is None:
            model.stdev = stdev[0]

        # embedding
        feats_out = model.embedding(feats)
        # gated layer
        for layer in model.gated_layers:
            feats_out = layer(batched_graph, feats_out, norm_atom, norm_bond)

        feats_out_reactant_only = process_batch_mol_rxn(
                    feats = feats_out,
                    reactions = reactions,
                    device = device,
                    reverse = False,
                    batch_data=batch_data, 
                    reactant_only=True
                )
        
        feats_out_full = process_batch_mol_rxn(
                    feats = feats_out,
                    reactions = reactions,
                    device = device,
                    reverse = False,
                    batch_data=batch_data
                )
        
        feats_out_full = model.readout_layer(batch_data["batched_rxn_graphs"], feats_out_full)
        feats_out_reactant_only = model.readout_layer(batch_data["batched_rxn_graphs"], feats_out_reactant_only)

        # check embeddings

        
        #assert rxn_graph.num_nodes(node_type) == reactant_graph.num_nodes(node_type)
        zero_mat = torch.zeros_like(feats_out_full)
        assert not torch.allclose(
            feats_out_full, zero_mat
        ), "feats only is zero"

        assert not torch.allclose(feats_out_reactant_only, zero_mat), "feats reactant only is zero"
        
        # assert they arent the same
        assert not torch.allclose(feats_out_full, feats_out_reactant_only, atol=1e-3, rtol=0)
            

def test_profiler():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)

    feat_size, feat_name = dm.prepare_data()
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")
    #profiler = pl.profiler.AdvancedProfiler(dirpath="./profiler_res/", filename="res.txt")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        #log_every_n_steps=1,
        profiler='advanced'
    )

    trainer.fit(model, dm)


# new bondnet times

# time for set2set 8.181310653686523
# time for attention 6.96621036529541
# time for mean 9.632676362991333
