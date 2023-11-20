import torch
import pytorch_lightning as pl

from bondnet.model.training_utils import (
    load_model_lightning,
)
from bondnet.data.datamodule import BondNetLightningDataModule
from bondnet.data.utils import mol_graph_to_rxn_graph
from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)
from bondnet.test_utils import get_defaults

# suppress warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.trainer.data_loading",
    lineno=102,
)

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def test_model_construction():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
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
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 4,
            "num_workers": 1,
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
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 256,
            "num_workers": 16,
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
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=config["model"]["precision"],
        log_every_n_steps=1,
        profiler='simple'
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint("./test_save_load/test.ckpt")
    config["restore_path"] = "./test_save_load/test.ckpt"
    config["restore"] = True
    model_restart = load_model_lightning(config["model"], load_dir="./test_save_load/")

    trainer_restart = pl.Trainer(resume_from_checkpoint="./test_save_load/test.ckpt")
    trainer_restart.fit_loop.max_epochs = 5
    trainer_restart.fit(model_restart, dm)
    trainer.test(model, dm)
    assert type(model_restart) == GatedGCNReactionNetworkLightning
    assert type(trainer_restart) == pl.Trainer


def test_transfer_learning():
    dataset_loc = "../data/testdata/barrier_100.json"

    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
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
            "augment": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 4,
            "num_workers": 1,
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
            "batch_size": 4,
            "num_workers": 1,
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
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_checkpoints/",
        precision=config["model"]["precision"],
        log_every_n_steps=1,
    )

    trainer_transfer.fit(model, dm_transfer)

    # trainer.fit(model, dm)
    # trainer.test(model, dm)
    print("training transfer works!")


def test_augmentation():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
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
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 4,
            "num_workers": 8,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)
    feat_size, feat_name = dm.prepare_data()

    config = get_defaults()
    config["model"]["in_feats"] = feat_size
    config["model"]["augment"] = True
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=config["model"]["precision"],
        log_every_n_steps=1,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_reactant_only_construction():
    dataset_loc = "../data/testdata/symmetric.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
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
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 4,
            "num_workers": 1,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value
    config["model"]["reactant_only"] = True
    # print("config", config)
    dm = BondNetLightningDataModule(config)
    feat_size, feat_name = dm.prepare_data()
    dm.setup(stage="predict")
    nodes = ["atom", "bond", "global"]

    for it, (batched_graph, label) in enumerate(dm.test_dataloader()):
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"].view(-1)
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        stdev = label["scaler_stdev"]
        reactions = label["reaction"]

        rxn_graph, rxn_feats = mol_graph_to_rxn_graph(
            batched_graph,
            feats,
            reactions,
            reactant_only=False,
        )
        reactant_graph, reactant_feat = mol_graph_to_rxn_graph(
            batched_graph,
            feats,
            reactions,
            reactant_only=True,
        )

        for node_type in nodes:
            assert rxn_graph.num_nodes(node_type) == reactant_graph.num_nodes(node_type)
            zero_mat = torch.zeros_like(rxn_feats[node_type])
            assert not torch.allclose(
                reactant_feat[node_type], zero_mat
            ), "{}: {}".format(node_type, reactant_feat[node_type], atol=1e-3, rtol=0)

            assert torch.allclose(rxn_feats[node_type], zero_mat), "{}: {}".format(
                node_type, rxn_feats[node_type], atol=1e-3, rtol=0
            )
            # test that reactant_feat is not all zeros


# TODO: test multi-gpu

def test_profiler():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",
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
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 256,
            "num_workers": 16,
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
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=config["model"]["precision"],
        log_every_n_steps=1,
        profiler='advanced'
    )

    trainer.fit(model, dm)


#test_profiler()