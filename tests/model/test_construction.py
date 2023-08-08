import torch, json
import pytorch_lightning as pl
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.model.training_utils import (
    get_grapher,
    load_model_lightning,
)
from bondnet.data.datamodule import BondNetLightningDataModule

from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)

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


def get_defaults():
    
    config = {
        "model" : {
            "augment": False,
            "classifier": False,
            "classif_categories": 3,
            "cat_weights": [1.0, 1.0, 1.0],
            "embedding_size": 24,
            "epochs": 100,
            "extra_features": ["bond_length"],
            "extra_info": [],
            "filter_species": [3, 5],
            "fc_activation": "ReLU",
            "fc_batch_norm": True,
            "fc_dropout": 0.2,
            "fc_hidden_size_1": 256,
            "fc_hidden_size_shape": "flat",
            "fc_num_layers": 1,
            "gated_activation": "ReLU",
            "gated_batch_norm": False,
            "gated_dropout": 0.1,
            "gated_graph_norm": False,
            "gated_hidden_size_1": 512,
            "gated_hidden_size_shape": "flat",
            "gated_num_fc_layers": 1,
            "gated_num_layers": 2,
            "gated_residual": True,
            "learning_rate": 0.003,
            "precision": 32,
            "loss": "mse",
            "num_lstm_iters": 3,
            "num_lstm_layers": 1,
            "restore": False,
            "weight_decay": 0.0,
            "max_epochs": 1000,
            "max_epochs_transfer": 10,
            "transfer": False,
            "filter_outliers": True,
            "freeze": True
        }
    }
    #config = "./settings.json"
    #config = json.load(open(config, "r"))
    return config


def test_model_construction():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",

        },
        "model": 
            {
                "extra_features": [],
                "extra_info": [],
                "debug": False,
                "classifier": False,
                "classif_categories": 3,
                "filter_species": [3, 6],
                "filter_outliers": False,
                "filter_sparse_rxns": False,
                "restore": False

            },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 4,
            "num_workers": 1,
        }
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModule(config)
    feat_size, feat_name = dm.prepare_data()
    #config = get_defaults()
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
        "model": 
            {
                "extra_features": [],
                "extra_info": [],
                "debug": False,
                "classifier": False,
                "classif_categories": 3,
                "filter_species": [3, 6],
                "filter_outliers": False,
                "filter_sparse_rxns": False,
                "restore": False

            },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 4,
            "num_workers": 1,
        }
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
        "model": 
            {
                "extra_features": [],
                "extra_info": [],
                "debug": False,
                "classifier": False,
                "classif_categories": 3,
                "filter_species": [3, 6],
                "filter_outliers": False,
                "filter_sparse_rxns": False,
                "augment": False,
                "restore": False

            },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 4,
            "num_workers": 1,
        }
    }

    config_transfer = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG",

        },
        "model": 
            {
                "extra_features": [],
                "extra_info": [],
                "debug": False,
                "classifier": False,
                "classif_categories": 3,
                "filter_species": [3, 6],
                "filter_outliers": False,
                "filter_sparse_rxns": False,
                "restore": False

            },
        "optim": {
            "val_size": 0.2,
            "test_size": 0.2,
            "batch_size": 4,
            "num_workers": 1,
        }
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
    model = load_model_lightning(config_transfer["model"], load_dir="./test_checkpoints/")

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

    """trainer = pl.Trainer(
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
    """
    #trainer.fit(model, dm)
    #trainer.test(model, dm)
    print("training transfer works!")


def test_augmentation():
    dataset_loc = "../data/testdata/barrier_100.json"
    config = {
        "dataset": {
            "data_dir": dataset_loc,
            "target_var": "dG_barrier",

        },
        "model": 
            {
                "extra_features": [],
                "extra_info": [],
                "debug": False,
                "classifier": False,
                "classif_categories": 3,
                "filter_species": [3, 6],
                "filter_outliers": False,
                "filter_sparse_rxns": False,
                "restore": False

            },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 4,
            "num_workers": 1,
        }
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

# TODO: lmdb tests
# TODO: test multi-gpu


"""
def test_classifier():
    precision = 16
    dataset_loc = "../data/testdata/barrier_100.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher([]),
        file=dataset_loc,
        target="dG_barrier",
        classifier=True,
        classif_categories=3,
        filter_species=[3, 6],
        filter_outliers=False,
        filter_sparse_rxns=False,
        debug=False,
        device=device,
        extra_keys=[],
        extra_info=[],
    )

    train_loader = DataLoaderPrecomputedReactionGraphs(
        dataset, batch_size=128, shuffle=False
    )

    config = get_defaults()
    config["in_feats"] = dataset.feature_size
    config["classifier"] = True
    model = load_model_lightning(config, device=device, load_dir="./test_checkpoints/")

    # init model
    def var_init(model, std=0.01):
        for name, param in model.named_parameters():
            param.data.normal_(mean=0.0, std=std)

    model(model, var_init)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir="./test_checkpoints/",
        precision=precision,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, train_loader)
"""

