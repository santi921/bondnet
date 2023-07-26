import wandb, argparse, torch, json
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from bondnet.data.dataset import (
    BondNetLightningDataModule,
)
from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    LogParameters,
    load_model_lightning,
)

from pytorch_lightning.loggers import WandbLogger
from bondnet.utils import merge_dicts

# print("ntasks:", os.environ["SLURM_NTASKS"])
# print("gloabl rank:", os.environ["SLURM_PROCID"])
# print("local_rank:", os.environ["SLURM_LOCALID"])

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")
seed_torch()


def main():
    # device for ML model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config, duplicates_warning, duplicates_error.

    config = {
        "model": {
            "augment": True,
            "batch_size": 4,
            "debug": False,
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
            "on_gpu": True,
            "restore": False,
            "target_var": "ts",
            "target_var_transfer": "diff",
            "weight_decay": 0.0,
            "max_epochs": 100,
            "max_epochs_transfer": 100,
            "transfer": False,
            "filter_outliers": True,
        },
        "dataset": {
            "log_save_dir": "./test_lmdb/",
            "data_dir": "../../../tests/data/testdata/barrier_100.json",
            "lmdb_dir": "./lmdb_data/",
        },
        "optim": {
            "batch_size": 128,
            "num_devices": 1,
            "num_nodes": 1,
            "num_workers": 4,
            "val_size": 0.15,
            "test_size": 0.1,
        },
    }

    # override config if any, and print override
    dict_for_model = {
        "model": {
            "classifier": False,
            "classif_categories": config["model"]["classif_categories"],
            "filter_species": config["model"]["filter_species"],
            "filter_outliers": config["model"]["filter_outliers"],
            "filter_sparse_rxns": False,
            "debug": config["model"]["debug"],
            # "in_feats": {"atom": 20, "bond": 8, "global": 7},
        },
        "optim": {"batch_size": 12},
    }

    print(config)
    # config datamodule
    dm = BondNetLightningDataModule(config)
    feature_size, feature_names = dm.prepare_data()
    dict_for_model["model"]["in_feats"] = feature_size
    config, overrides = merge_dicts(config, dict_for_model)
    for override_i in overrides:
        print("overrides:", override_i)

    # config lightning model
    model = load_model_lightning(
        config["model"], device=device, load_dir=config["dataset"]["log_save_dir"]
    )

    # config, logger, lr_monitor, checkpoint, early_stopping.
    log_parameters = LogParameters()
    logger_tb = TensorBoardLogger(config["dataset"]["log_save_dir"], name="test_logs")
    logger_wb = WandbLogger(project="test_multi_gpu", name="test_logs")
    logger = [logger_tb, logger_wb]

    # logger_wb = WandbLogger(project="test_multi_gpu", name="test_logs")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["dataset"]["log_save_dir"],
        filename="model_lightning_{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=True,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=500, verbose=False, mode="min"
    )

    # config trainer
    trainer = pl.Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator="gpu",
        devices=config["optim"]["num_devices"],
        num_nodes=config["optim"]["num_nodes"],
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        callbacks=[
            early_stopping_callback,
            lr_monitor,
            log_parameters,
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        # strategy="deepspeed",  # this works
        # strategy="horovod",  # this works
        # strategy=DDPStrategy(find_unused_parameters=False),  # this works
        strategy="ddp",
        default_root_dir=config["dataset"]["log_save_dir"],
        logger=logger,
        precision=config["model"]["precision"],
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)
    # run.finish()


main()
