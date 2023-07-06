import wandb, argparse, torch, json
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from bondnet.data.dataset import ReactionNetworkDatasetPrecomputed
from bondnet.data.dataloader import (
    DataLoaderPrecomputedReactionGraphs,
    DataLoaderPrecomputedReactionGraphsParallel,
    collate_parallel,
)
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    get_grapher,
    LogParameters,
    load_model_lightning,
)

# from bondnet.data.dataloader import collate
import torch.multiprocessing as mp
import torch.distributed as dist

# import dill as pickle

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
seed_torch()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
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
    }

    dataset_loc = "../../../tests/data/testdata/barrier_100.json"

    on_gpu = config["on_gpu"]
    extra_keys = config["extra_features"]
    debug = config["debug"]
    precision = config["precision"]

    if precision == "16" or precision == "32":
        precision = int(precision)

    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    extra_keys = config["extra_features"]

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher(extra_keys),
        file=dataset_loc,
        target=config["target_var"],
        classifier=config["classifier"],
        classif_categories=config["classif_categories"],
        filter_species=config["filter_species"],
        filter_outliers=config["filter_outliers"],
        filter_sparse_rxns=False,
        debug=debug,
        device="cpu",
        extra_keys=extra_keys,
        extra_info=config["extra_info"],
    )

    log_save_dir = "./logs_lightning/"
    dict_for_model = {
        "extra_features": extra_keys,
        "classifier": False,
        "classif_categories": config["classif_categories"],
        "filter_species": config["filter_species"],
        "filter_outliers": config["filter_outliers"],
        "filter_sparse_rxns": False,
        "debug": debug,
        "in_feats": dataset.feature_size,
    }
    config["batch_size"] = 24
    config.update(dict_for_model)
    #! 2. split dataset.  train_validation_test_split is in bondnet dataset.
    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))

    print(">" * 40 + "config_settings" + "<" * 40)

    #! 3. dataloader
    val_loader = DataLoaderPrecomputedReactionGraphsParallel(
        dataset=valset,
        batch_size=len(valset),
        shuffle=False,
        collate_fn=collate_parallel,
    )
    test_loader = DataLoaderPrecomputedReactionGraphsParallel(
        dataset=testset,
        batch_size=len(testset),
        shuffle=False,
        collate_fn=collate_parallel,
    )
    train_loader = DataLoaderPrecomputedReactionGraphsParallel(
        dataset=trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_parallel,
    )

    # pickle.dump(train_loader, open("train_loader.pkl", "wb"))
    # pickle.dump(val_loader, open("val_loader.pkl", "wb"))
    # pickle.dump(test_loader, open("test_loader.pkl", "wb"))

    model = load_model_lightning(config, device=device, load_dir=log_save_dir)

    project_name = "test_multi_gpu"

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
        logger_wb = WandbLogger(project=project_name, name="test_logs")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=log_save_dir,
            filename="model_lightning_{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",  # TODO
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=500, verbose=False, mode="min"
        )
        from pytorch_lightning.strategies import DDPStrategy

        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu",
            devices=[0, 1],
            accumulate_grad_batches=5,
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
            strategy="ddp",  # this works
            default_root_dir=log_save_dir,
            logger=[logger_tb, logger_wb],
            precision=precision,
        )
        # print world and rank
        print(f"world size: {trainer.world_size}")
        print(f"global rank: {trainer.global_rank}")
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

    run.finish()


main()
