import wandb, argparse, torch, json
import numpy as np
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from bondnet.data.datamodule import BondNetLightningDataModule

from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    LogParameters,
    load_model_lightning,
)

seed_torch()
torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_gpu", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument("-project_name", type=str, default="hydro_lightning")
    parser.add_argument(
        "-dataset_loc", type=str, default="../../dataset/qm_9_merge_3_qtaim.json"
    )
    parser.add_argument("-log_save_dir", type=str, default="./logs_lightning/")
    parser.add_argument("-config", type=str, default="./settings.json")
    parser.add_argument(
        "--lmdb", default=False, action="store_true", help="use lmdb for dataset"
    )

    args = parser.parse_args()

    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    use_lmdb = bool(args.lmdb)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    config = args.config
    config = json.load(open(config, "r"))

    if config["model"]["precision"] == "16" or config["model"]["precision"] == "32":
        config["model"]["precision"] = int(config["model"]["precision"])

    # dataset
    config["dataset"]["data_dir"] = dataset_loc
    extra_keys = config["model"]["extra_features"]
    config["model"]["filter_sparse_rxns"] = False
    config["model"]["debug"] = debug

    config["dataset_transfer"]["data_dir"] = dataset_loc

    # if use_lmdb:
    #    dm = BondNetLightningDataModuleLMDB(config)
    # else:
    dm = BondNetLightningDataModule(config)

    feature_size, feature_names = dm.prepare_data()
    config["model"]["in_feats"] = feature_size
    config["dataset"]["feature_names"] = feature_names

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))

    print(">" * 40 + "config_settings" + "<" * 40)

    model = load_model_lightning(config["model"], load_dir=log_save_dir)
    print("model constructed!")
    if config["model"]["transfer"]:
        with wandb.init(project=project_name + "_transfer") as run_transfer:
            config_transfer = deepcopy(config)
            config_transfer["dataset"] = config_transfer["dataset_transfer"]

            # if use_lmdb:
            #    dm_transfer = BondNetLightningDataModuleLMDB(config_transfer)
            # else:
            dm_transfer = BondNetLightningDataModule(config_transfer)

            log_parameters = LogParameters()
            logger_tb_transfer = TensorBoardLogger(
                config_transfer["dataset"]["log_save_dir"], name="test_logs_transfer"
            )
            logger_wb_transfer = WandbLogger(
                project=project_name, name="test_logs_transfer"
            )
            lr_monitor_transfer = LearningRateMonitor(logging_interval="step")

            checkpoint_callback_transfer = ModelCheckpoint(
                dirpath=config_transfer["dataset"]["log_save_dir"],
                filename="model_lightning_transfer_{epoch:02d}-{val_l1:.2f}",
                monitor="val_l1",
                mode="min",
                auto_insert_metric_name=True,
                save_last=True,
            )

            early_stopping_callback_transfer = EarlyStopping(
                monitor="val_l1",
                min_delta=0.00,
                patience=500,
                verbose=False,
                mode="min",
            )

            trainer_transfer = pl.Trainer(
                max_epochs=config_transfer["model"]["max_epochs_transfer"],
                accelerator="gpu",
                devices=config_transfer["optim"]["num_devices"],
                num_nodes=config_transfer["optim"]["num_nodes"],
                accumulate_grad_batches=config_transfer["optim"][
                    "accumulate_grad_batches"
                ],
                strategy=config["optim"]["strategy"],
                enable_progress_bar=True,
                gradient_clip_val=config_transfer["optim"]["gradient_clip_val"],
                callbacks=[
                    early_stopping_callback_transfer,
                    lr_monitor_transfer,
                    log_parameters,
                    checkpoint_callback_transfer,
                ],
                enable_checkpointing=True,
                default_root_dir=log_save_dir,
                logger=[logger_tb_transfer, logger_wb_transfer],
                precision=config_transfer["model"]["precision"],
            )

            trainer_transfer.fit(model, dm_transfer)
            model_parameters_prior = filter(
                lambda p: p.requires_grad, model.parameters()
            )

            if config_transfer["model"]["freeze"]:
                params_prior = sum([np.prod(p.size()) for p in model_parameters_prior])
                print(">" * 25 + "Freezing Module" + "<" * 25)
                print("Freezing Gated Layers....")
                print("Number of Trainable Model Params Prior: {}".format(params_prior))
                model.gated_layers.requires_grad_(False)
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("Number of Trainable Model Params: {}".format(params))

        run_transfer.finish()

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(
            config["dataset"]["log_save_dir"], name="test_logs"
        )
        logger_wb = WandbLogger(project=project_name, name="test_logs")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=config["dataset"]["log_save_dir"],
            filename="model_lightning_{epoch:02d}-{val_l1:.2f}",
            monitor="val_l1",
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_l1", min_delta=0.00, patience=500, verbose=False, mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=config["model"]["max_epochs"],
            accelerator="gpu",
            devices=config["optim"]["num_devices"],
            num_nodes=config["optim"]["num_nodes"],
            gradient_clip_val=config["optim"]["gradient_clip_val"],
            accumulate_grad_batches=config["optim"]["accumulate_grad_batches"],
            enable_progress_bar=True,
            callbacks=[
                early_stopping_callback,
                lr_monitor,
                log_parameters,
                checkpoint_callback,
            ],
            enable_checkpointing=True,
            strategy=config["optim"]["strategy"],
            default_root_dir=config["dataset"]["log_save_dir"],
            logger=[logger_tb, logger_wb],
            precision=config["model"]["precision"],
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)
    run.finish()
