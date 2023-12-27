import wandb, argparse, json, os
import torch
import numpy as np
from glob import glob
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


def train_single(
    config,
    dm=None,
    dm_transfer=None,
    # device=None,
    project_name="hydro_lightning",
    log_save_dir="./logs_lightning/",
    run_name="settings_run_0",
    restore=True,
    use_lmdb=False,
):
    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    print(">" * 40 + "config_settings" + "<" * 40)

    if dm is None:
        # if use_lmdb:
        #    dm = BondNetLightningDataModuleLMDB(config)
        # else:
        dm = BondNetLightningDataModule(config)

    feature_size, feature_names = dm.prepare_data()
    config["model"]["in_feats"] = feature_size
    config["dataset"]["feature_names"] = feature_names

    if dm_transfer is None and config["model"]["transfer"]:
        config_transfer = deepcopy(config)
        config_transfer["dataset"] = config_transfer["dataset_transfer"]

        dm_transfer = BondNetLightningDataModule(config_transfer)
        # if use_lmdb:
        #    dm_transfer = BondNetLightningDataModuleLMDB(config_transfer)
        # else:
        dm_transfer = BondNetLightningDataModule(config_transfer)

    if restore:
        config["model"]["restore"] = True
        # get all files
        config["model"]["restore_dir"] = log_save_dir
        print(
            "\n extracting latest checkpoint from {}\n".format(
                log_save_dir + run_name + ".ckpt"
            )
        )
        files_ckpt = glob(log_save_dir + "/" + run_name + "*ckpt")
        # get latest file
        files_ckpt.sort(key=os.path.getmtime)
        if len(files_ckpt) == 0:
            config["model"]["restore"] = False
        else:
            config["model"]["restore_file"] = files_ckpt[-1]

    model = load_model_lightning(config["model"], load_dir=log_save_dir)

    if config["model"]["transfer"]:
        with wandb.init(project=project_name + "_transfer") as run_transfer:
            # log config
            wandb.config.update(config)

            log_parameters = LogParameters()
            logger_tb_transfer = TensorBoardLogger(
                config_transfer["dataset"]["log_save_dir"],
                name="test_logs_transfer",
            )
            logger_wb_transfer = WandbLogger(
                project=project_name + "_transfer", name="test_logs_transfer"
            )
            lr_monitor_transfer = LearningRateMonitor(logging_interval="step")

            checkpoint_callback_transfer = ModelCheckpoint(
                dirpath=log_save_dir,
                filename=run_name
                + "_model_lightning_transfer_{epoch:02d}-{val_l1:.2f}",
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

            if config["model"]["freeze"]:
                model_parameters_prior = filter(
                    lambda p: p.requires_grad, model.parameters()
                )
                params_prior = sum([np.prod(p.size()) for p in model_parameters_prior])
                print(">" * 25 + "Freezing Module" + "<" * 25)
                print("Freezing Gated Layers....")
                print("Number of Trainable Model Params Prior: {}".format(params_prior))
                model.gated_layers.requires_grad_(False)
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("Number of Trainable Model Params: {}".format(params))

            config["model"]["transfer"] = False  # signals the end of transfer learning
            with open(run_name + ".json", "w") as f:
                json.dump(config, f, indent=4)

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


def controller_main(project_name, log_save_dir):
    files = glob("settings*.json")
    first_setting = files[0]
    config = json.load(open(first_setting, "r"))

    # if config["precision"] == "16" or config["precision"] == "32":
    #    config["precision"] = int(config["precision"])
    if config["model"]["precision"] == "16" or config["model"]["precision"] == "32":
        config["model"]["precision"] = int(config["model"]["precision"])
    dm = BondNetLightningDataModule(config)
    feature_size, feature_names = dm.prepare_data()
    config["model"]["in_feats"] = feature_size
    config["dataset"]["feature_names"] = feature_names

    # config["in_feats"] = dataset.feature_size

    # dataset_transfer = None
    dm_transfer = None
    if config["model"]["transfer"]:
        config_transfer = deepcopy(config)
        config_transfer["dataset"] = config_transfer["dataset_transfer"]
        dm_transfer = BondNetLightningDataModule(config_transfer)

    for ind, file in enumerate(files):
        # try:
        print("loading file {}".format(file))
        dict_train = json.load(open(file, "r"))

        if (
            dict_train["model"]["precision"] == "16"
            or dict_train["model"]["precision"] == "32"
        ):
            dict_train["model"]["precision"] = int(dict_train["model"]["precision"])
        # dict_train["precision"] = config["precision"]
        dict_train["in_feats"] = feature_size

        train_single(
            dict_train,
            dm=dm,
            dm_transfer=dm_transfer,
            # device=device,
            project_name=project_name,
            log_save_dir=log_save_dir,
            run_name=file.split(".")[0],
            restore=True,
        )
        print("finished file {}".format(file))
        os.rename(file, str(file.split(".")[0].split("_")[-1]) + "_done.json")

        # except:
        #    print("failed on file {}".format(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-project_name", type=str, default="hydro_lightning")
    parser.add_argument("-log_save_dir", type=str, default="./logs_lightning/")
    parser.add_argument(
        "--lmdb", default=False, action="store_true", help="use lmdb for dataset"
    )
    args = parser.parse_args()
    project_name = args.project_name
    log_save_dir = args.log_save_dir
    controller_main(project_name, log_save_dir, use_lmdb=bool(args.lmdb))
