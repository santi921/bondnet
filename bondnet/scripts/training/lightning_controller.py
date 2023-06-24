import wandb, argparse, json, os
import torch

import numpy as np
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from bondnet.data.dataset import ReactionNetworkDatasetPrecomputed
from bondnet.data.dataloader import DataLoaderPrecomputedReactionGraphs
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    get_grapher,
    LogParameters,
    load_model_lightning,
)

seed_torch()
torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs


def train_single(
    config,
    dataset=None,
    dataset_transfer=None,
    device=None,
    project_name="hydro_lightning",
    log_save_dir="./logs_lightning/",
    run_name="settings_run_0",
    restore=True,
):
    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    print(">" * 40 + "config_settings" + "<" * 40)

    if dataset is None:
        dataset = ReactionNetworkDatasetPrecomputed(
            grapher=get_grapher(config["extra_features"]),
            file=config["dataset_loc"],
            target=config["target_var"],
            classifier=config["classifier"],
            classif_categories=config["categories"],
            filter_species=config["filter_species"],
            filter_sparse_rxns=config["filter_sparse_rxns"],
            filter_outliers=config["filter_outliers"],
            debug=config["debug"],
            device=config["gpu"],
            feature_filter=config["featurizer_filter"],
            extra_keys=config["extra_features"],
        )

    if dataset_transfer is None and config["transfer"]:
        dataset_transfer = ReactionNetworkDatasetPrecomputed(
            grapher=get_grapher(config["extra_features"]),
            file=config["dataset_loc"],
            target=config["target_var_transfer"],
            classifier=config["classifier"],
            classif_categories=config["categories"],
            filter_species=config["filter_species"],
            filter_outliers=config["filter_outliers"],
            filter_sparse_rxns=config["filter_sparse_rxns"],
            debug=config["debug"],
            device=device,
            extra_keys=config["extra_features"],
            extra_info=config["extra_info"],
        )

    if device is None:
        if config["on_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )
    val_loader = DataLoaderPrecomputedReactionGraphs(
        valset, batch_size=len(valset), shuffle=False
    )
    test_loader = DataLoaderPrecomputedReactionGraphs(
        testset, batch_size=len(testset), shuffle=False
    )
    train_loader = DataLoaderPrecomputedReactionGraphs(
        trainset, batch_size=config["batch_size"], shuffle=True
    )

    if restore:
        config["restore"] = True
        # get all files
        config["restore_dir"] = log_save_dir
        print(
            "\n extracting latest checkpoint from {}\n".format(
                log_save_dir + run_name + ".ckpt"
            )
        )
        files_ckpt = glob(log_save_dir + "/" + run_name + "*ckpt")
        # get latest file
        files_ckpt.sort(key=os.path.getmtime)
        if len(files_ckpt) == 0:
            config["restore"] = False
        else:
            config["restore_file"] = files_ckpt[-1]

    model = load_model_lightning(config, device=device, load_dir=log_save_dir)

    if config["transfer"]:
        with wandb.init(project=project_name + "_transfer") as run_transfer:
            # log config
            wandb.config.update(config)
            trainset_transfer, valset_transfer, _ = train_validation_test_split(
                dataset_transfer, validation=0.15, test=0.0
            )
            val_transfer_loader = DataLoaderPrecomputedReactionGraphs(
                valset_transfer, batch_size=len(valset), shuffle=False
            )
            train_loader_loader = DataLoaderPrecomputedReactionGraphs(
                trainset_transfer, batch_size=config["batch_size"], shuffle=True
            )

            log_parameters = LogParameters()
            logger_tb_transfer = TensorBoardLogger(
                log_save_dir, name="test_logs_transfer"
            )
            logger_wb_transfer = WandbLogger(
                project=project_name, name="test_logs_transfer"
            )
            lr_monitor_transfer = LearningRateMonitor(logging_interval="step")

            checkpoint_callback_transfer = ModelCheckpoint(
                dirpath=log_save_dir,
                filename=run_name
                + "_model_lightning_transfer_{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                auto_insert_metric_name=True,
                save_last=True,
            )

            early_stopping_callback_transfer = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=500,
                verbose=False,
                mode="min",
            )

            trainer_transfer = pl.Trainer(
                max_epochs=config["max_epochs_transfer"],
                accelerator="gpu",
                devices=[0],
                accumulate_grad_batches=5,
                enable_progress_bar=True,
                gradient_clip_val=1.0,
                callbacks=[
                    early_stopping_callback_transfer,
                    lr_monitor_transfer,
                    log_parameters,
                    checkpoint_callback_transfer,
                ],
                enable_checkpointing=True,
                default_root_dir=log_save_dir,
                logger=[logger_tb_transfer, logger_wb_transfer],
                precision=config["precision"],
            )

            trainer_transfer.fit(model, train_loader_loader, val_transfer_loader)

            if config["freeze"]:
                model.gated_layers.requires_grad_(False)
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("Freezing Gated Layers....")
            print("Number of Trainable Model Params: {}".format(params))

            config["transfer"] = False  # signals the end of transfer learning
            with open(run_name + ".json", "w") as f:
                json.dump(config, f, indent=4)

        run_transfer.finish()

    with wandb.init(project=project_name) as run:
        wandb.config.update(config)
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
        logger_wb = WandbLogger(project=project_name, name="test_logs")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback = ModelCheckpoint(
            dirpath=log_save_dir,
            filename=run_name + "_model_lightning_{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=500, verbose=False, mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu",
            devices=[0],
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
            default_root_dir=log_save_dir,
            logger=[logger_tb, logger_wb],
            precision=config["precision"],
        )

        trainer.fit(model, train_loader, val_loader)

        trainer.test(model, test_loader)

    run.finish()


def controller_main(project_name, log_save_dir):
    files = glob("settings*.json")
    first_setting = files[0]
    config = json.load(open(first_setting, "r"))

    if config["precision"] == "16" or config["precision"] == "32":
        config["precision"] = int(config["precision"])

    if config["on_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    extra_keys = config["extra_features"]

    dataset = ReactionNetworkDatasetPrecomputed(
        grapher=get_grapher(extra_keys),
        file=config["dataset_loc"],
        target=config["target_var"],
        classifier=config["classifier"],
        classif_categories=config["categories"],
        filter_species=config["filter_species"],
        filter_outliers=config["filter_outliers"],
        filter_sparse_rxns=config["filter_sparse_rxns"],
        debug=config["debug"],
        device=device,
        extra_keys=extra_keys,
        extra_info=config["extra_info"],
    )

    config["in_feats"] = dataset.feature_size

    dataset_transfer = None

    if config["transfer"]:
        dataset_transfer = ReactionNetworkDatasetPrecomputed(
            grapher=get_grapher(extra_keys),
            file=config["dataset_loc"],
            target=config["target_var_transfer"],
            classifier=config["classifier"],
            classif_categories=config["categories"],
            filter_species=config["filter_species"],
            filter_outliers=config["filter_outliers"],
            filter_sparse_rxns=config["filter_sparse_rxns"],
            debug=config["debug"],
            device=device,
            extra_keys=extra_keys,
            extra_info=config["extra_info"],
        )

    for ind, file in enumerate(files):
        # try:
        print("loading file {}".format(file))
        dict_train = json.load(open(file, "r"))
        dict_train["precision"] = config["precision"]
        dict_train["in_feats"] = dataset.feature_size

        train_single(
            dict_train,
            dataset=dataset,
            dataset_transfer=dataset_transfer,
            device=device,
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
    args = parser.parse_args()
    project_name = args.project_name
    log_save_dir = args.log_save_dir
    controller_main(project_name, log_save_dir)
