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
    parser.add_argument("-log_save_dir", type=str, default="./logs_ckpt/")
    parser.add_argument("-config", type=str, default="./settings.json")
    parser.add_argument(
        "--lmdb", default=False, action="store_true", help="use lmdb for dataset"
    )
    parser.add_argument("-period", default=False, type=str, help="save every n epochs")

    args = parser.parse_args()

    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    use_lmdb = bool(args.lmdb)
    period = int(args.period)
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

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
        logger_wb = WandbLogger(project=project_name, name="test_logs")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            log_save_dir,
            filename="model_epoch_{epoch}",
            every_n_epochs=period,
            save_last=False,
            save_top_k=-1,
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
            default_root_dir=log_save_dir,
            logger=[logger_tb, logger_wb],
            precision=config["model"]["precision"],
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)
    run.finish()
