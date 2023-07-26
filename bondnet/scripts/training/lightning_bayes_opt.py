import wandb, argparse, torch, json
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from copy import deepcopy

from bondnet.data.dataloader import DataLoaderPrecomputedReactionGraphs
from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    LogParameters,
    load_model_lightning,
)

from bondnet.data.datamodule import (
    BondNetLightningDataModule,
    BondNetLightningDataModuleLMDB,
)


seed_torch()
torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


class TrainingObject:
    def __init__(
        self, sweep_config, log_save_dir, project_name, dataset_loc, lmdb_root, use_lmdb
    ):
        self.sweep_config = sweep_config
        self.log_save_dir = log_save_dir
        self.wandb_name = project_name
        self.dataset_loc = dataset_loc
        self.lmdb_root = lmdb_root
        self.use_lmdb = use_lmdb

        # if self.config["parameters"]["on_gpu"]:
        #    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #    self.device = torch.device("cpu")

        if "extra_features" in self.sweep_config["parameters"]:
            self.extra_keys = self.sweep_config["parameters"]["extra_features"][
                "values"
            ][0]
        else:
            self.extra_keys = None
        print("extra keys: ", self.extra_keys)
        print("debug value: ", self.sweep_config["parameters"]["debug"]["values"])
        print(
            "target value: ", self.sweep_config["parameters"]["target_var"]["values"][0]
        )
        print(
            "tranfer learning?: ",
            bool(self.sweep_config["parameters"]["transfer"]["values"][0]),
        )
        dm_config = {
            "dataset": {
                "data_dir": self.dataset_loc,
                "log_save_dir": self.log_save_dir,
                "target_var": self.sweep_config["parameters"]["target_var"]["values"][
                    0
                ],
                "overwrite": self.sweep_config["parameters"]["overwrite"]["values"][0],
                "lmdb_dir": self.lmdb_root,
            },
            "model": {
                "classifier": self.sweep_config["parameters"]["classifier"]["values"][
                    0
                ],
                "classif_categories": self.sweep_config["parameters"][
                    "classif_categories"
                ]["values"][0],
                "filter_species": self.sweep_config["parameters"]["filter_species"][
                    "values"
                ][0],
                "filter_outliers": self.sweep_config["parameters"]["filter_outliers"][
                    "values"
                ][0],
                "filter_sparse_rxns": self.sweep_config["parameters"][
                    "filter_sparse_rxns"
                ]["values"][0],
                "debug": self.sweep_config["parameters"]["debug"]["values"][0],
                "extra_features": self.extra_keys,
                "extra_info": self.sweep_config["parameters"]["extra_info"]["values"][
                    0
                ],
            },
            "optim": {
                "batch_size": self.sweep_config["parameters"]["batch_size"]["values"][
                    0
                ],
                "num_workers": self.sweep_config["parameters"]["num_workers"]["values"][
                    0
                ],
                "num_devices": self.sweep_config["parameters"]["num_devices"]["values"][
                    0
                ],
                "num_nodes": self.sweep_config["parameters"]["num_nodes"]["values"][0],
                "val_size": self.sweep_config["parameters"]["val_size"]["values"][0],
                "test_size": self.sweep_config["parameters"]["test_size"]["values"][0],
            },
        }
        if self.use_lmdb:
            self.dm = BondNetLightningDataModuleLMDB(dm_config)
        else:
            self.dm = BondNetLightningDataModule(dm_config)

        feature_size, feature_names = self.dm.prepare_data()
        # config["model"]["in_feats"] = feature_size
        # config["dataset"]["feature_names"] = feature_names
        self.in_feats = feature_size
        self.feature_names = feature_names

        if bool(self.sweep_config["parameters"]["transfer"]["values"][0]):
            config_transfer = deepcopy(dm_config)
            config_transfer["dataset"]["target_var"] = self.sweep_config["parameters"][
                "target_var_transfer"
            ]["values"][0]

            if self.use_lmdb:
                self.dm = BondNetLightningDataModuleLMDB(config_transfer)
            else:
                self.dm = BondNetLightningDataModule(config_transfer)

    def make_model(self, config):
        # convert old config to new config TODO
        config["model"]["in_feats"] = self.in_feats
        model = load_model_lightning(config["model"], load_dir=self.log_save_dir)
        return model

    def train(self):
        with wandb.init(project=self.wandb_name) as run:
            init_config = wandb.config
            if "cat_weights" not in init_config:
                init_config["cat_weights"] = [1.0, 1.0, 1.0]
            config = {
                "model": {
                    "in_feats": self.in_feats,
                    "extra_features": self.extra_keys,
                    "precision": init_config["precision"],
                    "max_epochs": init_config["max_epochs"],
                    "max_epochs_transfer": init_config["max_epochs_transfer"],
                    "freeze": init_config["freeze"],
                    "transfer": init_config["transfer"],
                    "augment": init_config["augment"],
                    "classifier": init_config["classifier"],
                    "classif_categories": init_config["classif_categories"],
                    "cat_weights": init_config["cat_weights"],
                    "embedding_size": init_config["embedding_size"],
                    "extra_info": init_config["extra_info"],
                    "filter_species": init_config["filter_species"],
                    "fc_activation": init_config["fc_activation"],
                    "fc_batch_norm": init_config["fc_batch_norm"],
                    "fc_dropout": init_config["fc_dropout"],
                    "fc_hidden_size_1": init_config["fc_hidden_size_1"],
                    "fc_hidden_size_shape": init_config["fc_hidden_size_shape"],
                    "fc_num_layers": init_config["fc_num_layers"],
                    "gated_activation": init_config["gated_activation"],
                    "gated_batch_norm": init_config["gated_batch_norm"],
                    "gated_dropout": init_config["gated_dropout"],
                    "gated_graph_norm": init_config["gated_graph_norm"],
                    "gated_hidden_size_1": init_config["gated_hidden_size_1"],
                    "gated_hidden_size_shape": init_config["gated_hidden_size_shape"],
                    "gated_num_fc_layers": init_config["gated_num_fc_layers"],
                    "gated_num_layers": init_config["gated_num_layers"],
                    "gated_residual": init_config["gated_residual"],
                    "learning_rate": init_config["learning_rate"],
                    "loss": init_config["loss"],
                    "num_lstm_iters": init_config["num_lstm_iters"],
                    "num_lstm_layers": init_config["num_lstm_layers"],
                    "on_gpu": init_config["on_gpu"],
                    "restore": init_config["restore"],
                    "weight_decay": init_config["weight_decay"],
                    "filter_outliers": init_config["filter_outliers"],
                },
                "dataset": {
                    "data_dir": self.dataset_loc,
                    "log_save_dir": self.log_save_dir,
                    "feature_names": self.feature_names,
                    "target_var": init_config["target_var"],
                    "overwrite": init_config["overwrite"],
                    "lmdb_dir": self.lmdb_root,
                },
                "dataset_transfer": {
                    "data_dir": self.dataset_loc,
                    "log_save_dir": self.log_save_dir,
                    "feature_names": self.feature_names,
                    "target_var": init_config["target_var_transfer"],
                    "overwrite": init_config["overwrite"],
                    "lmdb_dir": self.lmdb_root,
                },
                "optim": {
                    "batch_size": init_config["batch_size"],
                    "num_workers": init_config["num_workers"],
                    "num_devices": init_config["num_devices"],
                    "num_nodes": init_config["num_nodes"],
                    "accumulate_grad_batches": init_config["accumulate_grad_batches"],
                    "gradient_clip_val": init_config["gradient_clip_val"],
                    "strategy": init_config["strategy"],
                    "val_size": init_config["val_size"],
                    "test_size": init_config["test_size"],
                },
            }

            # make helper to convert from old config to new config

            model = self.make_model(config)
            # log dataset
            wandb.log({"dataset": self.dataset_loc})
            if config["model"]["transfer"]:
                # print("transfer learning -- " * 10)
                config_transfer = deepcopy(config)
                config_transfer["dataset"] = config_transfer["dataset_transfer"]
                log_parameters_transfer = LogParameters()

                logger_tb_transfer = TensorBoardLogger(
                    self.log_save_dir,
                    name="test_logs_transfer",
                )
                logger_wb_transfer = WandbLogger(
                    project=self.wandb_name + "_transfer",
                    name=self.log_save_dir + "_transfer",
                )
                lr_monitor_transfer = LearningRateMonitor(logging_interval="step")

                checkpoint_callback_transfer = ModelCheckpoint(
                    dirpath=self.log_save_dir,
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
                        checkpoint_callback_transfer,
                    ],
                    enable_checkpointing=True,
                    default_root_dir=self.log_save_dir,
                    logger=[logger_tb_transfer, logger_wb_transfer],
                    precision=config_transfer["model"]["precision"],
                )

                trainer_transfer.fit(model, self.dm_transfer)

                model_parameters_prior = filter(
                    lambda p: p.requires_grad, model.parameters()
                )

                if config_transfer["model"]["freeze"]:
                    params_prior = sum(
                        [np.prod(p.size()) for p in model_parameters_prior]
                    )
                    print(">" * 25 + "Freezing Module" + "<" * 25)
                    print("Freezing Gated Layers....")
                    print(
                        "Number of Trainable Model Params Prior: {}".format(
                            params_prior
                        )
                    )
                    model.gated_layers.requires_grad_(False)
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                print("Number of Trainable Model Params: {}".format(params))

            checkpoint_callback = ModelCheckpoint(
                dirpath=self.log_save_dir,
                filename="model_lightning_{epoch:02d}-{val_l1:.2f}",
                monitor="val_l1",
                mode="min",
                auto_insert_metric_name=True,
                save_last=True,
            )

            early_stopping_callback = EarlyStopping(
                monitor="val_l1",
                min_delta=0.00,
                patience=500,
                verbose=False,
                mode="min",
            )
            lr_monitor = LearningRateMonitor(logging_interval="step")
            logger_wb = WandbLogger(name="test_logs")

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
                logger=[logger_wb],
                precision=config["model"]["precision"],
            )

            trainer.fit(model, self.dm)
            trainer.test(model, self.dm)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default="bayes")
    # parser.add_argument("--on_gpu", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument(
        "-dataset_loc", type=str, default="../../dataset/qm_9_merge_3_qtaim.json"
    )
    parser.add_argument("-log_save_dir", type=str, default="./logs_lightning/")
    parser.add_argument("-project_name", type=str, default="hydro_lightning")
    parser.add_argument("-sweep_config", type=str, default="./sweep_config.json")
    parser.add_argument("-lmdb_root", type=str, default="./lmdb_out/")
    parser.add_argument(
        "--lmdb", default=False, action="store_true", help="use lmdb for dataset"
    )

    args = parser.parse_args()
    method = str(args.method)
    # on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    use_lmdb = bool(args.lmdb)
    lmdb_root = args.lmdb_root

    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    wandb_project_name = args.project_name
    sweep_config_loc = args.sweep_config
    sweep_config = {}
    sweep_params = json.load(open(sweep_config_loc, "r"))
    sweep_params["debug"] = {"values": [debug]}
    sweep_config["parameters"] = sweep_params

    if method == "bayes":
        sweep_config["method"] = method
        sweep_config["metric"] = {"name": "val_l1", "goal": "minimize"}

    # wandb loop
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    # print(sweep_config)
    training_obj = TrainingObject(
        sweep_config,
        log_save_dir,
        dataset_loc=dataset_loc,
        project_name=wandb_project_name,
        lmdb_root=lmdb_root,
        use_lmdb=use_lmdb,
    )

    print("method: {}".format(method))
    # print("on_gpu: {}".format(on_gpu))
    print("debug: {}".format(debug))
    print("dataset_loc: {}".format(dataset_loc))
    print("log_save_dir: {}".format(log_save_dir))
    print("wandb_project_name: {}".format(wandb_project_name))
    print("sweep_config_loc: {}".format(sweep_config_loc))

    wandb.agent(sweep_id, function=training_obj.train, count=300)
