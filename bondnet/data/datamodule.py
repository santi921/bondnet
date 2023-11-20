import pytorch_lightning as pl
import torch.distributed as dist
import os
from bondnet.data.dataset import (
    ReactionNetworkDatasetGraphs,
    train_validation_test_split,
)

# from bondnet.data.lmdb import LmdbDataset, CRNs2lmdb


from bondnet.data.dataloader import collate_parallel, DataLoaderReactionNetworkParallel
from bondnet.model.training_utils import get_grapher

"""

class BondNetLightningDataModuleLMDB(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        if config["dataset"].get("train_lmdb") == None:
            self.train_lmdb = "train_data.lmdb"
        else:
            self.train_lmdb = config["dataset"]["train_lmdb"]

        if config["dataset"].get("val_lmdb") == None:
            self.val_lmdb = "val_data.lmdb"
        else:
            self.val_lmdb = config["dataset"]["val_lmdb"]

        if config["dataset"].get("test_data_lmdb") == None:
            self.test_lmdb = "test_data.lmdb"
        else:
            self.test_lmdb = config["dataset"]["test_lmdb"]

        self.config = config
        self.prepared = False

    def _check_exists(self, data_folder: str) -> bool:
        if self.config["dataset"]["overwrite"]:
            return False

        existing = True
        for fname in (self.train_lmdb, self.val_lmdb, self.test_lmdb):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self):
        # https://github.com/Lightning-AI/lightning/blob/6d888b5ce081277a89dc2fb9a2775b81d862fe54/src/lightning/pytorch/demos/mnist_datamodule.py#L90
        if not self.prepared and not self._check_exists(
            self.config["dataset"]["lmdb_dir"]
        ):
            # Load json file, preprocess data, and write to lmdb file

            entire_dataset = ReactionNetworkDatasetGraphs(
                grapher=get_grapher(self.config["model"]["extra_features"]),
                file=self.config["dataset"]["data_dir"],
                target=self.config["dataset"]["target_var"],
                classifier=self.config["model"]["classifier"],
                classif_categories=self.config["model"]["classif_categories"],
                filter_species=self.config["model"]["filter_species"],
                filter_outliers=self.config["model"]["filter_outliers"],
                filter_sparse_rxns=False,
                debug=self.config["model"]["debug"],
                extra_keys=self.config["model"]["extra_features"],
                extra_info=self.config["model"]["extra_info"],
            )
            # print("done loading dataset" * 10)
            train_CRNs, val_CRNs, test_CRNs = train_validation_test_split(
                entire_dataset,
                validation=self.config["optim"]["val_size"],
                test=self.config["optim"]["test_size"],
            )

            for CRNs_i, lmdb_i in zip(
                [train_CRNs, val_CRNs, test_CRNs],
                [self.train_lmdb, self.val_lmdb, self.test_lmdb],
            ):
                CRNs2lmdb(
                    CRNsDb=CRNs_i,
                    lmdb_dir=self.config["dataset"]["lmdb_dir"],
                    num_workers=self.config["optim"]["num_workers"],
                    lmdb_name=lmdb_i,
                )
            # print("done creating lmdb" * 10)
            self.prepared = True
            return entire_dataset._feature_size, entire_dataset._feature_name

        else:
            train_dataset = LmdbDataset(
                {"src": f"{self.config['dataset']['lmdb_dir']}" + self.train_lmdb}
            )
            self.prepared = True
            return train_dataset.feature_size, train_dataset.feature_name

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_ds = LmdbDataset(
                {"src": f"{self.config['dataset']['lmdb_dir']}" + self.train_lmdb}
            )
            self.val_ds = LmdbDataset(
                {"src": f"{self.config['dataset']['lmdb_dir']}" + self.val_lmdb}
            )

        if stage in ("test", "predict"):
            self.test_ds = LmdbDataset(
                {"src": f"{self.config['dataset']['lmdb_dir']}" + self.test_lmdb}
            )

        # if stage in (None, "fit"):

        #     #load lmdb file from lmdb dir
        #     _lmdb = LmdbDataset({"src": f"{self.config['dataset']['lmdb_dir']}" + self.train_lmdb})

        #     # Split the dataset into train, val, test
        #     self.train_ds, self.val_ds, self.test_ds = train_validation_test_split(
        #         _lmdb,
        #         validation=self.config['optim']['val_size'],
        #         test=self.config['optim']['test_size']
        #     )

    def train_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
        )

    # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def test_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.test_ds,
            batch_size=len(self.test_ds),
            shuffle=False,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
        )

    # return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
        )

"""


class BondNetLightningDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prepared = False

    def prepare_data(self):
        entire_dataset = ReactionNetworkDatasetGraphs(
            grapher=get_grapher(self.config["model"]["extra_features"]),
            file=self.config["dataset"]["data_dir"],
            target=self.config["dataset"]["target_var"],
            classifier=self.config["model"]["classifier"],
            classif_categories=self.config["model"]["classif_categories"],
            filter_species=self.config["model"]["filter_species"],
            filter_outliers=self.config["model"]["filter_outliers"],
            filter_sparse_rxns=False,
            debug=self.config["model"]["debug"],
            extra_keys=self.config["model"]["extra_features"],
            extra_info=self.config["model"]["extra_info"],
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = train_validation_test_split(
            entire_dataset,
            validation=self.config["optim"]["val_size"],
            test=self.config["optim"]["test_size"],
        )

        # print("done creating lmdb" * 10)
        self.prepared = True
        return entire_dataset._feature_size, entire_dataset._feature_name

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def train_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.test_ds,
            batch_size=len(self.test_ds),
            shuffle=False,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
        )
