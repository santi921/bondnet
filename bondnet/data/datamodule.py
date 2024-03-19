import pytorch_lightning as pl
import torch
import torch.distributed as dist
import os
from bondnet.data.dataset import (
    train_validation_test_split
)


from bondnet.data.dataloader import DataLoaderReactionLMDB, DataLoaderReaction
from bondnet.model.training_utils import get_grapher
from bondnet.data.lmdb import construct_lmdb_and_save, TransformMol
from bondnet.data.dataloader import DataLoaderReactionLMDB, DataLoaderReaction
from bondnet.data.dataset import (
    ReactionDatasetGraphs, 
    ReactionDatasetLMDBDataset, 
    LmdbReactionDataset, 
    LmdbMoleculeDataset 
)
from bondnet.data.reaction_network import ReactionLMDB

from bondnet.data.dataloader import DataLoaderReaction, DataLoaderReactionLMDB
#from bondnet.data.reaction_network import ReactionNetworkLMDB

torch.multiprocessing.set_sharing_strategy("file_system")


class BondNetLightningDataModuleLMDB(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        #self.prepared = False
        self.train_lmdb_loc = config["dataset"]["train_lmdb"]
        
        if "val_lmdb" in self.config["dataset"]:
            self.val_lmdb_loc = config["dataset"]["val_lmdb"]

        if "test_lmdb" in self.config["dataset"]:
            self.test_lmdb_loc = config["dataset"]["test_lmdb"]


    def prepare_data(self):
        if "test_lmdb" in self.config["dataset"]:
            self.test_rxn_dataset = LmdbReactionDataset(
                config = {
                    "src": os.path.join(self.test_lmdb_loc, "reaction.lmdb")
                }
            )

            self.test_molecule_dataset = LmdbMoleculeDataset(
                config={
                    "src": os.path.join(self.test_lmdb_loc, "molecule.lmdb")
                },
                transform=TransformMol

            )
            self.test_dataset = ReactionLMDB(self.test_molecule_dataset, self.test_rxn_dataset)
        
        if "val_lmdb" in self.config["dataset"]:
            config_val = {
                "src": os.path.join(self.val_lmdb_loc, "molecule.lmdb")
            }
            config_val_rxn = {
                "src": os.path.join(self.val_lmdb_loc, "reaction.lmdb")
            }

            self.val_rxn_dataset = LmdbReactionDataset(
                config = config_val_rxn
            )
            self.val_molecule_dataset = LmdbMoleculeDataset(
                config=config_val,
            transform=TransformMol
            )
        
            self.val_dataset = ReactionLMDB(self.val_molecule_dataset, self.val_rxn_dataset)


        config_train = {
            "src": os.path.join(self.train_lmdb_loc, "molecule.lmdb")
        }

        config_train_rxn = {
            "src": os.path.join(self.train_lmdb_loc, "reaction.lmdb")
        }


        self.train_rxn_dataset = LmdbReactionDataset(
            config = config_train_rxn
        )
        
        self.train_molecule_dataset = LmdbMoleculeDataset(
            config=config_train, transform=TransformMol

        )


        self.train_dataset = ReactionLMDB(self.train_molecule_dataset, self.train_rxn_dataset)
        
        return self.train_rxn_dataset.feature_size, self.train_rxn_dataset.feature_name


    def setup(self, stage):
        if stage in (None, "fit", "validate"):
    
            self.train_ds = ReactionDatasetLMDBDataset(self.train_dataset)
            if "val_lmdb" in self.config["dataset"]:
                self.val_ds = ReactionDatasetLMDBDataset(self.val_dataset)
            #self.test_ds = ReactionDatasetLMDBDataset(self.test_dataset)


        if stage in ("test", "predict"):
            self.test_ds = ReactionDatasetLMDBDataset(self.test_dataset)


    def train_dataloader(self):
        return DataLoaderReactionLMDB(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=self.config["optim"]["pin_memory"],
            persistent_workers=self.config["optim"]["persistent_workers"],
        )


    def test_dataloader(self):
        return DataLoaderReactionLMDB(
            dataset=self.test_ds,
            batch_size=len(self.test_ds),
            shuffle=False,
            num_workers=self.config["optim"]["num_workers"],
        )


    def val_dataloader(self):
        return DataLoaderReactionLMDB(
            dataset=self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            num_workers=self.config["optim"]["num_workers"],
        )



class BondNetLightningDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prepared = False

    def prepare_data(self):
        if self.prepared:
            return self.entire_dataset._feature_size, self.entire_dataset._feature_name
        
        else:
            self.entire_dataset = ReactionDatasetGraphs(
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
                self.entire_dataset,
                validation=self.config["optim"]["val_size"],
                test=self.config["optim"]["test_size"],
            )

            # print("done creating lmdb" * 10)
            self.prepared = True
            return self.entire_dataset._feature_size, self.entire_dataset._feature_name

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def train_dataloader(self):
        return DataLoaderReaction(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=self.config["optim"]["pin_memory"],
            persistent_workers=self.config["optim"]["persistent_workers"],
        )

    def test_dataloader(self):
        return DataLoaderReaction(
            dataset=self.test_ds,
            batch_size=len(self.test_ds),
            shuffle=False,
            num_workers=self.config["optim"]["num_workers"],
        )

    def val_dataloader(self):
        return DataLoaderReaction(
            dataset=self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            num_workers=self.config["optim"]["num_workers"]
        )
