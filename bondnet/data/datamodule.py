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
from bondnet.data.dataloader import DataLoaderReaction, DataLoaderReactionLMDB
from bondnet.data.reaction_network import ReactionNetworkLMDB

torch.multiprocessing.set_sharing_strategy("file_system")

class BondNetLightningDataModuleLMDB(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        if config["dataset"].get("train_lmdb") == None:
            self.train_lmdb = "train_rxn.lmdb"
        else:
            self.train_lmdb = config["dataset"]["train_lmdb"]

        if config["dataset"].get("val_lmdb") == None:
            self.val_lmdb = "val_data.lmdb"
        else:
            self.val_lmdb = config["dataset"]["val_lmdb"]

        if "test" in config["dataset"]:
            if config["dataset"].get("test_data_lmdb") == None:
                self.test_lmdb = "test_data.lmdb"
            else:
                self.test_lmdb = config["dataset"]["test_lmdb"]

        self.config = config
        self.prepared = False

    def _check_exists(self, data_folder: str) -> bool:
        if bool(self.config["dataset"]["overwrite"]):
            return False

        #for fname in (self.train_lmdb, self.val_lmdb, self.test_lmdb):
        #    existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        if not os.path.isfile(os.path.join(data_folder, self.train_lmdb)):
            return False 
        if not os.path.isfile(os.path.join(data_folder, self.val_lmdb)):
            return False
        return True

    def prepare_data(self):
        if not self.prepared and not self._check_exists(
            self.config["dataset"]["lmdb_dir"]
        ):
            # Load json file, preprocess data, and write to lmdb file

            entire_dataset = ReactionDatasetGraphs(
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
            
            construct_lmdb_and_save(
                entire_dataset, 
                self.config['dataset']['lmdb_dir'], 
                workers=1
                )
            if not bool(self.config["optim"]["test_only"]):
                if self.config["optim"]["test_size"] == 0.0:
                    train_dataset, val_dataset = train_validation_test_split(
                        entire_dataset,
                        validation=self.config["optim"]["val_size"],
                        random_seed=self.config["optim"]["random_seed"],
                        test=0.0,
                    )

                else:
                    train_dataset, val_dataset, test_dataset = train_validation_test_split(
                        entire_dataset,
                        validation=self.config["optim"]["val_size"],
                        test=self.config["optim"]["test_size"],
                        random_seed=self.config["optim"]["random_seed"],
                    )
                    construct_lmdb_and_save(
                        test_dataset, 
                        self.config['dataset']['lmdb_dir'] + "/test/", 
                        workers=1
                    )
                    self.test_rxn_dataset = LmdbReactionDataset(
                        {"src": f"{self.config['dataset']['lmdb_dir']}" + "/test/reaction.lmdb"}
                    )

                    self.test_molecule_dataset = LmdbMoleculeDataset(
                        {
                            "src": f"{self.config['dataset']['lmdb_dir']}" + "/test/molecule.lmdb", 
                            "transform": TransformMol
                        }

                    )
                    self.test_rxn_dataset = LmdbReactionDataset(
                        {"src": f"{self.config['dataset']['lmdb_dir']}" + "/test/reaction.lmdb"}
                    )

                construct_lmdb_and_save(
                    train_dataset, 
                    self.config['dataset']['lmdb_dir'] + "/train/", 
                    workers=1
                )

                construct_lmdb_and_save(
                    val_dataset, 
                    self.config['dataset']['lmdb_dir'] + "/val/", 
                    workers=1
                )

                self.train_rxn_dataset = LmdbReactionDataset(
                    {"src": f"{self.config['dataset']['lmdb_dir']}" + "/train/reaction.lmdb"}
                )
                self.val_rxn_dataset = LmdbReactionDataset(
                    {"src": f"{self.config['dataset']['lmdb_dir']}" + "/val/reaction.lmdb"}
                )

                self.train_molecule_dataset = LmdbMoleculeDataset(
                    {
                        "src": f"{self.config['dataset']['lmdb_dir']}" + "/train/molecule.lmdb", 
                        "transform": TransformMol
                    }

                )
                self.val_molecule_dataset = LmdbMoleculeDataset(
                    {
                        "src": f"{self.config['dataset']['lmdb_dir']}" + "/val/molecule.lmdb", 
                        "transform": TransformMol
                    }
                )

                self.prepared = True
                return entire_dataset._feature_size, entire_dataset._feature_name

        else:
            
            self.test_rxn_dataset = LmdbReactionDataset(
                {"src": f"{self.config['dataset']['lmdb_dir']}" + "/reaction.lmdb"}
            )

            self.test_molecule_dataset = LmdbMoleculeDataset(
                {
                    "src": f"{self.config['dataset']['lmdb_dir']}" + "/molecule.lmdb", 
                    "transform": TransformMol
                }

            )
        
            self.prepared = True
            return train_dataset.feature_size, train_dataset.feature_name



    def setup(self, stage):
        if stage in (None, "fit", "validate"):

            self.train_ds = ReactionDatasetLMDBDataset(ReactionNetworkLMDB(self.train_molecule_dataset, self.train_rxn_dataset))
            self.val_ds = ReactionDatasetLMDBDataset(ReactionNetworkLMDB(self.val_molecule_dataset, self.val_rxn_dataset))
            self.test_ds = ReactionDatasetLMDBDataset(ReactionNetworkLMDB(self.test_molecule_dataset, self.test_rxn_dataset))


        if stage in ("test", "predict"):
            self.test_ds = ReactionDatasetLMDBDataset(ReactionNetworkLMDB(self.test_molecule_dataset, self.test_rxn_dataset))



    def train_dataloader(self):
        return DataLoaderReactionLMDB(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
            num_workers=self.config["optim"]["num_workers"],
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
