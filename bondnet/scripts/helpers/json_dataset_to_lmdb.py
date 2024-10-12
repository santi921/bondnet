import argparse
import torch
import json
import numpy as np
import argparse
import dgl
import torch
import tempfile
from copy import deepcopy

from bondnet.data.lmdb import construct_lmdb_and_save_reaction_dataset
from bondnet.data.dataset import ReactionDatasetGraphs
from bondnet.utils import seed_torch
from bondnet.model.training_utils import get_grapher
from bondnet.data.dataset import train_validation_test_split



torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
seed_torch()
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="write to molecule or reaction lmdb dataset with multiprocess"
    )
    parser.add_argument(
        "--dataset_loc",
        type=str,
        help="location of json file containing dataset",
    )

    parser.add_argument(
        "-config_loc",
        type=str,
        help="location of json file containing config",
    )

    parser.add_argument(
        "-lmdb_dir",
        type=str,
        help="location of lmdb directory",
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="whether or not to presplit the dataset into train, val, and test sets",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )

    args = parser.parse_args()
    config_loc = args.config_loc
    dataset_loc = args.dataset_loc
    lmdb_dir = args.lmdb_dir
    #workers = int(args.workers)
    debug = bool(args.debug)
    split = bool(args.split)

    # read json
    with open(config_loc) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extra_keys = config["extra_features"]
    precision = config["precision"]

    if precision == "16" or precision == "32":
        precision = int(precision)

    extra_keys = config["extra_features"]


    dataset = ReactionDatasetGraphs(
        grapher=get_grapher(extra_keys),
        file=dataset_loc,
        target=config["target_var"],
        classifier=config["classifier"],
        classif_categories=config["classif_categories"],
        filter_species=config["filter_species"],
        filter_outliers=config["filter_outliers"],
        filter_sparse_rxns=False,
        debug=debug,
        species=config["species"],
        extra_keys=extra_keys,
        extra_info=config["extra_info"],
    )

    if split == True:
        if config["test_size"]>0.0:
            train_dataset, val_dataset, test_dataset = train_validation_test_split(
                dataset,
                validation=config["val_size"],
                test=config["test_size"],
                random_seed=config["random_seed_split"],
                lmdb=True
            )

            construct_lmdb_and_save_reaction_dataset(dataset, lmdb_dir)
            construct_lmdb_and_save_reaction_dataset(val_dataset, lmdb_dir+"/val/")
            construct_lmdb_and_save_reaction_dataset(train_dataset, lmdb_dir+"/train/")
            construct_lmdb_and_save_reaction_dataset(test_dataset, lmdb_dir+"/test/")

        else: 

            train_dataset, val_dataset = train_validation_test_split(
                dataset,
                test=0.0,
                validation=config["val_size"],
                random_seed=config["random_seed_split"],
                lmdb=True
            )
            construct_lmdb_and_save_reaction_dataset(dataset, lmdb_dir)
            construct_lmdb_and_save_reaction_dataset(val_dataset, lmdb_dir+"/val/")
            construct_lmdb_and_save_reaction_dataset(train_dataset, lmdb_dir+"/train/")


    else: 
        construct_lmdb_and_save_reaction_dataset(dataset, lmdb_dir)
