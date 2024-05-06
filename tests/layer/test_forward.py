import torch
import pytorch_lightning as pl

from bondnet.model.training_utils import (
    load_model_lightning,
)
from bondnet.data.datamodule import BondNetLightningDataModule, BondNetLightningDataModuleLMDB
from bondnet.data.utils import process_batch_mol_rxn
from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)
from bondnet.test_utils import get_defaults

# suppress warnings
import warnings
import time 
import dgl

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.trainer.data_loading",
    lineno=102,
)


torch.set_float32_matmul_precision("medium")  # might have to disable on older GPUs



def test_forward():

    config = {
        "dataset": {
            "log_save_dir": "./model_log/",
            "train_lmdb": "../data/testdata/lmdb/",
            "val_lmdb": "../data/testdata/lmdb/",
            "test_lmdb": "../data/testdata/lmdb/",
            "target_var": "value",
            "overwrite": False,
            "no_splits": False
        },
        "model": {
            "extra_features": {},
            "extra_info": [],
            "debug": False,
            "classifier": False,
            "classif_categories": 3,
            "filter_species": [3, 6],
            "filter_outliers": False,
            "filter_sparse_rxns": False,
            "restore": False,
        },
        "optim": {
            "val_size": 0.1,
            "test_size": 0.1,
            "batch_size": 64,
            "num_workers": 3,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    config_model = get_defaults()
    # update config with model settings
    for key, value in config_model["model"].items():
        config["model"][key] = value

    dm = BondNetLightningDataModuleLMDB(config)
    feat_size, feat_name = dm.prepare_data()
    dm.setup(stage="fit")
    config["model"]["in_feats"] = feat_size
    model = load_model_lightning(config["model"], load_dir="./test_save_load/")
    
    dataloader = dm.train_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nodes = ["atom", "bond", "global"]

    for it, batch in enumerate(dataloader):

        batched_graph_in, label, batch_data = batch

        nodes = ["atom", "bond", "global"]
        feats_in = {nt: batched_graph_in.nodes[nt].data["ft"] for nt in nodes}
        target = label["value"].view(-1)
        target_aug = label["value_rev"].view(-1)
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        stdev = label["scaler_stdev"]
        reactions = len(target)

        feats_out_a = model(
            graph=batched_graph_in, 
            feats=feats_in,
            reactions=reactions,
            norm_atom=norm_atom,
            norm_bond=norm_bond,
            reverse=False,
            batch_data=batch_data
        )

        if model.stdev is None:
            model.stdev = stdev[0]


        # embedding
        feats_out_b = model.embedding(feats_in)
        # gated layer
        for layer in model.gated_layers:
            feats_out_b = layer(batched_graph_in, feats_out_b, norm_atom, norm_bond)

        # get device
        device = feats_out_b["bond"].device

        # convert mol graphs to reaction graphs
        feats_out_b = process_batch_mol_rxn(
                    feats = feats_out_b,
                    reactions = reactions,
                    device = device,
                    reverse = False,
                    batch_data=batch_data
                )

        # readout layer
        feats_out_b = model.readout_layer(batch_data["batched_rxn_graphs"], feats_out_b)

        for layer in model.fc_layers:
            feats_out_b = layer(feats_out_b)




        #compare outputs
        #print max difference 
        # print max difference 
        # assert they both have same output dims 

        assert feats_out_a.shape == feats_out_b.shape
