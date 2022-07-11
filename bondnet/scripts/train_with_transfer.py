import argparse
import torch
import time
import warnings
import datetime
import sysconfig
import sys
import numpy as np
from rdkit import RDLogger
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from torch.nn import MSELoss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bondnet.data.dataset import ReactionNetworkDataset
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import (
    AtomFeaturizerMinimum,
    BondAsNodeFeaturizerMinimum,
    GlobalFeaturizer,
    BondAsNodeFeaturizerFull,
)
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.dataset import train_validation_test_split
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.scripts.create_label_file import read_input_files
from bondnet.model.metric import WeightedL1Loss, EarlyStopping
from bondnet.utils import (
    load_checkpoints,
    save_checkpoints,
    seed_torch,
    pickle_dump,
    yaml_dump,
    parse_settings,
)
from bondnet.prediction.load_model import load_dataset, load_model

from torchsummary import summary


def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn, device=None):
    """
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (bg, label) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        stdev = label["scaler_stdev"]

        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target = target.to(device)
            norm_atom = norm_atom.to(device)
            norm_bond = norm_bond.to(device)
            stdev = stdev.to(device)

        pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
        pred = pred.view(-1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target, stdev).detach().item()
        count += len(target)

    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy


def evaluate(model, nodes, data_loader, metric_fn, device=None):
    """
    Evaluate the accuracy of an validation set of test set.

    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        for it, (bg, label) in enumerate(data_loader):
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            stdev = label["scaler_stdev"]

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1)

            accuracy += metric_fn(pred, target, stdev).detach().item()
            count += len(target)

    return accuracy / count


def get_grapher():
    atom_featurizer = AtomFeaturizerMinimum()
    # bond_featurizer = BondAsNodeFeaturizerMinimum()
    bond_featurizer = BondAsNodeFeaturizerFull()
    # our example dataset contains molecules of charges -1, 0, and 1
    global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer)

    return grapher


if __name__ == "__main__":
    test = True
    # seed random number generators
    seed_torch()
    dict_ret = parse_settings()
    feature_names = ["atom", "bond", "global"]
    set2set_ntypes_direct = ["global"]

    if dict_ret["on_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_ret["gpu"] = device
    else:
        dict_ret["gpu"] = "cpu"

    if bool(dict_ret["test"]):
        path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"

        mols, attrs, labels = read_input_files(
            path_mg_data + "mg_struct_bond_rgrn.sdf",
            path_mg_data + "mg_feature_bond_rgrn.yaml",
            path_mg_data + "mg_label_bond_rgrn.yaml",
        )
    else:
        path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"

        # todo
        mols, attrs, labels = read_input_files(
            path_mg_data + "molecules_libe.sdf",
            path_mg_data + "examples/train/molecule_attributes_libe.yaml",
            path_mg_data + "examples/train/reactions_libe.yaml",
        )
    model = load_model(dict_ret['model_path'])
    dataset = load_dataset(
        dict_ret["model_path"],
        molecules=mols,
        labels=labels,
        extra_features=attrs)
    feature_size = dataset.feature_size

    trainset, valset, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)
    # we train with a batch size of 100
    train_loader = DataLoaderReactionNetwork(trainset, batch_size=dict_ret["batch_size"],shuffle=True)
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderReactionNetwork(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoaderReactionNetwork(testset, batch_size=bs, shuffle=False)

    model.gated_layers.requires_grad_(False)
    checkpoint = torch.load("checkpoint.pkl")
    print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Trainable Model Params: {}".format(params))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=dict_ret["lr"], weight_decay=dict_ret["weight_decay"]
    )

    loss_func = MSELoss(reduction="mean")
    metric = WeightedL1Loss(reduction="sum")
    t1 = time.time()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)


    print("Number of Trainable Model Params: {}".format(params))
    print("-" * 20 + "now disabling gradients" + "-" * 20)
    # model.fc_layers.requires_grad_(False)
    # model.readout_layer.requires_grad_(False)

    best = 1e10

    # main training loop
    print("# Epoch     Loss         TrainAcc        ValAcc")


    for epoch in range(dict_ret["start_epoch"], dict_ret["epochs"]):
        ti = time.time()

        # In distributed mode, calling the set_epoch method is needed to make shuffling
        # work; each process will use the same random seed otherwise.

        # train
        loss, train_acc = train(
            optimizer,
            model,
            feature_names,
            train_loader,
            loss_func,
            metric,
            dict_ret["gpu"],
        )

        # bad, we get nan
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. Existing")
            sys.stdout.flush()
            sys.exit(1)

        # evaluate
        val_acc = evaluate(model, feature_names, val_loader, metric, dict_ret["gpu"])

        if stopper.step(val_acc):
            pickle_dump(
                best, dict_ret["save_hyper_params"]
            )  # save results for hyperparam tune
            break

        scheduler.step(val_acc)

        is_best = val_acc < best
        if is_best:
            best = val_acc

        # save checkpoint

        misc_objs = {"best": best, "epoch": epoch}
        state_dict_objs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

        save_checkpoints(
            state_dict_objs,
            misc_objs,
            is_best,
            msg=f"epoch: {epoch}, score {val_acc}",
        )

        tt = time.time() - ti

        print(
            "{:5d}   {:12.6e}   {:12.6e}   {:12.6e}   {:.2f}".format(
                epoch, loss, train_acc, val_acc, tt
            )
        )
        if epoch % 10 == 0:
            sys.stdout.flush()

    t2 = time.time()

    test_acc = evaluate(model, feature_names, test_loader, metric)

    print("TestAcc: {:12.6e}".format(test_acc))
    print("Time to Train: {:5.1f} seconds".format(float(t2 - t1)))
    print("Number of Trainable Model Params: {}".format(params))
