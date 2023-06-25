import torch
import pytorch_lightning as pl

import itertools, copy, dgl
import numpy as np
import logging
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import lr_scheduler
import torchmetrics
import matplotlib.pyplot as plt

from bondnet.layer.gatedconv import GatedGCNConv, GatedGCNConv1, GatedGCNConv2
from bondnet.layer.readout import Set2SetThenCat
from bondnet.layer.utils import UnifySize
from bondnet.model.metric import (
    Metrics_Cross_Entropy,
    Metrics_Accuracy_Weighted,
)
from bondnet.data.utils import (
    _split_batched_output,
    mol_graph_to_rxn_graph,
)


logger = logging.getLogger(__name__)


class GatedGCNReactionNetworkLightningClassifier(pl.LightningModule):
    """
    Gated graph neural network model to predict molecular property.

    This model is similar to most GNN for molecular property such as MPNN and MEGNet.
    It iteratively updates atom, bond, and global features, then aggregates the
    features to form a representation of the molecule, and finally map the
    representation to a molecular property.


    Args:
        in_feats (dict): input feature size.
        embedding_size (int): embedding layer size.
        gated_num_layers (int): number of graph attention layer
        gated_hidden_size (list): hidden size of graph attention layers
        gated_num_fc_layers (int):
        gated_graph_norm (bool):
        gated_batch_norm(bool): whether to apply batch norm to gated layer.
        gated_activation (torch activation): activation fn of gated layers
        gated_residual (bool, optional): [description]. Defaults to False.
        gated_dropout (float, optional): dropout ratio for gated layer.
        fc_num_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_batch_norm (bool): whether to apply batch norm to fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_dropout (float, optional): dropout ratio for fc layer.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
    """

    def __init__(
        self,
        in_feats,
        embedding_size=32,
        gated_num_layers=2,
        gated_hidden_size=[64, 64, 32],
        gated_num_fc_layers=1,
        gated_graph_norm=False,
        gated_batch_norm=True,
        gated_activation="ReLU",
        gated_residual=True,
        gated_dropout=0.0,
        num_lstm_iters=6,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        fc_num_layers=2,
        fc_hidden_size=[32, 16],
        fc_batch_norm=False,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=3,
        conv="GatedGCNConv",
        learning_rate=1e-3,
        weight_decay=0.0,
        scheduler_name="ReduceLROnPlateau",
        warmup_epochs=10,
        max_epochs=2000,
        eta_min=1e-6,
        loss_fn="CrossEntropyLoss",
        device="cpu",
        wandb=True,
        augment=False,
        cat_weights=torch.Tensor([1.0, 1.0, 1.0]),
    ):
        if type(cat_weights) == list:
            cat_weights = torch.Tensor(cat_weights)

        super().__init__()
        self.learning_rate = learning_rate
        cat_weights = cat_weights.to(device)
        params = {
            "in_feats": in_feats,
            "embedding_size": embedding_size,
            "gated_num_layers": gated_num_layers,
            "gated_hidden_size": gated_hidden_size,
            "gated_num_fc_layers": gated_num_fc_layers,
            "gated_graph_norm": gated_graph_norm,
            "gated_batch_norm": gated_batch_norm,
            "gated_activation": gated_activation,
            "gated_residual": gated_residual,
            "gated_dropout": gated_dropout,
            "num_lstm_iters": num_lstm_iters,
            "num_lstm_layers": num_lstm_layers,
            "set2set_ntypes_direct": set2set_ntypes_direct,
            "fc_num_layers": fc_num_layers,
            "fc_hidden_size": fc_hidden_size,
            "fc_batch_norm": fc_batch_norm,
            "fc_activation": fc_activation,
            "fc_dropout": fc_dropout,
            "outdim": outdim,
            "conv": conv,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "scheduler_name": scheduler_name,
            "warmup_epochs": warmup_epochs,
            "max_epochs": max_epochs,
            "eta_min": eta_min,
            "loss_fn": loss_fn,
            "device": device,
            "wandb": wandb,
            "augment": augment,
            "cat_weights": cat_weights,
        }
        self.hparams.update(params)
        self.save_hyperparameters()

        if isinstance(gated_activation, str):
            gated_activation = getattr(nn, gated_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        # embedding layer
        self.embedding = UnifySize(in_feats, embedding_size)

        # gated layer
        if conv == "GatedGCNConv":
            conv_fn = GatedGCNConv
        elif conv == "GatedGCNConv1":
            conv_fn = GatedGCNConv1
        elif conv == "GatedGCNConv2":
            conv_fn = GatedGCNConv2
        else:
            raise ValueError()

        in_size = embedding_size
        self.gated_layers = nn.ModuleList()
        for i in range(gated_num_layers):
            self.gated_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=gated_hidden_size[i],
                    num_fc_layers=gated_num_fc_layers,
                    graph_norm=gated_graph_norm,
                    batch_norm=gated_batch_norm,
                    activation=gated_activation,
                    residual=gated_residual,
                    dropout=gated_dropout,
                )
            )
            in_size = gated_hidden_size[i]

        # set2set readout layer
        ntypes = ["atom", "bond"]
        in_size = [gated_hidden_size[-1]] * len(ntypes)

        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size  of in feature)
        readout_out_size = gated_hidden_size[-1] * 2 + gated_hidden_size[-1] * 2
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size
        for i in range(fc_num_layers):
            out_size = fc_hidden_size[i]
            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_dropout))

            in_size = out_size

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))
        self.fc_layers.append(nn.Softmax(dim=1))  # this makes it a classifier

        self.loss = self.loss_function()

        self.train_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams.outdim, multidim_average="global", average="macro"
        )
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams.outdim, multidim_average="global", average="macro"
        )
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams.outdim, multidim_average="global", average="macro"
        )

        self.train_acc = Metrics_Accuracy_Weighted(reduction="mean")
        self.val_acc = Metrics_Accuracy_Weighted(reduction="mean")
        self.test_acc = Metrics_Accuracy_Weighted(reduction="mean")

        self.val_cross = Metrics_Cross_Entropy(
            n_categories=self.hparams.outdim, reduction="sum"
        )
        self.test_cross = Metrics_Cross_Entropy(
            n_categories=self.hparams.outdim, reduction="sum"
        )
        self.train_cross = Metrics_Cross_Entropy(
            n_categories=self.hparams.outdim, reduction="sum"
        )

    def forward(self, graph, feats, norm_atom=None, norm_bond=None, reverse=False):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            reactions (list): a sequence of :class:`bondnet.data.reaction_network.Reaction`,
                each representing a reaction.
            norm_atom (2D tensor or None): graph norm for atom
            norm_bond (2D tensor or None): graph norm for bond

        Returns:
            2D tensor: of shape(N, M), where `M = outdim`.
        """

        if reverse:
            for key in feats:
                # multiply by -1
                feats[key] = -1 * feats[key]

        # embedding
        feats = self.embedding(feats)
        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # readout layer
        feats = self.readout_layer(graph, feats)

        for layer in self.fc_layers:
            feats = layer(feats)

        return feats

    def feature_before_fc(self, graph, feats, reactions, norm_atom, norm_bond):
        """
        Get the features before the final fully-connected.

        This is used for feature visualization.
        """
        # embedding
        feats = self.embedding(feats)
        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # readout layer
        feats = self.readout_layer(graph, feats)
        return feats

    def feature_at_each_layer(self, graph, feats, reactions, norm_atom, norm_bond):
        """
        Get the features at each layer before the final fully-connected layer.

        This is used for feature visualization to see how the model learns.

        Returns:
            dict: (layer_idx, feats), each feats is a list of
        """

        layer_idx = 0
        all_feats = dict()

        # embedding
        feats = self.embedding(feats)

        # store bond feature of each molecule
        fts = _split_batched_output(graph, feats["bond"])
        all_feats[layer_idx] = fts
        layer_idx += 1

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

            # store bond feature of each molecule
            fts = _split_batched_output(graph, feats["bond"])
            all_feats[layer_idx] = fts
            layer_idx += 1

        return all_feats

    def shared_step(self, batch, mode):
        # ========== compute predictions ==========
        batched_graph, label = batch
        nodes = ["atom", "bond", "global"]
        feats = {nt: batched_graph.nodes[nt].data["ft"] for nt in nodes}
        target = label["value"]
        target_aug = label["value_rev"]
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]

        stdev = label["scaler_stdev"]
        if self.device is not None:
            feats = {k: v.to(self.device) for k, v in feats.items()}
            target = target.to(self.device)
            norm_atom = norm_atom.to(self.device)
            norm_bond = norm_bond.to(self.device)
            stdev = stdev.to(self.device)
            if self.hparams.augment and not empty_aug:
                target_aug = target_aug.to(self.device)

        pred = self(
            batched_graph,
            feats,
            norm_bond=norm_bond,
            norm_atom=norm_atom,
        )
        # print("pred shape: ", pred.shape)

        if self.hparams.augment and not empty_aug:
            # target_aug_new_shape = (len(target_aug), 1)
            # target_aug = target_aug.view(target_aug_new_shape)
            pred_aug = self(
                batched_graph,
                feats,
                reverse=True,
                norm_bond=norm_bond,
                norm_atom=norm_atom,
            )
            all_loss = self.compute_loss(
                target=torch.cat((target, target_aug), axis=0),
                pred=torch.cat((pred, pred_aug), axis=0),
                weight=self.hparams.cat_weights,
            )

        else:
            # ========== compute losses ==========
            all_loss = self.compute_loss(
                target=target,
                pred=pred,
                weight=self.hparams.cat_weights,
            )
            # ========== logger the loss ==========

        self.log(
            f"{mode}_loss",
            all_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(label),
        )
        self.update_metrics(
            target=target, pred=pred, weight=self.hparams.cat_weights, mode=mode
        )

        return all_loss

    def loss_function(self):
        """
        Initialize loss function
        """
        print("using cross entropy")
        loss_fn = Metrics_Cross_Entropy(
            n_categories=self.hparams.outdim, reduction="sum"
        )
        return loss_fn

    def compute_loss(self, target, pred, weight):
        """
        Compute loss
        """
        return self.loss(target=target, preds=pred, weight=weight)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = self._config_lr_scheduler(optimizer)

        lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}

        return [optimizer], [lr_scheduler]

    def _config_lr_scheduler(self, optimizer):
        scheduler_name = self.hparams["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.lr_scheduler}")

        return scheduler

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        return self.shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        return self.shared_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        """
        Test step
        """
        return self.shared_step(batch, mode="test")

    def training_epoch_end(self, outputs):
        """
        Training epoch end
        """
        f1, acc, cross = self.compute_metrics(mode="train")
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_cross", cross, prog_bar=True)

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end
        """
        f1, acc, cross = self.compute_metrics(mode="val")
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_cross", cross, prog_bar=True)

    def test_epoch_end(self, outputs):
        """
        Test epoch end
        """
        f1, acc, cross = self.compute_metrics(mode="test")
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_cross", cross, prog_bar=True)

    def update_metrics(self, pred, target, weight, mode):
        if mode == "train":
            self.train_f1(preds=pred, target=target)
            self.train_acc(preds=pred, target=target)
            self.train_cross(preds=pred, target=target, weight=weight)

        elif mode == "val":
            self.val_f1(preds=pred, target=target)
            self.val_acc(preds=pred, target=target)
            self.val_cross(preds=pred, target=target, weight=weight)

        elif mode == "test":
            self.test_f1(preds=pred, target=target)
            self.test_acc(preds=pred, target=target)
            self.test_cross(preds=pred, target=target, weight=weight)

    def compute_metrics(self, mode):
        if mode == "train":
            f1 = self.train_f1.compute()
            acc = self.train_acc.compute()
            cross = self.train_cross.compute()
            self.train_f1.reset()
            self.train_acc.reset()
            self.train_cross.reset()

        elif mode == "val":
            f1 = self.val_f1.compute()
            acc = self.val_acc.compute()
            cross = self.val_cross.compute()
            self.val_f1.reset()
            self.val_acc.reset()
            self.val_cross.reset()

        elif mode == "test":
            f1 = self.test_f1.compute()
            acc = self.test_acc.compute()
            cross = self.test_cross.compute()
            self.test_f1.reset()
            self.test_acc.reset()
            self.test_cross.reset()

        return f1, acc, cross
