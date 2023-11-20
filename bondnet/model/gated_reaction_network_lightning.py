import torch
import torch
import pytorch_lightning as pl

import numpy as np
import logging
import torch.nn as nn
from torch.optim import lr_scheduler
import torchmetrics

import matplotlib.pyplot as plt

from bondnet.layer.gatedconv import GatedGCNConv, GatedGCNConv1, GatedGCNConv2
from bondnet.layer.readout import Set2SetThenCat
from bondnet.layer.utils import UnifySize

from bondnet.data.utils import (
    _split_batched_output,
    mol_graph_to_rxn_graph,
)

logger = logging.getLogger(__name__)


class GatedGCNReactionNetworkLightning(pl.LightningModule):
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
        outdim=1,
        conv="GatedGCNConv",
        learning_rate=1e-3,
        weight_decay=0.0,
        scheduler_name="ReduceLROnPlateau",
        warmup_epochs=10,
        max_epochs=2000,
        eta_min=1e-6,
        loss_fn="MSELoss",
        # device="cpu",
        wandb=True,
        augment=False,
        reactant_only=False,
    ):
        super().__init__()
        self.learning_rate = learning_rate
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
            # "device": device,
            "wandb": wandb,
            "augment": augment,
            "reactant_only": reactant_only,
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
            print("NB: using GatedGCNConv")
            conv_fn = GatedGCNConv
        elif conv == "GatedGCNConv1":
            print("NB: using GatedGCNConv1")
            conv_fn = GatedGCNConv1
        elif conv == "GatedGCNConv2":
            print("NB: using GatedGCNConv2")
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

        self.readout_out_size = readout_out_size

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
        # create stdev with the same number of output
        self.stdev = None

        self.loss = self.loss_function()

        self.train_r2 = torchmetrics.R2Score(
            num_outputs=1, multioutput="variance_weighted"
        )
        self.train_torch_l1 = torchmetrics.MeanAbsoluteError()
        self.train_torch_mse = torchmetrics.MeanSquaredError(square=False)

        self.val_r2 = torchmetrics.R2Score(
            num_outputs=1, multioutput="variance_weighted"
        )
        self.val_torch_l1 = torchmetrics.MeanAbsoluteError()
        self.val_torch_mse = torchmetrics.MeanSquaredError(square=False)

        self.test_r2 = torchmetrics.R2Score(
            num_outputs=1, multioutput="variance_weighted"
        )
        self.test_torch_l1 = torchmetrics.MeanAbsoluteError()
        self.test_torch_mse = torchmetrics.MeanSquaredError(square=False)

    def forward(
        self,
        graph,
        feats,
        reactions,
        norm_atom=None,
        norm_bond=None,
        reverse=False,
    ):
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
                feats[key] = -1 * feats[key]

        # embedding
        feats = self.embedding(feats)
        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # get device
        device = feats["bond"].device

        # convert mol graphs to reaction graphs
        graph, feats = mol_graph_to_rxn_graph(
            graph=graph,
            feats=feats,
            reactions=reactions,
            device=device,
            reverse=reverse,
            reactant_only=self.hparams.reactant_only,
        )

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
        # get device
        device = feats["bond"].device
        graph, feats = mol_graph_to_rxn_graph(
            graph=graph,
            feats=feats,
            reactions=reactions,
            reverse=False,
            device=device,
            reactant_only=self.hparams.reactant_only,
        )

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
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"].view(-1)
        target_aug = label["value_rev"].view(-1)
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        stdev = label["scaler_stdev"]
        mean = label["scaler_mean"]
        reactions = label["reaction"]

        if self.stdev is None:
            self.stdev = stdev[0]

        pred = self(
            graph=batched_graph,
            feats=feats,
            reactions=reactions,
            reverse=False,
            norm_bond=norm_bond,
            norm_atom=norm_atom,
        )

        pred = pred.view(-1)

        if self.hparams.augment and not empty_aug:
            # target_aug_new_shape = (len(target_aug), 1)
            # target_aug = target_aug.view(target_aug_new_shape)
            pred_aug = self(
                graph=batched_graph,
                feats=feats,
                reactions=reactions,
                reverse=True,
                norm_bond=norm_bond,
                norm_atom=norm_atom,
            )
            pred_aug = pred_aug.view(-1)
            all_loss = self.compute_loss(
                torch.cat((target, target_aug), axis=0),
                torch.cat((pred, pred_aug), axis=0),
            )

        else:
            # ========== compute losses ==========
            all_loss = self.compute_loss(pred, target)
            # ========== logger the loss ==========

        self.log(
            f"{mode}_loss",
            all_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(label),
            sync_dist=True,
        )
        self.update_metrics(target, pred, mode)

        return all_loss

    def loss_function(self):
        """
        Initialize loss function
        """

        if self.hparams.loss_fn == "mse":
            # loss_fn = WeightedMSELoss(reduction="mean")
            loss_fn = torchmetrics.MeanSquaredError()
        elif self.hparams.loss_fn == "smape":
            loss_fn = torchmetrics.SymmetricMeanAbsolutePercentageError()
        elif self.hparams.loss_fn == "mae":
            # loss_fn = WeightedL1Loss(reduction="mean")
            loss_fn = torchmetrics.MeanAbsoluteError()
        else:
            loss_fn = torchmetrics.MeanSquaredError()

        return loss_fn

    def compute_loss(self, target, pred):
        """
        Compute loss
        """

        return self.loss(target, pred)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
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
        # elif scheduler_name == "cosine":
        #    scheduler = LinearWarmupCosineAnnealingLR(
        #        optimizer,
        #        warmup_epochs=self.hparams.lr_scheduler["lr_warmup_step"],
        #        max_epochs=self.hparams.lr_scheduler["epochs"],
        #        eta_min=self.hparams.lr_scheduler["lr_min"],
        #    )
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
        # ========== compute predictions ==========
        batched_graph, label = batch
        nodes = ["atom", "bond", "global"]
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"].view(-1)
        target_aug = label["value_rev"].view(-1)
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        reactions = label["reaction"]

        if self.stdev is not None:
            stdev = self.stdev

        pred = self(
            batched_graph,
            feats,
            reactions,
            reverse=False,
            norm_bond=norm_bond,
            norm_atom=norm_atom,
        )

        pred = pred.view(-1)

        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        stdev = stdev.to(torch.float32)
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        stdev_np = stdev.detach().cpu().numpy()
        plt.scatter(pred_np * stdev_np, target_np * stdev_np)
        min_val = np.min([np.min(pred_np), np.min(target_np)]) - 0.5
        max_val = np.max([np.max(pred_np), np.max(target_np)]) + 0.5
        # manually compute mae and mse
        mae = np.mean(np.abs(pred_np * stdev_np - target_np * stdev_np))
        mse = np.mean((pred_np * stdev_np - target_np * stdev_np) ** 2)
        r2 = np.corrcoef(pred_np, target_np)[0, 1] ** 2
        print("-" * 30)
        print("MANUALLY COMPUTED METRICS")
        print("-" * 30)
        print("mae: ", mae)
        print("mse: ", mse)
        print("r2: ", r2)

        plt.title("Predicted vs. True")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("./{}.png".format("./test"))
        return self.shared_step(batch, mode="test")

    def training_epoch_end(self, outputs):
        """
        Training epoch end
        """
        r2, torch_l1, torch_mse = self.compute_metrics(mode="train")
        # self.log("train_l1", l1, prog_bar=True, sync_dist=True)
        self.log("train_r2", r2, prog_bar=True, sync_dist=True)
        self.log("train_l1", torch_l1, prog_bar=True, sync_dist=True)
        self.log("train_mse", torch_mse, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end
        """
        r2, torch_l1, torch_mse = self.compute_metrics(mode="val")
        # self.log("val_l1", l1, prog_bar=True, sync_dist=True)
        self.log("val_r2", r2, prog_bar=True, sync_dist=True)
        self.log("val_l1", torch_l1, prog_bar=True, sync_dist=True)
        self.log("val_mse", torch_mse, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        """
        Test epoch end
        """
        r2, torch_l1, torch_mse = self.compute_metrics(mode="test")
        # self.log("test_l1", l1, prog_bar=True, sync_dist=True)
        self.log("test_r2", r2, prog_bar=True, sync_dist=True)
        self.log("test_l1", torch_l1, prog_bar=True, sync_dist=True)
        self.log("test_mse", torch_mse, prog_bar=True, sync_dist=True)

    def update_metrics(self, pred, target, mode):
        if mode == "train":
            self.train_r2.update(pred, target)
            self.train_torch_l1.update(pred, target)
            self.train_torch_mse.update(pred, target)
        elif mode == "val":
            self.val_r2.update(pred, target)
            self.val_torch_l1.update(pred, target)
            self.val_torch_mse.update(pred, target)

        elif mode == "test":
            self.test_r2.update(pred, target)
            self.test_torch_l1.update(pred, target)
            self.test_torch_mse.update(pred, target)

    def compute_metrics(self, mode):
        if mode == "train":
            r2 = self.train_r2.compute()
            torch_l1 = self.train_torch_l1.compute()
            torch_mse = self.train_torch_mse.compute()
            self.train_r2.reset()
            self.train_torch_l1.reset()
            self.train_torch_mse.reset()

        elif mode == "val":
            # l1 = self.val_l1.compute()
            r2 = self.val_r2.compute()
            torch_l1 = self.val_torch_l1.compute()
            torch_mse = self.val_torch_mse.compute()
            self.val_r2.reset()
            # self.val_l1.reset()
            self.val_torch_l1.reset()
            self.val_torch_mse.reset()

        elif mode == "test":
            # l1 = self.test_l1.compute()
            r2 = self.test_r2.compute()
            torch_l1 = self.test_torch_l1.compute()
            torch_mse = self.test_torch_mse.compute()
            self.test_r2.reset()
            # self.test_l1.reset()
            self.test_torch_l1.reset()
            self.test_torch_mse.reset()

        if self.stdev is not None:
            # print("stdev", self.stdev)
            torch_l1 = torch_l1 * self.stdev
            torch_mse = torch_mse * self.stdev * self.stdev
        else:
            print("scaling is 1!" + "*" * 20)

        # if self.mean is not None:
        #    #torch_l1 = torch_l1 + self.mean
        #    # torch_mse = torch_mse + self.mean

        return r2, torch_l1, torch_mse
