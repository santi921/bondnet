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
from bondnet.layer.readout import (
    Set2SetThenCat, 
    GlobalAttentionPoolingThenCat, 
    MeanPoolingThenCat,
    WeightAndMeanThenCat
)

from bondnet.layer.utils import UnifySize

from bondnet.data.utils import (
    _split_batched_output,
    mol_graph_to_rxn_graph,
    unbatch_mol_graph_to_rxn_graph,
    process_batch_mol_rxn,
)

import time
import dgl

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
        conv (str): type of convolution layer. Currently support "GatedGCNConv",
        readout (str): type of readout layer. Currently support "Set2SetThenCat", "Mean", "WeightedMean", "Attention"
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
        readout="Set2SetThenCat",
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
            "readout": readout,
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
        
        # readout layer
        if readout == "Set2SetThenCat":
            print("NB: using Set2SetThenCat")
            readout_fn = Set2SetThenCat
        elif readout == "Mean":
            print("NB: using Mean")
            readout_fn = MeanPoolingThenCat
        elif readout == "WeightedMean":
            print("NB: using WeightedMean")
            readout_fn = WeightAndMeanThenCat
        elif readout == "Attention":
            print("NB: using Attention")
            readout_fn = GlobalAttentionPoolingThenCat
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
        self.readout_out_size = 0
        ntypes = ["atom", "bond"]
        in_size = [gated_hidden_size[-1]] * len(ntypes)

        if self.hparams.readout == "Set2SetThenCat":
            self.readout_layer = readout_fn(
                n_iters=num_lstm_iters,
                n_layer=num_lstm_layers,
                ntypes=ntypes,
                in_feats=in_size,
                ntypes_direct_cat=set2set_ntypes_direct,
            )
            # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
            # feature twice the the size  of in feature)
            self.readout_out_size = gated_hidden_size[-1] * 2 + gated_hidden_size[-1] * 2
            # for global feat
            if set2set_ntypes_direct is not None:
                self.readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

        else:
            # print("other readout used")
            self.readout_layer = readout_fn(
                ntypes=ntypes,
                in_feats=in_size,
                ntypes_direct_cat=set2set_ntypes_direct,
            )

            self.readout_out_size = gated_hidden_size[-1] + gated_hidden_size[-1] 

            if set2set_ntypes_direct is not None:
                self.readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

            #for i in self.hparams.pooling_ntypes:
            #    if i in self.hparams.ntypes_pool_direct_cat:
            #        self.readout_out_size += self.conv_out_size[i]
            #    else:
            #        self.readout_out_size += self.conv_out_size[i]


        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = self.readout_out_size
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
        atom_batch_indices = None,
        bond_batch_indices = None,
        global_batch_indices = None,
        batched_rxn_graphs = None, 
        batched_atom_reactant = None, 
        batched_atom_product = None, 
        batched_bond_reactant = None, 
        batched_bond_product = None,
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
        #breakpoint()
        if reverse:
            for key in feats:
                feats[key] = -1 * feats[key]

        # embedding
        feats = self.embedding(feats)
        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)
        #atom: 2584, bond: 2618, global: 80
        # get device
        device = feats["bond"].device
        #breakpoint()
        # convert mol graphs to reaction graphs

        # graph, feats = unbatch_mol_graph_to_rxn_graph(
        #     graph = graph,
        #     feats = feats,
        #     reactions = reactions,
        #     device = device,
        #     reverse = reverse,
        #     reactant_only=self.hparams.reactant_only,
        #     atom_batch_indices = atom_batch_indices,
        #     bond_batch_indices = bond_batch_indices,
        #     global_batch_indices = global_batch_indices,
        #     mappings = None, #!needed for atom and bond.
        #     has_bonds = None, #!needed for atom and bond.
        #     ntypes=("global", "atom", "bond"),
        #     ft_name="ft",
        #     zero_fts=False,
        #     empty_graph_fts=True,
        # )

        feats = process_batch_mol_rxn(
            graph = graph,
            feats = feats,
            reactions = reactions,
            device = device,
            reverse = reverse,
            reactant_only=self.hparams.reactant_only,
            atom_batch_indices = atom_batch_indices,
            bond_batch_indices = bond_batch_indices,
            global_batch_indices = global_batch_indices,
            batched_rxn_graphs = batched_rxn_graphs, 
            batched_atom_reactant = batched_atom_reactant, 
            batched_atom_product = batched_atom_product, 
            batched_bond_reactant = batched_bond_reactant, 
            batched_bond_product = batched_bond_product,

            mappings = None, #!needed for atom and bond.
            has_bonds = None, #!needed for atom and bond.
            ntypes=("global", "atom", "bond"),
            ft_name="ft",
            zero_fts=False,
            empty_graph_fts=True,
        )

        breakpoint()

        # readout layer
        feats = self.readout_layer(batched_rxn_graphs, feats)
        #feats = self.readout_layer(graph, feats)

        for layer in self.fc_layers:
            feats = layer(feats)

        return feats
    
    #used in write_reaction_features.py
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

        #!batch molecule graph.
        nodes = ["atom", "bond", "global"]
        feats = {nt: batched_graph.nodes[nt].data["ft"] for nt in nodes}
        target = label["value"].view(-1)
        target_aug = label["value_rev"].view(-1)
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        stdev = label["scaler_stdev"]
        mean = label["scaler_mean"]
        reactions = label["reaction"]

        breakpoint()

        #!batch_indices of molecular graph
        atom_batch_indices = get_node_batch_indices(batched_graph, "atom")
        bond_batch_indices = get_node_batch_indices(batched_graph, "bond")
        global_batch_indices = get_node_batch_indices(batched_graph, "global")

        #!batch reaction and features
        device="cuda:0"
        (batched_rxn_graphs, 
        batched_atom_reactant, 
        batched_atom_product, 
        batched_bond_reactant, 
        batched_bond_product)=create_batched_reaction_data(reactions, 
                               atom_batch_indices,
                               bond_batch_indices,
                               global_batch_indices, 
                               device)
        #breakpoint()

        if self.stdev is None:
            self.stdev = stdev[0]

        pred = self(
            graph=batched_graph,
            feats=feats,
            reactions=reactions,
            reverse=False,
            norm_bond=norm_bond,
            norm_atom=norm_atom,
            atom_batch_indices=atom_batch_indices,
            bond_batch_indices=bond_batch_indices,
            global_batch_indices=global_batch_indices,
            batched_rxn_graphs = batched_rxn_graphs, 
            batched_atom_reactant =batched_atom_reactant, 
            batched_atom_product =batched_atom_product, 
            batched_bond_reactant =batched_bond_reactant, 
            batched_bond_product = batched_bond_product,
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
                atom_batch_indices=atom_batch_indices,
                bond_batch_indices=bond_batch_indices,
                global_batch_indices=global_batch_indices,
                batched_rxn_graphs = batched_rxn_graphs, 
                batched_atom_reactant =batched_atom_reactant, 
                batched_atom_product =batched_atom_product, 
                batched_bond_reactant =batched_bond_reactant, 
                batched_bond_product = batched_bond_product,
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
        feats = {nt: batched_graph.nodes[nt].data["ft"] for nt in nodes}
        target = label["value"].view(-1)
        target_aug = label["value_rev"].view(-1)
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        reactions = label["reaction"]
        stdev = label["scaler_stdev"]
        mean = label["scaler_mean"]

        atom_batch_indices = get_node_batch_indices(batched_graph, "atom")
        bond_batch_indices = get_node_batch_indices(batched_graph, "bond")
        global_batch_indices = get_node_batch_indices(batched_graph, "global")

        if self.stdev is None:
            self.stdev = stdev[0]

        pred = self(
            graph=batched_graph,
            feats=feats,
            reactions=reactions,
            reverse=False,
            norm_bond=norm_bond,
            norm_atom=norm_atom,
            atom_batch_indices=atom_batch_indices,
            bond_batch_indices=bond_batch_indices,
            global_batch_indices=global_batch_indices,
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

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        """
        Training epoch end
        """
        r2, torch_l1, torch_mse = self.compute_metrics(mode="train")
        # self.log("train_l1", l1, prog_bar=True, sync_dist=True)
        self.log("train_r2", r2, prog_bar=True, sync_dist=True)
        self.log("train_l1", torch_l1, prog_bar=True, sync_dist=True)
        self.log("train_mse", torch_mse, prog_bar=True, sync_dist=True)

        duration = time.time() - self.start_time
        self.log('epoch_duration', duration, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        """
        Validation epoch end
        """
        r2, torch_l1, torch_mse = self.compute_metrics(mode="val")
        # self.log("val_l1", l1, prog_bar=True, sync_dist=True)
        self.log("val_r2", r2, prog_bar=True, sync_dist=True)
        self.log("val_l1", torch_l1, prog_bar=True, sync_dist=True)
        self.log("val_mse", torch_mse, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
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
    

def get_node_batch_indices(batched_graph, node_type):
    """
    Generate batch indices for each node of the specified type in a batched DGL graph.
    
    Args:
    - batched_graph (DGLGraph): The batched graph.
    - node_type (str): The type of nodes for which to generate batch indices.
    
    Returns:
    - torch.Tensor: The batch indices for each node of the specified type.
    """
    batch_num_nodes = batched_graph.batch_num_nodes(node_type)
    return torch.repeat_interleave(torch.arange(len(batch_num_nodes), device=batched_graph.device), batch_num_nodes)

def get_batch_indices_mapping(batch_indices, reactant_ids, atom_bond_map, atom_bond_num, device):
    distinguishable_value = torch.iinfo(torch.long).max
    indices_full = torch.full((atom_bond_num,), distinguishable_value, dtype=torch.long, device=device)
    sorted_index_reaction = [torch.tensor([value for key, value in sorted(d.items())], device=device) for d in atom_bond_map]
    reactant_ids = torch.tensor(reactant_ids, device=device)
    matches = ((batch_indices[:, None] == reactant_ids[None, :]).any(dim=1).nonzero(as_tuple=False).squeeze())
    sorted_values_concat = torch.cat(sorted_index_reaction)
    #batch_indices_reaction = matches[sorted_values_concat]
    indices_full[sorted_values_concat] = matches

    return indices_full

def create_batched_reaction_data(reactions,atom_batch_indices,
                                 bond_batch_indices, global_batch_indices, device):
    
    batched_graphs = dgl.batch([reaction['reaction_graph'] for reaction in reactions])


    batched_atom_reactant = []
    batched_atom_product = []
    batched_bond_reactant = []
    batched_bond_product = []

    idx = 0
    for reaction in reactions:
            print(">>>>>>>>>>>>>>>>id:", idx)
            idx+=1

            num_atoms_total = reaction["mappings"]["num_atoms_total"]
            num_bond_total = reaction["mappings"]["num_bonds_total"]
        #!reactant
        #batched_indices_reaction for reactant.
            reactant_ids= reaction["reaction_molecule_info"]["reactants"]["reactants"]
            #!atom
            atom_map_react = reaction["mappings"]["atom_map"][0]
            batch_indices_react=get_batch_indices_mapping(atom_batch_indices, reactant_ids, atom_map_react, num_atoms_total, device=device)
            batched_atom_reactant.extend(batch_indices_react)

            #!bond
            #breakpoint()
            bond_map_react = reaction["mappings"]["bond_map"][0]
            batch_indices_react=get_batch_indices_mapping(bond_batch_indices, reactant_ids, bond_map_react, num_bond_total, device=device)
            batched_bond_reactant.extend(batch_indices_react)

        #!product
        #batched_indices_reaction for product.
            product_ids= reaction["reaction_molecule_info"]["products"]["products"]
            #!atom
            atom_map_product = reaction["mappings"]["atom_map"][1]
            batch_indices_product=get_batch_indices_mapping(atom_batch_indices, product_ids, atom_map_product, num_atoms_total, device=device)
            batched_atom_product.extend(batch_indices_product)
            #!bond
            bond_map_product = reaction["mappings"]["bond_map"][1]
            batch_indices_product=get_batch_indices_mapping(bond_batch_indices, product_ids, bond_map_product, num_bond_total, device=device)
            batched_bond_product.extend(batch_indices_product)
    
    #!batched indices will be used after MP step.
    return batched_graphs, batched_atom_reactant, batched_atom_product, batched_bond_reactant, batched_bond_product


        # #batched_indices_reaction for reactant.
        #     bond_map_react = reaction["mappings"]["bond_map"][0]
        #     reactant_ids= reaction["reaction_molecule_info"]["reactants"]["reactants"]
        #     batch_indices_reaction=get_batch_indices_mapping(atom_batch_indices, reactant_ids, atom_map_react, device=device)
        #     batched_atom_reactant.extend(batch_indices_reaction)
        # #batched_indices_reaction for product.
        #     atom_map_product = reaction["mappings"]["atom_map"][1]
        #     product_ids= reaction["reaction_molecule_info"]["reactants"]["reactants"]
        #     batch_indices_product=get_batch_indices_mapping(atom_batch_indices, product_ids, atom_map_product, device=device)
        #     batched_atom_product.extend(batch_indices_product)



    # # Initialize containers for the combined mappings and features
    # combined_atom_map = []
    # combined_bond_map = []
    # combined_total_bonds = []
    # combined_total_atoms = []

    # # Iterate through reactions to adjust and aggregate mappings
    # offset_atoms = 0  # Keeps track of the offset for atom indices
    # offset_bonds = 0  # Keeps track of the offset for bond indices
    # for reaction in reactions:
    #     atom_map = reaction['atom_map']
    #     bond_map = reaction['bond_map']
    #     total_atoms = reaction['total_atoms']
    #     total_bonds = reaction['total_bonds']
        
    #     # Adjust mappings with the current offset
    #     adjusted_atom_map = [{k+offset_atoms: v+offset_atoms for k, v in map_pair.items()} for map_pair in atom_map]
    #     adjusted_bond_map = [{k+offset_bonds: v+offset_bonds for k, v in map_pair.items()} for map_pair in bond_map]

    #     # Update the combined mappings
    #     combined_atom_map.extend(adjusted_atom_map)
    #     combined_bond_map.extend(adjusted_bond_map)

    #     # Adjust total_atoms and total_bonds with the current offset
    #     adjusted_total_atoms = [i + offset_atoms for i in total_atoms]
    #     adjusted_total_bonds = [i + offset_bonds for i in total_bonds]

    #     combined_total_atoms.extend(adjusted_total_atoms)
    #     combined_total_bonds.extend(adjusted_total_bonds)

    #     # Update offsets for the next iteration
    #     offset_atoms += max(total_atoms) + 1  # Assuming total_atoms is a list of atom indices
    #     offset_bonds += max(total_bonds) + 1  # Assuming total_bonds is a list of bond indices

    # # Construct batched features based on adjusted mappings. This step will depend on how your features are structured
    # # and may involve aggregating features from individual reactions into a combined tensor that aligns with the batched_graph.
    # # Placeholder for feature construction
    # batched_features = None  # Construct batched features based on the combined mappings and individual reaction features

    # return batched_graphs, combined_atom_map, combined_bond_map, combined_total_atoms, combined_total_bonds, batched_features