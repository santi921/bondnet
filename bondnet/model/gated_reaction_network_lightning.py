import torch
import pytorch_lightning as pl

import itertools, copy, dgl
import numpy as np
import logging
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import lr_scheduler
import torchmetrics
from bondnet.layer.gatedconv import GatedGCNConv, GatedGCNConv1, GatedGCNConv2
from bondnet.layer.readout import Set2SetThenCat
from bondnet.layer.utils import UnifySize
from bondnet.model.metric import (
    WeightedL1Loss, 
    WeightedMSELoss, 
    WeightedSmoothL1Loss,   
    Metrics_WeightedMAE, 
    Metrics_WeightedMSE
)
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt

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
        device="cpu",
        wandb=True,
        augment=False,
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
            "device": device,
            "wandb": wandb,
            "augment": augment
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
        
        self.loss = self.loss_function()
        
        self.train_l1 = Metrics_WeightedMAE()
        #self.train_mse = Metrics_WeightedMSE()
        self.train_r2 = torchmetrics.R2Score(num_outputs=1, multioutput = "variance_weighted")

        self.val_l1 = Metrics_WeightedMAE()
        #self.val_mse = Metrics_WeightedMSE()
        self.val_r2 = torchmetrics.R2Score(num_outputs=1, multioutput = "variance_weighted")

        self.test_l1 = Metrics_WeightedMAE()
        #self.test_mse = Metrics_WeightedMSE()
        self.test_r2 = torchmetrics.R2Score(num_outputs=1, multioutput = "variance_weighted")
        

    def forward(self, graph, feats, reactions, norm_atom=None, norm_bond=None, device=None, reverse = False):
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
        # embedding
        feats = self.embedding(feats)
        # gated layer
        
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)
        
        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        # graph is actually batch graphs, not just a graph
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions, device, reverse)
        
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

        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)
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
        if self.device is not None:
            feats = {k: v.to(self.device) for k, v in feats.items()}
            target = target.to(self.device)
            norm_atom = norm_atom.to(self.device)
            norm_bond = norm_bond.to(self.device)   
            stdev = stdev.to(self.device)
            if(self.hparams.augment and not empty_aug): 
                target_aug = target_aug.to(self.device)

        pred = self(
            batched_graph, 
            feats, 
            label["reaction"], 
            device=self.device, 
            norm_bond = norm_bond, 
            norm_atom=norm_atom)
        
        pred = pred.view(-1)

        if(self.hparams.augment and not empty_aug):
            #target_aug_new_shape = (len(target_aug), 1)
            #target_aug = target_aug.view(target_aug_new_shape) 
            pred_aug = self(
                batched_graph, 
                feats,
                label["reaction"],
                device=self.device, 
                reverse=True, 
                norm_bond = norm_bond, 
                norm_atom=norm_atom)
            pred_aug = pred_aug.view(-1)

            all_loss = self.compute_loss(
                torch.cat((pred, pred_aug), axis=0),
                torch.cat((target, target_aug), axis=0),
                torch.cat((stdev, stdev), axis=0)
            )
            
        else:
            # ========== compute losses ==========
            all_loss = self.compute_loss(pred, target, stdev)
            # ========== logger the loss ==========

        self.log(f"{mode}_loss", all_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(label))
        self.update_metrics(target, pred, stdev, mode)

        return all_loss

    
    def loss_function(self): 
        """
        Initialize loss function
        """

        if(self.hparams.loss_fn == 'mse'):
            loss_fn = MSELoss(reduction="mean")
        elif(self.hparams.loss_fn == 'huber'):
            loss_fn = WeightedSmoothL1Loss(reduction='mean')
        elif(self.hparams.loss_fn == 'mae'):
            loss_fn = WeightedL1Loss(reduction='mean')
        else: 
            loss_fn = WeightedMSELoss(reduction="mean")
    
        return loss_fn 
    

    def compute_loss(self, target, pred, weight): 
        """
        Compute loss
        """
        return self.loss(target, pred, weight)


    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = self._config_lr_scheduler(optimizer)
        
        lr_scheduler = {
            "scheduler": scheduler, 
            "monitor": "val_loss"
            }
        
        return [optimizer], [lr_scheduler]


    def _config_lr_scheduler(self, optimizer):

        scheduler_name = self.hparams["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif scheduler_name == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.lr_scheduler["lr_warmup_step"],
                max_epochs=self.hparams.lr_scheduler["epochs"],
                eta_min=self.hparams.lr_scheduler["lr_min"],
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
        if self.device is not None:
            feats = {k: v.to(self.device) for k, v in feats.items()}
            target = target.to(self.device)
            norm_atom = norm_atom.to(self.device)
            norm_bond = norm_bond.to(self.device)   
            stdev = stdev.to(self.device)
            #if(self.augment and not empty_aug): 
            #    target_aug = target_aug.to(self.device)

        pred = self(
            batched_graph, 
            feats, 
            label["reaction"], 
            device=self.device, 
            norm_bond = norm_bond, 
            norm_atom=norm_atom)
        
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
        #plt.ylim(min_val,max_val)
        #plt.xlim(min_val,max_val)
        plt.title("Predicted vs. True")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("./{}.png".format("./test"))

        return self.shared_step(batch, mode="test")


    def training_epoch_end(self, outputs):
        """
        Training epoch end
        """
        l1, r2 =  self.compute_metrics(mode = "train")
        self.log("train_l1", l1, prog_bar=True)
        self.log("train_r2", r2, prog_bar=True)


    def validation_epoch_end(self, outputs):
        """
        Validation epoch end
        """
        l1, r2 =  self.compute_metrics(mode = "val")
        self.log("val_l1", l1, prog_bar=True)
        self.log("val_r2", r2, prog_bar=True)


    def test_epoch_end(self, outputs):
        """
        Test epoch end
        """
        l1, r2 =  self.compute_metrics(mode = "test")
        self.log("test_l1", l1, prog_bar=True)
        self.log("test_r2", r2, prog_bar=True)


    def update_metrics(self, pred, target, weight, mode):

        if mode == 'train': 
            self.train_l1.update(pred, target, weight, reduction='sum')
            self.train_r2.update(pred, target)
        
        elif mode == 'val':
            self.val_l1.update(pred, target, weight, reduction='sum')
            self.val_r2.update(pred, target) 
        
        elif mode == 'test':            
            self.test_l1.update(pred, target, weight, reduction='sum')
            self.test_r2.update(pred, target) 


    def compute_metrics(self, mode):
        if mode == 'train': 
            l1 = self.train_l1.compute()
            r2 = self.train_r2.compute()
            self.train_r2.reset()
            self.train_l1.reset()

        elif mode == 'val':
            l1 = self.val_l1.compute()
            r2 = self.val_r2.compute()
            self.val_r2.reset()
            self.val_l1.reset()

        elif mode == 'test':
            l1 = self.test_l1.compute()
            r2 = self.test_r2.compute()
            self.test_r2.reset()
            self.test_l1.reset()
        
        return l1, r2


def _split_batched_output(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.

    Returns:
        list of tensor.

    """
    nbonds = graph.batch_num_nodes("bond")
    return torch.split(value, nbonds)


def mol_graph_to_rxn_graph(graph, feats, reactions, device=None, reverse = False):
    """
    Convert a batched molecule graph to a batched reaction graph.

    Essentially, a reaction graph has the same graph structure as the reactant and
    its features are the difference between the products features and reactant features.

    Args:
        graph (BatchedDGLHeteroGraph): batched graph representing molecules.
        feats (dict): node features with node type as key and the corresponding
            features as value.
        reactions (list): a sequence of :class:`bondnet.data.reaction_network.Reaction`,
            each representing a reaction.

    Returns:
        batched_graph (BatchedDGLHeteroGraph): a batched graph representing a set of
            reactions.
        feats (dict): features for the batched graph
    """
    #updates node features on batches
    for nt, ft in feats.items():
        graph.nodes[nt].data.update({"ft": ft}) 
        #feats = {k: v.to(device) for k, v in feats.items()}
    # unbatch molecule graph
    graphs = dgl.unbatch(graph)
    reaction_graphs, reaction_feats = [], []
    batched_feats = {}

    for rxn in reactions:
        reactants = [graphs[i] for i in rxn.reactants]
        products = [graphs[i] for i in rxn.products]
        # might need to rewrite to create dynamics here
        mappings = {
                    "bond_map": rxn.bond_mapping, 
                    "atom_map": rxn.atom_mapping, 
                    "total_bonds": rxn.total_bonds,
                    "total_atoms": rxn.total_atoms,
                    "num_bonds_total":rxn.num_bonds_total,
                    "num_atoms_total":rxn.num_atoms_total
                    }
        has_bonds = {
            "reactants": [True if len(mp) > 0 else False 
                    for mp in rxn.bond_mapping[0]],
            "products": [True if len(mp) > 0 else False 
                    for mp in rxn.bond_mapping[1]],
        }
        if(len(has_bonds["reactants"])!=len(reactants) or 
            len(has_bonds["products"])!=len(products)): 
            print("unequal mapping & graph len")
        
        
        g, fts = create_rxn_graph(
            reactants, products, mappings, has_bonds, device, reverse = reverse
        )
        reaction_graphs.append(g)
        #if(device !=None):
        #    fts = {k: v.to(device) for k, v in fts.items()}
        reaction_feats.append(fts)
        ##################################################
    for i, g, ind in zip(reaction_feats, reaction_graphs, range(len(reaction_feats))): 
        feat_len_dict = {}
        total_feats = 0 
        
        for nt in i.keys():
            total_feats+=i[nt].shape[0]
            feat_len_dict[nt] = i[nt].shape[0]

        if(total_feats!=g.number_of_nodes()): # error checking
            print(g)
            print(feat_len_dict)
            print(reactions[ind].atom_mapping)
            print(reactions[ind].bond_mapping)
            print(reactions[ind].total_bonds)
            print(reactions[ind].total_atoms)
            print("--"*20)

    batched_graph = dgl.batch(reaction_graphs) # batch graphs
    for nt in feats: # batch features
        batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])
    return batched_graph, batched_feats


def construct_rxn_graph_empty(mappings, device=None, self_loop = True):

    bonds_compile = mappings["total_bonds"]
    atoms_compile = mappings["total_atoms"]
    num_bonds = len(bonds_compile)
    num_atoms = len(atoms_compile)
    a2b, b2a = [], []
    
    if num_bonds == 0:
        num_bonds = 1
        a2b, b2a = [(0, 0)], [(0, 0)]

    else:
        a2b, b2a = [], []
        for b in range(num_bonds):
            u = bonds_compile[b][0]
            v = bonds_compile[b][1]
            b2a.extend([[b, u], [b, v]])
            a2b.extend([[u, b], [v, b]])

    a2g = [(a, 0) for a in range(num_atoms)]
    g2a = [(0, a) for a in range(num_atoms)]
    b2g = [(b, 0) for b in range(num_bonds)]
    g2b = [(0, b) for b in range(num_bonds)]

    edges_dict = {
        ("atom", "a2b", "bond"): a2b,
        ("bond", "b2a", "atom"): b2a,
        ("atom", "a2g", "global"): a2g,
        ("global", "g2a", "atom"): g2a,
        ("bond", "b2g", "global"): b2g,
        ("global", "g2b", "bond"): g2b,
    }

    if self_loop:
        a2a = [(i, i) for i in atoms_compile]
        b2b = [(i, i) for i in range(num_bonds)]
        g2g = [(0, 0)]
        edges_dict.update(
            {
                ("atom", "a2a", "atom"): a2a,
                ("bond", "b2b", "bond"): b2b,
                ("global", "g2g", "global"): g2g,
            }
        )
        
    rxn_graph_zeroed = dgl.heterograph(edges_dict)
    if(device is not None): rxn_graph_zeroed = rxn_graph_zeroed.to(device)
    return rxn_graph_zeroed


def create_rxn_graph(
    reactants,
    products,
    mappings,
    has_bonds,
    device=None, 
    ntypes=("global", "atom", "bond"),
    ft_name="ft",
    reverse = False
):
    """
    A reaction is represented by:

    feats of products - feats of reactant

    Args:
        reactants (list of DGLHeteroGraph): a sequence of reactants graphs
        products (list of DGLHeteroGraph): a sequence of product graphs
        mappings (dict): with node type as the key (e.g. `atom` and `bond`) and a list
            as value, which is a mapping between reactant feature and product feature
            of the same atom (bond).
        has_bonds (dict): whether the reactants and products have bonds.
        ntypes (list): node types of which the feature are manipulated
        ft_name (str): key of feature inf data dict

    Returns:
        graph (DGLHeteroGraph): a reaction graph with feats constructed from between
            reactant and products.
        feats (dict): features of reaction graph
    """
    verbose = False
    reactants_ft, products_ft = [], []
    feats = dict()
    
    # note, this assumes we have one reactant
    num_products = int(len(products))
    num_reactants = int(len(reactants))
    graph = construct_rxn_graph_empty(mappings, device)
    if(verbose):
        print("# reactions: {}, # products: {}".format(int(len(reactants)), int(len(products))))

    for nt in ntypes:
        
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]
        
        if(device is not None):
            reactants_ft = [r.to(device) for r in reactants_ft]
            products_ft = [p.to(device) for p in products_ft]

        if(nt=='bond'):
            if(num_products>1):
                products_ft = list(itertools.compress(products_ft, has_bonds["products"]))
                filter_maps = list(itertools.compress(mappings[nt+"_map"][1], has_bonds["products"]))
                mappings[nt+"_map"] = [mappings[nt+"_map"][0], filter_maps]
                
            if(num_reactants>1):      
                reactants_ft = list(itertools.compress(reactants_ft, has_bonds["reactants"]))
                filter_maps = list(itertools.compress(mappings[nt+"_map"][0], has_bonds["reactants"]))
                mappings[nt+"_map"] = [filter_maps, mappings[nt+"_map"][1]]

        if(nt=='global'):
            reactants_ft = [torch.sum(reactant_ft, dim=0, keepdim=True) for reactant_ft in reactants_ft]
            products_ft = [torch.sum(product_ft, dim=0, keepdim=True) for product_ft in products_ft]
            if(device is not None):
                reactants_ft = [r.to(device) for r in reactants_ft]
                products_ft = [p.to(device) for p in products_ft]
        
        len_feature_nt = reactants_ft[0].shape[1]
        #if(len_feature_nt!=64): print(mappings)

        if(nt == 'global'):
            net_ft_full = torch.zeros(1, len_feature_nt)
        elif(nt == 'bond'):
            net_ft_full = torch.zeros(mappings['num_bonds_total'],len_feature_nt) 
        else: 
            net_ft_full = torch.zeros(mappings['num_atoms_total'], len_feature_nt) 
        
        if(device is not None):
            net_ft_full = net_ft_full.to(device)
        
        if nt == "global": 
            coef = 1
            if(reverse==True): coef = -1

            for product_ft in products_ft:
                net_ft_full += coef * torch.sum(product_ft, dim=0, keepdim=True)                
            for reactant_ft in reactants_ft:
                net_ft_full -= coef * torch.sum(reactant_ft, dim=0, keepdim=True) 

        else:
            net_ft_full_temp = copy.deepcopy(net_ft_full) 
            coef = 1
            if(reverse==True): coef = -1
            for ind, reactant_ft in enumerate(reactants_ft):
                mappings_raw = mappings[nt+"_map"][0][ind]
                mappings_react = list(mappings_raw.keys())
                mappings_total = [mappings_raw[mapping] for mapping in mappings_react]
                assert np.max(np.array(mappings_total)) < len(net_ft_full_temp), f"invalid index  {mappings}"
                net_ft_full_temp[mappings_total] = reactant_ft[mappings_react]
                net_ft_full -= coef * net_ft_full_temp

            for ind, product_ft in enumerate(products_ft):
                mappings_raw = mappings[nt+"_map"][1][ind]
                mappings_prod = list(mappings_raw.keys())
                mappings_total = [mappings_raw[mapping] for mapping in mappings_prod]
                assert np.max(np.array(mappings_total)) < len(net_ft_full_temp), f"invalid index  {mappings}"
                net_ft_full_temp[mappings_total] = product_ft[mappings_prod]
                net_ft_full += coef * net_ft_full_temp
            
        feats[nt] = net_ft_full
    return graph, feats