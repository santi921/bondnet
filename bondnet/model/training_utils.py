import torch
import numpy as np
import pytorch_lightning as pl

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from bondnet.model.metric import WeightedL1Loss

from bondnet.model.gated_reaction_network_lightning_classifier import (
    GatedGCNReactionNetworkLightningClassifier,
)
from bondnet.model.gated_reaction_network_lightning import (
    GatedGCNReactionNetworkLightning,
)
from bondnet.data.grapher import HeteroCompleteGraphFromMolWrapper
from bondnet.data.featurizer import (
    AtomFeaturizerGraphGeneral,
    BondAsNodeGraphFeaturizerGeneral,
    GlobalFeaturizerGraph,
)


class LogParameters(pl.Callback):
    # weight and biases to tensorboard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n, p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:  # WARN: sanity_check is turned on by default
            lp = []
            tensorboard_logger_index = 0
            for n, p in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(
                    n, p.data, trainer.current_epoch
                )
                self.d_parameters[n].append(p.ravel().cpu().numpy())
                lp.append(p.ravel().cpu().numpy())

            p = np.concatenate(lp)
            trainer.logger.experiment.add_histogram(
                "Parameters", p, trainer.current_epoch
            )


def evaluate_breakdown(model, nodes, data_loader, device=None):
    """
    basic loop for evaluating performance across a trained model. Get mae

    Args:
        model(pytorch model): pytorch model
        nodes(dict): node feature dictionary
        data_loader(loader obj): loader object with data to eval
        device(str): cpu/gpu
    Returns:
        mae(float): mae
    """
    metric_fn = WeightedL1Loss(reduction="sum")
    model.eval()

    dict_result_raw = {}

    with torch.no_grad():
        count, mae = 0.0, 0.0
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            # norm_atom = label["norm_atom"]
            norm_atom = None
            # norm_bond = label["norm_bond"]
            norm_bond = None
            stdev = label["scaler_stdev"]
            reaction_types = label["reaction_types"]

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                # norm_atom = norm_atom.to(device)
                # norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)

            pred = model(
                batched_graph,
                feats,
                label["reaction"],
                device=device,
                norm_atom=norm_atom,
                norm_bond=norm_bond,
            )

            try:
                res = metric_fn(pred, target, stdev).detach().item()
            except:
                res = metric_fn(pred, target, stdev)  # .detach().item()

            for ind, i in enumerate(reaction_types):
                for type in i:
                    if type in list(dict_result_raw.keys()):
                        dict_result_raw[type].append(res[ind].detach().item())
                    else:
                        dict_result_raw[type] = [res[ind].detach().item()]

    for k, v in dict_result_raw.items():
        dict_result_raw[k] = np.mean(np.array(v))

    return dict_result_raw


def load_model_lightning(dict_train, load_dir=None):
    """
    returns model and optimizer from dict of parameters

    Args:
        dict_train(dict): dictionary
    Returns:
        model (pytorch model): model to train
        optimizer (pytorch optimizer obj): optimizer
    """
    # print(dict_train)
    if dict_train["restore"]:
        print(":::RESTORING MODEL FROM EXISTING FILE:::")

        if dict_train["restore_path"] != None:
            model = GatedGCNReactionNetworkLightning.load_from_checkpoint(
                checkpoint_path=dict_train["restore_path"]
            )
            # model.to(device)
            print(":::MODEL LOADED:::")
            return model

        if load_dir == None:
            load_dir = "./"

        try:
            model = GatedGCNReactionNetworkLightning.load_from_checkpoint(
                checkpoint_path=load_dir + "/last.ckpt"
            )
            # model.to(device)
            print(":::MODEL LOADED:::")
            return model

        except:
            print(":::NO MODEL FOUND LOADING FRESH MODEL:::")

    shape_fc = dict_train["fc_hidden_size_shape"]
    shape_gat = dict_train["gated_hidden_size_shape"]
    base_fc = dict_train["fc_hidden_size_1"]
    base_gat = dict_train["gated_hidden_size_1"]

    if shape_fc == "flat":
        fc_layers = [base_fc for i in range(dict_train["fc_num_layers"])]
    else:
        fc_layers = [
            int(base_fc / (2**i)) for i in range(dict_train["fc_num_layers"])
        ]

    if shape_gat == "flat":
        gat_layers = [base_gat for i in range(dict_train["gated_num_layers"])]
    else:
        gat_layers = [
            int(base_gat / (2**i)) for i in range(dict_train["gated_num_layers"])
        ]

    if dict_train["classifier"]:
        print("CONSTRUCTING CLASSIFIER MODEL")
        model = GatedGCNReactionNetworkLightningClassifier(
            in_feats=dict_train["in_feats"],
            embedding_size=dict_train["embedding_size"],
            gated_dropout=dict_train["gated_dropout"],
            gated_num_layers=len(gat_layers),
            gated_hidden_size=gat_layers,
            gated_activation=dict_train["gated_activation"],
            gated_batch_norm=dict_train["gated_batch_norm"],
            gated_graph_norm=dict_train["gated_graph_norm"],
            gated_num_fc_layers=dict_train["gated_num_fc_layers"],
            gated_residual=dict_train["gated_residual"],
            num_lstm_iters=dict_train["num_lstm_iters"],
            num_lstm_layers=dict_train["num_lstm_layers"],
            fc_dropout=dict_train["fc_dropout"],
            fc_batch_norm=dict_train["fc_batch_norm"],
            fc_num_layers=len(fc_layers),
            fc_hidden_size=fc_layers,
            fc_activation=dict_train["fc_activation"],
            learning_rate=dict_train["learning_rate"],
            weight_decay=dict_train["weight_decay"],
            scheduler_name="reduce_on_plateau",
            warmup_epochs=10,
            max_epochs=dict_train["max_epochs"],
            eta_min=1e-6,
            loss_fn=dict_train["loss"],
            augment=dict_train["augment"],
            # device=device,
            cat_weights=dict_train["cat_weights"],
            conv=dict_train["conv"],
            reactant_only=dict_train["reactant_only"],
        )

    else:
        model = GatedGCNReactionNetworkLightning(
            in_feats=dict_train["in_feats"],
            embedding_size=dict_train["embedding_size"],
            gated_dropout=dict_train["gated_dropout"],
            gated_num_layers=len(gat_layers),
            gated_hidden_size=gat_layers,
            gated_activation=dict_train["gated_activation"],
            gated_batch_norm=dict_train["gated_batch_norm"],
            gated_graph_norm=dict_train["gated_graph_norm"],
            gated_num_fc_layers=dict_train["gated_num_fc_layers"],
            gated_residual=dict_train["gated_residual"],
            num_lstm_iters=dict_train["num_lstm_iters"],
            num_lstm_layers=dict_train["num_lstm_layers"],
            fc_dropout=dict_train["fc_dropout"],
            fc_batch_norm=dict_train["fc_batch_norm"],
            fc_num_layers=len(fc_layers),
            fc_hidden_size=fc_layers,
            fc_activation=dict_train["fc_activation"],
            learning_rate=dict_train["learning_rate"],
            weight_decay=dict_train["weight_decay"],
            scheduler_name="reduce_on_plateau",
            warmup_epochs=10,
            max_epochs=dict_train["max_epochs"],
            eta_min=1e-6,
            loss_fn=dict_train["loss"],
            augment=dict_train["augment"],
            conv=dict_train["conv"],
            reactant_only=dict_train["reactant_only"],
            # device=device,
        )
    # model.to(device)

    return model


def get_grapher(
    features,
    allowed_charges=None,
    allowed_spin=None,
    global_feats=["charge", "functional_group_reacted"],
):
    """keys_selected_bonds = [
    "Lagrangian_K", "Hamiltonian_K", "e_density", "lap_e_density",
    "e_loc_func", "ave_loc_ion_E", "delta_g_promolecular",
    "delta_g_hirsh", "esp_nuc", "esp_e", "esp_total",
    "grad_norm", "lap_norm", "eig_hess", "det_hessian",
    "ellip_e_dens", "eta"]
    """
    """ keys_selected_atoms = [        
        "Lagrangian_K", "Hamiltonian_K", "e_density", "lap_e_density",
        "e_loc_func", "ave_loc_ion_E", "delta_g_promolecular",
        "delta_g_hirsh", "esp_nuc", "esp_e", "esp_total",
        "grad_norm", "lap_norm", "eig_hess", "det_hessian",
        "ellip_e_dens", "eta"]
        
        #keys_selected_atoms = ["valence_electrons", "total_electrons", 
        #"partial_charges_nbo", "partial_charges_mulliken", "partial_charges_resp",
        #]
        #"partial_spins1" # these might need to be imputed
        #"partial_spins2" # these might need to be imputed
    """
    """
    """
    """

        print("using general atom featurizer w/ electronic info ")

    else:
        print("using general baseline atom featurizer")

    if(bond_len_in_featurizer and electronic_info_in_bonds):
        # evans 
        #keys_selected_bonds = ["1_s", "2_s", "1_p", "2_p", "1_d", "2_d", "1_f", "2_f", "1_polar", "2_polar", "occ_nbo", "bond_length"]
        
        # qtaim
        keys_selected_bonds = [        
            "Lagrangian_K", "Hamiltonian_K", "e_density", "lap_e_density",
            "e_loc_func", "ave_loc_ion_E", "delta_g_promolecular",
            "delta_g_hirsh", "esp_nuc", "esp_e", "esp_total",
            "grad_norm", "lap_norm", "eig_hess", "det_hessian",
            "ellip_e_dens", "eta"]
        print("using general bond featurizer w/xyz + Electronic Info coords")

    elif(electronic_info_in_bonds): 
        keys_selected_bonds = ["1_s", "2_s", "1_p", "2_p", "1_d", "2_d", "1_f", "2_f", "1_polar", "2_polar", "occ_nbo"]
        print("using general bond featurizer w/Electronic Info coords")
        
    elif(bond_len_in_featurizer):
        keys_selected_bonds = ["bond_length"]
        print("using general bond featurizer w/xyz coords")
        
    else: 
        print("using general simple bond featurizer")"""

    # find keys with bonds in the name

    keys_selected_atoms, keys_selected_bonds, keys_selected_global = [], [], []

    for key in features:
        if "bond" in key:
            if key == "bond_length":
                keys_selected_bonds.append(key)
            else:
                keys_selected_bonds.append(key[5:])
        else:
            if "indices" not in key:
                if key not in global_feats:
                    keys_selected_atoms.append(key)
                else:
                    keys_selected_global.append(key)

    # print("keys_selected_atoms", keys_selected_atoms)
    # print("keys_selected_bonds", keys_selected_bonds)

    atom_featurizer = AtomFeaturizerGraphGeneral(selected_keys=keys_selected_atoms)
    bond_featurizer = BondAsNodeGraphFeaturizerGeneral(
        selected_keys=keys_selected_bonds
    )
    fg_list = None
    if len(keys_selected_global) > 0:
        if "functional_group_reacted" in features:
            # hard coded for now. Ideally need to implement get_hydro_data_functional_groups
            # this is tough just because not hard-encoding this make it less portable
            fg_list = [
                "PDK",
                "amide",
                "carbamate",
                "carboxylic acid ester",
                "cyclic carbonate",
                "epoxide",
                "imide",
                "lactam",
                "lactone",
                "nitrile",
                "urea",
            ]

    print("fg_list", fg_list)

    global_featurizer = GlobalFeaturizerGraph(
        allowed_charges=allowed_charges,
        allowed_spin=allowed_spin,
        functional_g_basis=fg_list,
        selected_keys=keys_selected_global,
    )
    grapher = HeteroCompleteGraphFromMolWrapper(
        atom_featurizer, bond_featurizer, global_featurizer
    )
    return grapher
