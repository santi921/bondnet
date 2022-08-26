import numpy as np 
import torch, wandb
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score
from sklearn.metrics import f1_score

from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.model.gated_reaction_network_classifier import GatedGCNReactionNetworkClassifier
from bondnet.data.featurizer import (
    AtomFeaturizerGraph,
    BondAsNodeGraphFeaturizer,
    BondAsNodeGraphFeaturizerBondLen,
    GlobalFeaturizerGraph,
)
from bondnet.data.grapher import (
    HeteroCompleteGraphFromMolWrapper,
)

def train_classifier(model, nodes, data_loader, optimizer, device = None, categories = 3):
    """
    basic loop for training a classifier. Gets loss and accuracy
        
    Args:
        model(pytorch model): pytorch model
        nodes(dict): node feature dictionary
        data_loader(loader obj): loader object with data to eval
        device(str): cpu/gpu
        categories(int): number of categories
    Returns: 
        accuracy (float): accuracy
        loss (float): cross entropy loss
    """
    model.train()

    targets, outputs = [], []
    epoch_loss, accuracy, count = 0.0, 0.0, 0.0

    for it, (batched_graph, label) in enumerate(data_loader):

        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        stdev = label["scaler_stdev"]
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        weight = torch.tensor([2., 1., 1.2 , 1.1, 1.1])

        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target = target.to(device)
            norm_atom = norm_atom.to(device)
            norm_bond = norm_bond.to(device)
            stdev = stdev.to(device)
            weight = weight.to(device)
        pred, target_filtered, stdev_filtered = model(
            batched_graph, 
            feats, 
            label["reaction"], 
            target,
            stdev,

        )
        #norm_atom=norm_atom, 
        #norm_bond=norm_bond

        target_filtered = torch.reshape(target_filtered, (int(target_filtered.shape[0]/categories), categories))
        target_filtered = torch.argmax(target_filtered, axis=1)
        outputs.append(torch.argmax(pred, axis = 1))
        targets.append(target_filtered)
        loss_fn = CrossEntropyLoss(weight = weight)
        loss = loss_fn(pred, torch.flatten(target_filtered))
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() # here is the actual optimizer step

        accuracy += (torch.argmax(pred, axis = 1) == target_filtered).sum().item()
        epoch_loss += loss.detach().item()
        count += len(target_filtered)

    epoch_loss /= it + 1
    accuracy /= count
    return epoch_loss, accuracy


def evaluate_classifier(model, nodes, data_loader, device = None, categories = 3):
    """
    basic loop for testing a classifier. Gets F1 and accuracy
        
    Args:
        model(pytorch model): pytorch model
        nodes(dict): node feature dictionary
        data_loader(loader obj): loader object with data to eval
        device(str): cpu/gpu
        categories(int): number of categories
    Returns: 
        accuracy (float): accuracy
        f1 (float): f1 score
    """

    model.eval()
    
    targets, outputs = torch.tensor([]), torch.tensor([])
    if device is not None:
        targets, outputs = torch.tensor([]).to(device), torch.tensor([]).to(device)

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
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

            pred, target_filtered, stdev_filtered = model(
                batched_graph, 
                feats, 
                label["reaction"], 
                target,
                stdev,
                
            )
            #norm_atom=norm_atom, 
            #norm_bond=norm_bond

            target_filtered = torch.reshape(target_filtered, (int(target_filtered.shape[0]/categories), categories))
            target_filtered = torch.argmax(target_filtered,axis=1)
            accuracy += (torch.argmax(pred, axis = 1) == target_filtered).sum().item()
            
            outputs = torch.cat([outputs, torch.argmax(pred, axis = 1)])
            targets = torch.cat([targets, target_filtered])
            count += len(target_filtered)

    try:
        outputs_numpy = outputs.long().detach().cpu().numpy()
        target_numpy = targets.long().detach().cpu().numpy()
    except:
        outputs_numpy = outputs.long().numpy()
        target_numpy = targets.long().copy().numpy()
    
    f1_val = f1_score(outputs_numpy, target_numpy, weight = 'macro')
    wandb.log({"F1 test": f1_val})
            
    return accuracy / count, f1_val


def evaluate(model, nodes, data_loader, device=None):
    """
    basic loop for training a classifier. Gets mae
        
    Args:
        model(pytorch model): pytorch model
        nodes(dict): node feature dictionary
        data_loader(loader obj): loader object with data to eval
        device(str): cpu/gpu
    Returns: 
        mae(float): mae
    """
    metric_fn = WeightedL1Loss(reduction="mean")
    model.eval()

    with torch.no_grad():
        count, mae = 0.0, 0.0
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
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
                
            #try:
            #    pred = model(batched_graph, feats, label["reaction"], norm_atom, norm_bond)
            #except:
            pred = model(batched_graph, feats, label["reaction"])
            mae += metric_fn(pred, target, stdev).detach().item() 
            count += len(target)

    l1_acc = mae / count
    
    return l1_acc

"""
def evaluate(model, nodes, data_loader, device = None):
    model.eval()
    metric_fn = WeightedL1Loss(reduction="mean")

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            stdev = label["scaler_stdev"]

            pred = model(batched_graph, feats, label["reaction"])
            #pred = pred.view(-1)

            accuracy += metric_fn(pred, target, stdev).detach().item()
            #accuracy += metric_fn(pred, target).detach().item() * stdev
            
            count += len(target)

    return accuracy / count
"""

def train(model, nodes, data_loader, optimizer, device=None):
    """
    basic loop for training a classifier. Gets loss and accuracy
        
    Args:
        model(pytorch model): pytorch model
        nodes(dict): node feature dictionary
        data_loader(loader obj): loader object with data to eval
        device(str): cpu/gpu
    Returns: 
        accuracy (float): accuracy
        loss (float): MSE
    """

    loss_fn = WeightedMSELoss(reduction="mean")
    metric_fn = WeightedL1Loss(reduction="mean")

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (batched_graph, label) in enumerate(data_loader):
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
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

        #try:
        #    pred = model(batched_graph, feats, label["reaction"], norm_atom, norm_bond)
        #except:
        pred = model(batched_graph, feats, label["reaction"])

        # pred = pred.view(-1)
        target_new_shape = (len(target), 1)
        target = target.view(target_new_shape)
        pred_new_shape = (len(pred), 1)
        pred = pred.view(pred_new_shape)

        #try:
        #    loss = loss_fn(pred, target, stdev)
        #except:
        loss = loss_fn(pred, target, weight = None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # here is the actual optimizer step

        epoch_loss += loss.detach().item()

        accuracy += metric_fn(pred, target, stdev).detach().item()
        count += len(target)

    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy


def load_model(dict_train): 
    """
    returns model and optimizer from dict of parameters
        
    Args:
        dict_train(dict): dictionary
    Returns: 
        model (pytorch model): model to train
        optimizer (pytorch optimizer obj): optimizer
    """

    if(dict_train["classifier"]):
        model = GatedGCNReactionNetworkClassifier(
        in_feats=dict_train['in_feats'],
        embedding_size=dict_train['embedding_size'],
        gated_num_layers=dict_train['gated_num_layers'],
        gated_hidden_size=dict_train['gated_hidden_size'],
        gated_activation=dict_train['gated_activation'],
        fc_num_layers=dict_train['fc_num_layers'],
        fc_hidden_size=dict_train['fc_hidden_size'],
        fc_activation=dict_train['fc_activation'],
        outdim = dict_train["categories"]
        )

    else: 
        model = GatedGCNReactionNetwork(
        in_feats=dict_train['in_feats'],
        embedding_size=dict_train['embedding_size'],
        gated_num_layers=dict_train['gated_num_layers'],
        gated_hidden_size=dict_train['gated_hidden_size'],
        gated_activation=dict_train['gated_activation'],
        fc_num_layers=dict_train['fc_num_layers'],
        fc_hidden_size=dict_train['fc_hidden_size'],
        fc_activation=dict_train['fc_activation'],
        )
    
    optimizer = Adam(model.parameters(), lr=dict_train['learning_rate'])
    optimizer_transfer = Adam(model.parameters(), lr=dict_train['learning_rate'])

    return model, optimizer, optimizer_transfer


def evaluate_r2(model, nodes, data_loader, device = None):
    model.eval()
    with torch.no_grad():
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
            pred = model(batched_graph, feats, label["reaction"])
            #pred = model(batched_graph, feats, label["reaction"], norm_atom, norm_bond)
            #pred = pred.view(-1)
            #target = target.view(-1)

    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def get_grapher():

    atom_featurizer = AtomFeaturizerGraph()
    bond_featurizer = BondAsNodeGraphFeaturizerBondLen()
    global_featurizer = GlobalFeaturizerGraph(allowed_charges=[-2, -1, 0, 1])
    grapher = HeteroCompleteGraphFromMolWrapper(
        atom_featurizer, bond_featurizer, global_featurizer
    )
    return grapher
