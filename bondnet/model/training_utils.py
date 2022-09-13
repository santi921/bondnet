import numpy as np 
import torch
from copy import deepcopy
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score


from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss, WeightedSmoothL1Loss
from bondnet.model.gated_reaction_network_graph import GatedGCNReactionNetwork
from bondnet.model.gated_reaction_classifier_graph import GatedGCNReactionNetworkClassifier

from bondnet.data.featurizer import (
    AtomFeaturizerGraph,
    BondAsNodeGraphFeaturizerBondLen, # might want to switch
    BondAsNodeGraphFeaturizer,
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
            device = device,
            norm_bond = norm_bond, 
            norm_atom = norm_atom
        )

        #target_filtered = torch.reshape(target_filtered, 
        #    (int(target_filtered.shape[0]/categories), categories))
        target_filtered = torch.argmax(target_filtered, axis=1)
        outputs.append(torch.argmax(pred, axis = 1))
        targets.append(target_filtered)
        loss_fn = CrossEntropyLoss(weight = weight)
        #print(pred)
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
                norm_bond = norm_bond,
                norm_atom = norm_atom, 
                device = device
            )
  

            #target_filtered = torch.reshape(target_filtered, (int(target_filtered.shape[0]/categories), categories))
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
    
    f1_val = f1_score(outputs_numpy, target_numpy, average = 'macro')
    
            
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
            pred = model(
                batched_graph, 
                feats, 
                label["reaction"], 
                device=device, 
                norm_atom = norm_atom, 
                norm_bond = norm_bond)

            mae += metric_fn(pred, target, stdev).detach().item() 
            count += len(target)

    l1_acc = mae / count
    
    return l1_acc


def train(model, nodes, data_loader, optimizer,loss_fn ='mse', device=None, augment=False):
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
    if(loss_fn == 'mse'):
        loss_fn = WeightedMSELoss(reduction="mean")
    elif(loss_fn == 'huber'):
        loss_fn = WeightedSmoothL1Loss(reduction='mean')
    elif(loss_fn == 'mae'):
        loss_fn = WeightedL1Loss(reduction='mean')
    else: 
        loss_fn = WeightedMSELoss(reduction="mean")

    metric_fn = WeightedL1Loss(reduction="mean")
    
    count, accuracy, epoch_loss = 0.0, 0.0, 0.0
    model.train()

    for it, (batched_graph, label) in enumerate(data_loader):
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        
        target_aug = label["value_rev"]        
        empty_aug = torch.isnan(target_aug).tolist()
        empty_aug = True in empty_aug
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]

        if(None in norm_bond.tolist()): print("nan value detected")

        stdev = label["scaler_stdev"]

        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target = target.to(device)
            norm_atom = norm_atom.to(device)
            norm_bond = norm_bond.to(device)   
            stdev = stdev.to(device)
            if(augment and not empty_aug): target_aug = target_aug.to(device)
        
        target_new_shape = (len(target), 1)
        target = target.view(target_new_shape) 
        pred = model(batched_graph, feats, label["reaction"], device=device, norm_bond = norm_bond, norm_atom=norm_atom)
        pred_new_shape = (len(pred), 1)
        pred = pred.view(pred_new_shape)

        if(augment and not empty_aug):
            target_aug_new_shape = (len(target_aug), 1)
            target_aug = target.view(target_aug_new_shape) 
            pred_aug = model(batched_graph, feats, label["reaction"], device=device, reverse=True, norm_bond = norm_bond, norm_atom=norm_atom)
            pred_aug_new_shape = (len(pred_aug), 1)
            pred_aug = pred_aug.view(pred_aug_new_shape)
            loss = loss_fn(torch.concat([pred, pred_aug]), torch.concat([target,target_aug]), stdev)
        
        else:
            loss = loss_fn(pred, target, stdev)


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
            gated_dropout=dict_train["gated_dropout"],
            gated_num_layers=dict_train['gated_num_layers'],
            gated_hidden_size=dict_train['gated_hidden_size'],
            gated_activation=dict_train['gated_activation'],
            gated_batch_norm=dict_train["gated_batch_norm"],
            gated_graph_norm=dict_train["gated_graph_norm"],
            gated_num_fc_layers=dict_train["gated_num_fc_layers"],
            gated_residual=dict_train["gated_residual"],
            num_lstm_iters=dict_train["num_lstm_iters"],
            num_lstm_layers=dict_train["num_lstm_layers"],           
            fc_num_layers=dict_train['fc_num_layers'],
            fc_hidden_size=dict_train['fc_hidden_size'],
            fc_batch_norm=dict_train['fc_batch_norm'],
            fc_activation=dict_train['fc_activation'],
            fc_dropout=dict_train["fc_dropout"],
            outdim = dict_train["categories"]
        )

    else: 
        model = GatedGCNReactionNetwork(
            in_feats=dict_train['in_feats'],
            embedding_size=dict_train['embedding_size'],
            gated_dropout=dict_train["gated_dropout"],
            gated_num_layers=dict_train['gated_num_layers'],
            gated_hidden_size=dict_train['gated_hidden_size'],
            gated_activation=dict_train['gated_activation'],
            gated_batch_norm=dict_train["gated_batch_norm"],
            gated_graph_norm=dict_train["gated_graph_norm"],
            gated_num_fc_layers=dict_train["gated_num_fc_layers"],
            gated_residual=dict_train["gated_residual"],
            num_lstm_iters=dict_train["num_lstm_iters"],
            num_lstm_layers=dict_train["num_lstm_layers"],
            fc_dropout=dict_train["fc_dropout"],
            fc_batch_norm=dict_train['fc_batch_norm'],
            fc_num_layers=dict_train['fc_num_layers'],
            fc_hidden_size=dict_train['fc_hidden_size'],
            fc_activation=dict_train['fc_activation'],
        )
    
    optimizer = Adam(model.parameters(), lr=dict_train['learning_rate'], weight_decay=dict_train['weight_decay'])
    optimizer_transfer = Adam(model.parameters(), lr=dict_train['learning_rate'], weight_decay=dict_train['weight_decay'])

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
            pred = model(batched_graph, feats, label["reaction"], device=device, 
                norm_atom = norm_atom, 
                norm_bond = norm_bond)

    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def get_grapher(bond_len_in_featurizer=True):

    atom_featurizer = AtomFeaturizerGraph()
    if(bond_len_in_featurizer):
        print("using bond featurizer w/xyz coords")
        bond_featurizer = BondAsNodeGraphFeaturizerBondLen()
    else: 
        print("using simple bond featurizer")
        bond_featurizer = BondAsNodeGraphFeaturizer()
        
    global_featurizer = GlobalFeaturizerGraph(allowed_charges=[-2, -1, 0, 1])
    grapher = HeteroCompleteGraphFromMolWrapper(
        atom_featurizer, bond_featurizer, global_featurizer
    )
    return grapher
