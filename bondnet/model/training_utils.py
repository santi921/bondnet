import numpy as np 
import torch, wandb
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score

from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.model.gated_reaction_network_classifier import GatedGCNReactionNetworkClassifier

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
    
    targets, outputs = [], []

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
                norm_atom=norm_atom, 
                norm_bond=norm_bond
            )

            target_filtered = torch.reshape(target_filtered, (int(target_filtered.shape[0]/categories), categories))
            target_filtered = torch.argmax(target_filtered,axis=1)
            accuracy += (torch.argmax(pred, axis = 1) == target_filtered).sum().item()
            
            outputs = torch.cat((torch.argmax(pred, axis = 1)))
            targets = torch.cat((targets, target_filtered))
            count += len(target_filtered)

    f1 = F1Score(num_classes=5, average='samples')
    f1_score = f1(outputs, targets)

    wandb.log({"F1 test": f1_score})
            
    return accuracy / count, f1_score

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
                
            try:
                pred = model(batched_graph, feats, label["reaction"], norm_atom, norm_bond)
            except:
                pred = model(batched_graph, feats, label["reaction"])

            pred = pred.view(-1)
            target = target.view(-1)

            try:
                mae += metric_fn(pred, target, weight=None).detach().item()
            except: 
                mae += metric_fn(pred, target).detach().item()     
            count += len(target)

    return mae / count

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
        weight = torch.tensor([5., 1., 2. , 1.5, 1.2])

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
            norm_atom=norm_atom, 
            norm_bond=norm_bond
        )

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
    #f1 = F1Score(num_classes=5)
    #f1_score = f1(outputs, targets, average='samples')
    #wandb.log({"F1 test": f1_score})

    return epoch_loss, accuracy

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

    loss_fn = WeightedMSELoss(reduction="sum")
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

        try:
            pred = model(batched_graph, feats, label["reaction"], norm_atom, norm_bond)
        except:
            pred = model(batched_graph, feats, label["reaction"])

        # pred = pred.view(-1)
        target_new_shape = (len(target), 1)
        target = target.view(target_new_shape)
        pred_new_shape = (len(pred), 1)
        pred = pred.view(pred_new_shape)


        try:
            loss = loss_fn(pred, target, stdev)
        except:
            loss = loss_fn(pred, target)

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

    return model, optimizer
