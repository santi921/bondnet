import torch, wandb
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchmetrics import R2Score, F1Score

from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.model.gated_reaction_network_classifier import GatedGCNReactionNetworkClassifier

def evaluate_classifier(model, nodes, data_loader, device = None, categories = 3):
    """
    basic loop for training a classifier 
        
    Args:
        model(pytorch model) 
        nodes() 
        data_loader(loader obj):
        device(str) :
        categories(int):
    Returns: 
        dict_ret(dictionary): dictionary with settings 
    """

    model.eval()

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
                #feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)
                feats = {k: v.to(device) for k, v in feats.items()}

            pred, target_filtered, stdev_filtered = model(batched_graph, feats, label["reaction"], target,  stdev)
            target_filtered = torch.reshape(target_filtered, (int(target_filtered.shape[0]/categories), categories))
            target_filtered = torch.argmax(target_filtered,axis=1)
            accuracy += (torch.argmax(pred, axis = 1) == target_filtered).sum().item()
            
            f1 = F1Score(num_classes=5)
            f1_score = f1(target_filtered, torch.argmax(pred, axis = 1))
            wandb.log({"F1 test": f1_score})

            count += len(target_filtered)

    return accuracy / count, f1_score

def evaluate(model, nodes, data_loader, device=None):
    metric_fn = WeightedL1Loss(reduction="mean")

    model.eval()

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
                #feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)
                feats = {k: v.to(device) for k, v in feats.items()}

            pred = model(batched_graph, feats, label["reaction"])
            pred = pred.view(-1)
            target = target.view(-1)

            #accuracy += metric_fn(pred, target, stdev).detach().item()
            try:
                accuracy += metric_fn(pred, target, weight=None).detach().item()
            except: 
                accuracy += metric_fn(pred, target).detach().item()     
            count += len(target)

    return accuracy / count

def train_classifier(model, nodes, data_loader, optimizer, device = None, categories = 3):

    model.train()
    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (batched_graph, label) in enumerate(data_loader):

        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        stdev = label["scaler_stdev"]
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]

        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target = target.to(device)
            norm_atom = norm_atom.to(device)
            norm_bond = norm_bond.to(device)
            stdev = stdev.to(device)


        pred, target_filtered, stdev_filtered = model(batched_graph, feats, label["reaction"], target,  stdev)
        target_filtered = torch.reshape(target_filtered, (int(target_filtered.shape[0]/categories), categories))
        target_filtered = torch.argmax(target_filtered, axis=1)
        loss_fn = CrossEntropyLoss(weight = torch.tensor([5., 1., 2. , 1.5, 1.2]))
        loss = loss_fn(pred,torch.flatten(target_filtered))
        optimizer.zero_grad()
        loss.backward() 

        optimizer.step() # here is the actual optimizer step

        accuracy += (torch.argmax(pred, axis = 1) == target_filtered).sum().item()
        epoch_loss += loss.detach().item()
        count += len(target_filtered)

    epoch_loss /= it + 1
    accuracy /= count
    return epoch_loss, accuracy

def train(model, nodes, data_loader, optimizer, device=None):

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
            pred = model(batched_graph, feats, label["reaction"])
        except:
            pred = model(batched_graph, feats, label["reaction"], target = target, stdev = stdev)


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
