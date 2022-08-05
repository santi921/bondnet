import torch, time, wandb
import numpy as np 
from tqdm import tqdm

from torchmetrics import R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bondnet.model.metric import EarlyStopping

from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import (
    AtomFeaturizerGraph,
    BondAsNodeGraphFeaturizer,
    GlobalFeaturizerGraph,
)
from bondnet.data.grapher import (
    HeteroCompleteGraphFromDGLAndPandas,
)
from bondnet.data.dataset import train_validation_test_split
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
#from bondnet.scripts.create_label_file import read_input_files
from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss
from bondnet.utils import seed_torch,pickle_dump,parse_settings

    
seed_torch()

def evaluate_r2(model, nodes, data_loader):

    model.eval()
    
    with torch.no_grad():
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            
            pred = model(batched_graph, feats, label["reaction"])
            pred = pred.view(-1)
            target = target.view(-1)

    r2_call = R2Score()
    r2 = r2_call(pred, target)
    return r2


def evaluate(model, nodes, data_loader, metric_fn, device=None):
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            stdev = label["scaler_stdev"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            if device is not None:
                #feats = {k: v.to(device) for k, v in feats.items()}
                feats = {nt: batched_graph.nodes[nt].data["feat"].to(device)  for nt in nodes}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)

            pred = model(batched_graph, feats, label["reaction"])
            pred = pred.view(-1)
            target = target.view(-1)

            #accuracy += metric_fn(pred, target, stdev).detach().item()
            accuracy += metric_fn(pred, target, weight=None).detach().item()
            
            count += len(target)

    return accuracy / count


def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn, device=None):

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (batched_graph, label) in enumerate(data_loader):
        target = label["value"]
        stdev = label["scaler_stdev"]
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]

        if device is not None:
            #feats = {k: v.to(device) for k, v in feats.items()}
            feats = {nt: batched_graph.nodes[nt].data["feat"].to(device)  for nt in nodes}
            target = target.to(device)
            norm_atom = norm_atom.to(device)
            norm_bond = norm_bond.to(device)
            stdev = stdev.to(device)


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


def get_grapher():

    atom_featurizer = AtomFeaturizerGraph()
    bond_featurizer = BondAsNodeGraphFeaturizer()
    global_featurizer = GlobalFeaturizerGraph(allowed_charges=[-2, -1, 0, 1])
    grapher = HeteroCompleteGraphFromDGLAndPandas(
        atom_featurizer, bond_featurizer, global_featurizer
    )
    return grapher


def load_model(dict_train): 
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
    optimizer = torch.optim.Adam(model.parameters(), lr=dict_train['learning_rate'])
    loss_func = WeightedMSELoss(reduction="sum")
    metric = WeightedL1Loss(reduction="mean")
    return model, optimizer, loss_func, metric


def main():
    wandb.init(project="my-test-project")
    
    path_mg_data = "../dataset/mg_dataset/" # still not working
    dict_train = parse_settings()

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), file=path_mg_data, out_file="./", target = 'ts'
    )
    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )
    train_loader = DataLoaderReactionNetwork(trainset, batch_size=dict_train['batch_size'], 
    shuffle=True)
    val_loader = DataLoaderReactionNetwork(
        valset, batch_size=len(valset), shuffle=False
    )
    test_loader = DataLoaderReactionNetwork(
        testset, batch_size=len(testset), shuffle=False
    )

    feature_names = ["atom", "bond", "global"]

    dict_train['in_feats'] = dataset.feature_size
    #wandb.config = dict_train
    wandb.config.update(dict_train)
    num_epochs = dict_train['epochs']
    model, optimizer, loss_func, metric = load_model(dict_train)


    if dict_train["on_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_train["gpu"] = device
    else:
        dict_train["gpu"] = "cpu"

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=25, verbose=True)
    stopper = EarlyStopping(patience=150)
    stopper_transfer = EarlyStopping(patience=100)


    if(dict_train['transfer']):
        dataset_transfer = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), file=path_mg_data, out_file="./", target = 'diff'
        )
        trainset_transfer, valset_tranfer, _ = train_validation_test_split(
        dataset_transfer, validation=0.15, test=0.01
        )
        dataset_transfer_loader = DataLoaderReactionNetwork(
            trainset_transfer, batch_size=dict_train['batch_size'], 
        shuffle=True)
        dataset_transfer_loader_val = DataLoaderReactionNetwork(
            valset_tranfer, batch_size=dict_train['batch_size'], 
        shuffle=True)


        print("Initiating Training w/ transfer...")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Trainable Model Params: {}".format(params))
        
        for epoch in tqdm(range(dict_train['transfer_epochs'])):
            loss_transfer, train_acc_transfer = train(
                optimizer, 
                model, 
                feature_names, 
                dataset_transfer_loader, 
                loss_func, 
                metric,
                dict_train["gpu"]
            )
            val_acc_transfer = evaluate(
                model, 
                feature_names, 
                dataset_transfer_loader_val, 
                metric, 
                dict_train["gpu"])

            if stopper_transfer.step(val_acc_transfer):
                break

        # freeze model layers but fc
        model.gated_layers.requires_grad_(False)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Freezing Gated Layers....")
        print("Number of Trainable Model Params: {}".format(params))




    t1 = time.time()
    # optimizer, loss function and metric function
    best = 1e10
    # main training loop
    print("# Epoch     Loss         TrainAcc        ValAcc        ValR2")
    for epoch in range(num_epochs):
        # train on training set
        loss, train_acc = train(
            optimizer, 
            model, 
            feature_names, 
            train_loader, 
            loss_func, 
            metric,
            dict_train["gpu"]
        )
        # evaluate on validation set
        val_acc = evaluate(model, feature_names, val_loader, metric, dict_train["gpu"])
        val_r2 = evaluate_r2(model, feature_names, val_loader)
        wandb.log({"loss": loss})
        wandb.log({"acc_val": val_acc})
        wandb.log({"r2_val": val_r2})

        # save checkpoint for best performing model
        is_best = val_acc < best
        if is_best:
            best = val_acc
            torch.save(model.state_dict(), "checkpoint.pkl")

        print(
            "{:5d}   {:12.6e}   {:12.2e}   {:12.6e}   {:.2f}".format(
                epoch, loss, train_acc, val_acc, val_r2
            )
        )

        if(dict_train["early_stop"]):
            if stopper.step(val_acc):
                pickle_dump(
                    best, dict_train["save_hyper_params"]
                )  # save results for hyperparam tune
                break
        scheduler.step(val_acc)

    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint)
    test_acc = evaluate(model, feature_names, test_loader, metric)
    wandb.log({"loss_test": test_acc})
    print("TestAcc: {:12.6e}".format(test_acc))
    t2 = time.time()

    print("Time to Training: {:5.1f} seconds".format(float(t2 - t1)))
main()

"""dict_train = {
    "learning_rate": 0.0001,
    "batch_size": 256,
    "loss": "weighted_mse",
    "embedding_size":24,
    "gated_num_layers":3,
    "gated_hidden_size":[64, 64, 64],
    "gated_activation":"ReLU",
    "fc_num_layers":2,
    "fc_hidden_size":[128, 64],
    "fc_activation":"ReLU",
    "transfer": False,
    "transfer_epochs": 100,
    "epochs": 100,        
    "scheduler": True,
    "on_gpu": False,
    "early_stop": True
}
"""