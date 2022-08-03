import torch
import time 
import pandas as pd 
from torch.nn import MSELoss
from bondnet.data.dataset import ReactionNetworkDataset, ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import (
    AtomFeaturizerMinimum, BondAsNodeFeaturizerMinimum, 
    GlobalFeaturizer, BondAsNodeFeaturizerFull,
    AtomFeaturizerGraph, BondAsNodeGraphFeaturizer, GlobalFeaturizerGraph,
)
from bondnet.data.grapher import HeteroMoleculeGraph, HeteroCompleteGraphFromPandas, HeteroCompleteGraphFromDGLAndPandas
from bondnet.data.dataset import train_validation_test_split
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.scripts.create_label_file import read_input_files
from bondnet.model.metric import WeightedL1Loss
from bondnet.utils import seed_torch
from torchsummary import summary

def evaluate(model, nodes, data_loader, metric_fn):
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            stdev = label["scaler_stdev"]

            pred = model(batched_graph, feats, label["reaction"])
            pred = pred.view(-1)

            accuracy += metric_fn(pred, target, stdev).detach().item()
            count += len(target)

    return accuracy / count
def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn):

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (batched_graph, label) in enumerate(data_loader):
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        stdev = label["scaler_stdev"]

        pred = model(batched_graph, feats, label["reaction"])
        pred = pred.view(-1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # here is the actual optimizer step

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target, stdev).detach().item()
        count += len(target)
    
    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy
def get_grapher():
    atom_featurizer = AtomFeaturizerMinimum()
    #bond_featurizer = BondAsNodeFeaturizerMinimum()
    #global_featurizer = GlobalFeaturizer(allowed_charges=[-2, -1, 0, 1])
    #grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer)
    
    atom_featurizer = None #AtomFeaturizerGraph()
    bond_featurizer = BondAsNodeGraphFeaturizer()
    global_featurizer = GlobalFeaturizerGraph(allowed_charges=[-2, -1, 0, 1])
    grapher = HeteroCompleteGraphFromDGLAndPandas(
        atom_featurizer, bond_featurizer, global_featurizer)
    return grapher

get_grapher()
seed_torch()


path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
dataset = ReactionNetworkDatasetGraphs(
    grapher=get_grapher(),
    file=path_mg_data,
    out_file = './'

)

trainset, valset, testset = train_validation_test_split(dataset, validation=0.1, test=0.15)
train_loader = DataLoaderReactionNetwork(trainset, batch_size=100,shuffle=True)
test_loader = DataLoaderReactionNetwork(testset, batch_size=100,shuffle=True)
val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)

model = GatedGCNReactionNetwork(
    in_feats=dataset.feature_size,
    embedding_size=24,
    gated_num_layers=2,
    gated_hidden_size=[64, 64, 64],
    gated_activation="ReLU",
    fc_num_layers=2,
    fc_hidden_size=[128, 64],
    fc_activation='ReLU'
)



t1 = time.time()
# optimizer, loss function and metric function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = MSELoss(reduction="mean")
metric = WeightedL1Loss(reduction="sum")

feature_names = ["atom", "bond", "global"]
best = 1e10
num_epochs = 20

# main training loop
print("# Epoch     Loss         TrainAcc        ValAcc")
for epoch in range(num_epochs):

    # train on training set 
    loss, train_acc = train(optimizer, model, feature_names, train_loader, loss_func, metric)
    # evaluate on validation set
    val_acc = evaluate(model, feature_names, test_loader, metric)
    # save checkpoint for best performing model 
    is_best = val_acc < best
    if is_best:
        best = val_acc
        torch.save(model.state_dict(), 'checkpoint.pkl')
    print("{:5d}   {:12.6e}   {:12.6e}   {:12.6e}".format(epoch, loss, train_acc, val_acc))


# load best performing model and test it's performance on the test set
checkpoint = torch.load("checkpoint.pkl")
model.load_state_dict(checkpoint)
test_acc = evaluate(model, feature_names, test_loader, metric)
print("TestAcc: {:12.6e}".format(test_acc))
t2 = time.time()
print("Time to Training: {:5.1f} seconds".format(float(t2 - t1)))