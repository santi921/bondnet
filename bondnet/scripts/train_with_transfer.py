import argparse
import torch
import time 
import numpy as np 
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from torch.nn import MSELoss
from bondnet.data.dataset import ReactionNetworkDataset
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import AtomFeaturizerMinimum, BondAsNodeFeaturizerMinimum, GlobalFeaturizer, BondAsNodeFeaturizerFull
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.dataset import train_validation_test_split
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.scripts.create_label_file import read_input_files
from bondnet.model.metric import WeightedL1Loss
from bondnet.utils import seed_torch
from torchsummary import summary


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

def get_grapher():
    atom_featurizer = AtomFeaturizerMinimum()
    #bond_featurizer = BondAsNodeFeaturizerMinimum()
    bond_featurizer = BondAsNodeFeaturizerFull()
    # our example dataset contains molecules of charges -1, 0, and 1
    global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer)
    
    return grapher

def parse_settings(file = "./input_files/input_1.txt"):

    #some default values that get written over if in the file
    test = True
    epochs = 10
    embedding_size = 24
    
    fc_hidden_size = [128, 64]
    fc_layers = -1
    fc_activation = "ReLU"
    fc_batch_norm = 0
    fc_dropout = 0.0

    gated_hidden_size = [64, 64, 64]
    gated_layers = -1
    gated_batch_norm = 0
    gated_graph_norm = 0
    gated_dropout = 0.0
    gated_activation = 'ReLU'

    num_lstm_layers = 3
    num_lstm_iters = 5

    
    

    with open(file) as f:
        lines =  f.readlines()
    
        for i in lines: 
            if(len(i.split()) > 1):
                if(i.split()[0] == 'test'):
                    test = bool(i.split()[1])
                if(i.split()[0] == 'epochs'):
                    epochs = int(i.split()[1])
                if(i.split()[0] == 'embedding_size'):
                    embedding_size = int(i.split()[1])

                if(i.split()[0] == 'gated_hidden_size'):
                    gated_hidden_size = [int(j) for j in i.split()[1:]]
                if(i.split()[0] == 'gated_layers'):
                    gated_layers = int(i.split()[1])
                if(i.split()[0] == 'gated_dropout'):
                    gated_dropout = float(i.split()[1])
                if(i.split()[0] == 'gated_graph_norm'):
                    gated_graph_norm = int(i.split()[1])
                if(i.split()[0] == 'gated_batch_norm'):
                    gated_batch_norm = int(i.split()[1])
                if(i.split()[0] == 'gated_activation'):
                    gated_activation = str(i.split()[1])

                if(i.split()[0] == 'fc_hidden_size'):
                    fc_hidden_size = [int(j) for j in i.split()[1:]]
                if(i.split()[0] == 'fc_layers'):
                    fc_layers = int(i.split()[1])
                if(i.split()[0] == 'fc_activation'):
                    fc_activation = str(i.split()[1])
                if(i.split()[0] == 'fc_batch_norm'):
                    fc_batch_norm = int(i.split()[1])
                if(i.split()[0] == 'fc_dropout'):
                    fc_dropout = float(i.split()[1])

                if(i.split()[0] == 'num_lstm_iters'):
                    num_lstm_iters = int(i.split()[1])
                if(i.split()[0] == 'num_lstm_layers'):
                    num_lstm_layers = int(i.split()[1])

        if(gated_layers == -1):
            gated_layers = len(gated_hidden_size)
        if(fc_layers == -1):
            fc_layers = len(fc_hidden_size)

        print("using the following settings:")
        print("--" * 20)

        print("epochs: {:1d}".format(epochs))
        print("Small Dataset?: " + str(test))
        print("embedding size: {:1d}".format(embedding_size))
        
        print("fc layers: {:1d}".format(fc_layers))
        print("fc hidden layer: " + str(fc_hidden_size))
        print("fc activation: " + str(fc_activation))
        print("fc batch norm: " + str(fc_batch_norm))
        print("fc dropout: {:.2f}".format(fc_dropout))

        print("gated layers: {:1d}".format(gated_layers))
        print("gated hidden layers: " + str(gated_hidden_size))
        print("gated activation: " + str(gated_activation))
        print("gated dropout: {:.2f}".format(gated_dropout))
        print("gated batch norm: " + str(gated_batch_norm))
        print("gated graph norm: " + str(gated_graph_norm))

        print("num lstm iters: " + str(num_lstm_iters))
        print("num lstm layer: " + str(num_lstm_layers))
        print("--" * 20)

        dict_ret = {}
        dict_ret["test"] = test
        dict_ret["epochs"] = epochs
        dict_ret["embedding_size"] = embedding_size
        
        dict_ret["fc_hidden_size"] = fc_hidden_size
        dict_ret["fc_layers"] = fc_layers
        dict_ret['fc_dropout'] = fc_dropout
        dict_ret['fc_batch_norm'] = fc_batch_norm
        dict_ret['fc_activation'] = fc_activation

        dict_ret["gated_hidden_size"] = gated_hidden_size
        dict_ret["gated_layers"] = gated_layers
        dict_ret["gated_activation"] = gated_activation
        dict_ret["gated_graph_norm"] = gated_graph_norm
        dict_ret["gated_batch_norm"] = gated_batch_norm
        dict_ret['gated_dropout'] = gated_dropout
        
        dict_ret["num_lstm_iters"] = num_lstm_iters
        dict_ret["num_lstm_layers"] = num_lstm_layers
        
        return dict_ret 

if __name__ == "__main__":

    # seed random number generators 
    seed_torch()
    dict_ret = parse_settings()
    
    if(bool(dict_ret["test"])):
        mols, attrs, labels = read_input_files(
            'examples/train/molecules.sdf', 
            'examples/train/molecule_attributes.yaml', 
            'examples/train/reactions.yaml', 
        )
    else:
        # todo 
        mols, attrs, labels = read_input_files(
            'examples/train/molecules_libe.sdf', 
            'examples/train/molecule_attributes_libe.yaml', 
            'examples/train/reactions_libe.yaml', 
        )
         
        # todo
        mols_mg, attrs_mg, labels_mg = read_input_files(
            '../train/molecules_libe.sdf', 
            '../train/train/molecule_attributes_libe.yaml', 
            '../train/train/reactions_libe.yaml', 
        )
        
        print("sheesh")
        #mols, attrs , labels = read_input_files()


    dataset = ReactionNetworkDataset(
        grapher=get_grapher(),
        molecules=mols,
        labels=labels,
        extra_features=attrs
    )

    trainset, valset, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)

    # we train with a batch size of 100
    train_loader = DataLoaderReactionNetwork(trainset, batch_size=100,shuffle=True)
    val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
    test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)

    model = GatedGCNReactionNetwork(
        in_feats=dataset.feature_size,
        embedding_size=dict_ret["embedding_size"],
        gated_num_layers=dict_ret["gated_layers"],
        gated_hidden_size=dict_ret["gated_hidden_size"],
        gated_activation=dict_ret["gated_activation"],
        gated_dropout=float(dict_ret["gated_dropout"]),
        gated_graph_norm=int(dict_ret["gated_graph_norm"]),
        gated_batch_norm=int(dict_ret["gated_batch_norm"]),
        fc_num_layers=dict_ret["fc_layers"],
        fc_hidden_size=dict_ret["fc_hidden_size"],
        fc_activation=dict_ret["fc_activation"],
        fc_dropout=float(dict_ret["fc_dropout"]),
        fc_batch_norm=int(dict_ret["fc_batch_norm"]),
        num_lstm_iters=dict_ret["num_lstm_iters"],
        num_lstm_layers=dict_ret["num_lstm_layers"],
        conv="GatedGCNConv"
    )

    # optimizer, loss function and metric function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = MSELoss(reduction="mean")
    metric = WeightedL1Loss(reduction="sum")
    feature_names = ["atom", "bond", "global"]
    best = 1e10

    # main training loop
    print("# Epoch     Loss         TrainAcc        ValAcc")
    t1 = time.time()

    for epoch in range(dict_ret["epochs"]):
        if(epoch % 5 == 0):

            # train on training set 
            loss, train_acc = train( optimizer, model, feature_names, train_loader, loss_func, metric)

            # evaluate on validation set
            val_acc = evaluate(model, feature_names, val_loader, metric)

            # save checkpoint for best performing model 
            if (val_acc < best):
                best = val_acc
                torch.save(model.state_dict(), 'checkpoint.pkl')
            
            print("{:5d}   {:12.6e}   {:12.6e}   {:12.6e}".format(epoch, loss, train_acc, val_acc))
    t2 = time.time()


    # load best performing model and test it's performance on the test set
    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint)
    test_acc = evaluate(model, feature_names, test_loader, metric)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])    

    print("TestAcc: {:12.6e}".format(test_acc))
    print("Time to Train: {:5.1f} seconds".format(float(t2 - t1)))
    print("Number of Trainable Model Params: {}".format(params))


    model = GatedGCNReactionNetwork(
        in_feats=dataset.feature_size,
        embedding_size=dict_ret["embedding_size"],
        gated_num_layers=dict_ret["gated_layers"],
        gated_hidden_size=dict_ret["gated_hidden_size"],
        gated_activation=dict_ret["gated_activation"],
        gated_dropout=float(dict_ret["gated_dropout"]),
        gated_graph_norm=int(dict_ret["gated_graph_norm"]),
        gated_batch_norm=int(dict_ret["gated_batch_norm"]),
        fc_num_layers=dict_ret["fc_layers"],
        fc_hidden_size=dict_ret["fc_hidden_size"],
        fc_activation=dict_ret["fc_activation"],
        fc_dropout=float(dict_ret["fc_dropout"]),
        fc_batch_norm=int(dict_ret["fc_batch_norm"]),
        num_lstm_iters=dict_ret["num_lstm_iters"],
        num_lstm_layers=dict_ret["num_lstm_layers"],
        conv="GatedGCNConv"
    )

    print("-" * 20 + "now disabling gradients" + "-" * 20)
    model.gated_layers.requires_grad_(False)
    
    #model.fc_layers.requires_grad_(False)
    #model.readout_layer.requires_grad_(False)
    
    
    best = 1e10
    
    # main training loop
    print("# Epoch     Loss         TrainAcc        ValAcc")
    t1 = time.time()

    for epoch in range(dict_ret["epochs"]):
        if(epoch % 5 == 0):
            # train on training set 
            loss, train_acc = train( optimizer, model, feature_names, train_loader, loss_func, metric)

            # evaluate on validation set
            val_acc = evaluate(model, feature_names, val_loader, metric)

            # save checkpoint for best performing model 
            
            if (val_acc < best):
                best = val_acc
                torch.save(model.state_dict(), 'checkpoint.pkl')
                
            print("{:5d}   {:12.6e}   {:12.6e}   {:12.6e}".format(epoch, loss, train_acc, val_acc))

    t2 = time.time()

    # load best performing model and test it's performance on the test set
    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint)
    test_acc = evaluate(model, feature_names, test_loader, metric)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])    

    print("TestAcc: {:12.6e}".format(test_acc))
    print("Time to Train: {:5.1f} seconds".format(float(t2 - t1)))
    print("Number of Trainable Model Params: {}".format(params))

