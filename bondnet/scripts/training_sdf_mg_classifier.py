import time
from importlib_metadata import Prepared
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from bondnet.data.dataset import ReactionNetworkDatasetClassify
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import (
    AtomFeaturizerFull,
    AtomFeaturizerMinimum,
    BondAsNodeFeaturizerMinimum,
    GlobalFeaturizer,
    BondAsNodeFeaturizerFull,
)
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.dataset import train_validation_test_split
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.model.gated_reaction_network_classifier import (
    GatedGCNReactionNetworkClassifier,
)
from bondnet.model.gated_reaction_network_debug import GatedGCNReactionNetworkDebug
from bondnet.scripts.create_label_file import read_input_files
from bondnet.model.metric import WeightedL1Loss
from bondnet.utils import seed_torch
from torchsummary import summary

print(torch.__version__)
seed_torch()
path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"


def train(optimizer, model, nodes, data_loader, loss_fn):

    model.train()
    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (batched_graph, label) in enumerate(data_loader):

        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        stdev = label["scaler_stdev"]

        pred, target_filtered, stdev_filtered = model(
            batched_graph, feats, label["reaction"], target, stdev
        )
        target_filtered = torch.reshape(
            target_filtered, (int(target_filtered.shape[0] / 5), 5)
        )
        target_filtered = torch.argmax(target_filtered, axis=1)
        loss = loss_fn(pred, torch.flatten(target_filtered))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # here is the actual optimizer step

        accuracy += (torch.argmax(pred, axis=1) == target_filtered).sum().item()
        epoch_loss += loss.detach().item()
        count += len(target_filtered)
        f1 = f1_score(
            target_filtered.data,
            torch.argmax(pred, axis=1).detach(),
            average="weighted",
        )

    epoch_loss /= it + 1
    accuracy /= count
    print("molecules considered: {}".format(int(count)))
    return epoch_loss, accuracy, f1


def evaluate(model, nodes, data_loader):
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            stdev = label["scaler_stdev"]

            pred, target_filtered, stdev_filtered = model(
                batched_graph, feats, label["reaction"], target, stdev
            )
            target_filtered = torch.reshape(
                target_filtered, (int(target_filtered.shape[0] / 5), 5)
            )
            target_filtered = torch.argmax(target_filtered, axis=1)

            accuracy += (torch.argmax(pred, axis=1) == target_filtered).sum().item()
            count += len(target_filtered)
            f1 = f1_score(
                target_filtered.data,
                torch.argmax(pred, axis=1).detach(),
                average="weighted",
            )

    return accuracy / count, f1


def get_grapher():
    atom_featurizer = AtomFeaturizerFull()
    # bond_featurizer = BondAsNodeFeaturizerMinimum()
    bond_featurizer = BondAsNodeFeaturizerFull()
    # our example dataset contains molecules of charges -1, 0, and 1
    global_featurizer = GlobalFeaturizer(allowed_charges=[-2, -1, 0, 1])
    grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer)

    return grapher


def training_loop(model, train_loader, val_loader, test_loader):
    t1 = time.time()
    # optimizer, loss function and metric function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_func = MSELoss(reduction="mean")
    loss_func = CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    feature_names = ["atom", "bond", "global"]
    best = 1e10
    num_epochs = 200

    # main training loop
    print(
        "# Epoch     Loss         TrainAcc        ValAcc        F1 train        F1 test"
    )
    for epoch in range(num_epochs):

        # train on training set
        loss, train_acc, f1_train = train(
            optimizer, model, feature_names, train_loader, loss_func
        )
        # evaluate on validation set
        val_acc, f1_val = evaluate(model, feature_names, val_loader)
        # save checkpoint for best performing model
        is_best = val_acc < best
        if is_best:
            best = val_acc
            torch.save(model.state_dict(), "checkpoint.pkl")
        print(
            "{:5d}   {:12.4e}   {:12.4e}   {:12.4e}   {:12.4e}   {:12.4e}".format(
                epoch, loss, train_acc, val_acc, f1_train, f1_val
            )
        )

    # load best performing model and test it's performance on the test set
    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint)
    test_acc = evaluate(model, feature_names, test_loader)
    print("TestAcc: {:12.6e}".format(test_acc))
    t2 = time.time()
    print("Time to Training: {:5.1f} seconds".format(float(t2 - t1)))


def main():
    mols_mg, attrs_mg, labels_mg = read_input_files(
        path_mg_data + "mg_struct_bond_rgrn.sdf",
        path_mg_data + "mg_feature_bond_rgrn.yaml",
        path_mg_data + "mg_label_bond_rgrn.yaml",
    )

    dataset_mg = ReactionNetworkDatasetClassify(
        grapher=get_grapher(),
        molecules=mols_mg,
        labels=labels_mg,
        extra_features=attrs_mg,
    )

    trainset, valset, testset = train_validation_test_split(
        dataset_mg, validation=0.1, test=0.1
    )
    train_loader_mg = DataLoaderReactionNetwork(trainset, batch_size=100, shuffle=True)
    val_loader_mg = DataLoaderReactionNetwork(
        valset, batch_size=len(valset), shuffle=False
    )
    test_loader_mg = DataLoaderReactionNetwork(
        testset, batch_size=len(testset), shuffle=False
    )

    model_mg = GatedGCNReactionNetworkClassifier(
        in_feats=dataset_mg.feature_size,
        embedding_size=24,
        gated_num_layers=2,
        gated_hidden_size=[64, 64, 64],
        gated_activation="ReLU",
        fc_num_layers=2,
        fc_hidden_size=[128, 64],
        fc_activation="ReLU",
        outdim=5,
    )

    training_loop(model_mg, train_loader_mg, val_loader_mg, test_loader_mg)


main()
