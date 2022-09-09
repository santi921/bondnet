import torch
import time, wandb, itertools, copy

import numpy as np 
from tqdm import tqdm
#from sklearn.metrics import r2_score
from torchmetrics import R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bondnet.model.metric import EarlyStopping
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
#from bondnet.scripts.create_label_file import read_input_files
#from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss
from bondnet.utils import seed_torch, pickle_dump, parse_settings
from bondnet.model.training_utils import (
    evaluate, 
    evaluate_classifier, 
    train, 
    train_classifier, 
    load_model, 
    evaluate_r2, 
    get_grapher
)
seed_torch()

def train_transfer(settings_file = "settings.txt", device = None, dataset = None, dataset_transfer = None):
    
    best = 1e10
    feature_names = ["atom", "bond", "global"]
    dict_train = parse_settings(settings_file)
    path_mg_data = "../../../dataset/mg_dataset/20220826_mpreact_reactions.json"
        
    if(dict_train["classifier"]):
        classif_categories = 5 # update this later
        run = wandb.init(project="project_classification_test", reinit=True)
    else:
        classif_categories = None
        run = wandb.init(project="project_regression", reinit=True)

    if(device == None):
        if dict_train["on_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dict_train["gpu"] = device
        else:
            device = torch.device("cpu")
            dict_train["gpu"] = "cpu"
    else: dict_train["gpu"] = device
    
    wandb.config.update(dict_train)

    print("train on device: {}".format(dict_train["gpu"]))
    if(dataset == None):
       dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), 
        file=path_mg_data, 
        out_file="./", 
        target = 'ts', 
        filter_species = dict_train["filter_species"],
        classifier = dict_train["classifier"], 
        classif_categories=classif_categories, 
        debug = dict_train["debug"],
        device = device 
        )
    
    dict_train['in_feats'] = dataset.feature_size
    model, optimizer, optimizer_transfer = load_model(dict_train)
    model.to(device)

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

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=50, verbose=True)
    scheduler_transfer = ReduceLROnPlateau(
        optimizer_transfer, mode="min", factor=0.4, patience=30, verbose=True)
    stopper = EarlyStopping(patience=100)
    stopper_transfer = EarlyStopping(patience=100)

    if(dict_train['transfer']):
        if(dataset_transfer == None):
            dataset_transfer = ReactionNetworkDatasetGraphs(
            grapher=get_grapher(), file=path_mg_data, out_file="./", 
            target = 'diff', 
            classifier = dict_train["classifier"], 
            classif_categories=classif_categories, 
            filter_species = dict_train["filter_species"],
            debug = dict_train["debug"],
            device = dict_train["gpu"]
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

        #print("Initiating Training w/ transfer...")
        #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        #params = sum([np.prod(p.size()) for p in model_parameters])
        #print("Number of Trainable Model Params: {}".format(params))
        
        
        for epoch in tqdm(range(dict_train['transfer_epochs'])):
            if(dict_train["classifier"]):
                loss_transfer, train_acc_transfer = train_classifier(
                    model, 
                    feature_names, 
                    dataset_transfer_loader,
                    optimizer_transfer, 
                    device = dict_train["gpu"], 
                    categories = classif_categories
                )
                val_acc_transfer, f1_score = evaluate_classifier(
                    model, 
                    feature_names, 
                    dataset_transfer_loader_val, 
                    device = dict_train["gpu"],
                    categories = classif_categories
                )
                wandb.log({"transfer_val_acc": val_acc_transfer})
                wandb.log({"transfer_val_f1": f1_score})
                wandb.log({"loss_transfer": loss_transfer})
                wandb.log({"train_acc_transfer": train_acc_transfer})
            
            else:
                loss_transfer, train_acc_transfer = train(
                    model, 
                    feature_names, 
                    dataset_transfer_loader, 
                    optimizer_transfer, 
                    device = dict_train["gpu"],
                    augment=dict_train["augment"]
                )
                val_acc_transfer = evaluate(
                    model, 
                    feature_names, 
                    dataset_transfer_loader_val, 
                    device = dict_train["gpu"]
                )
                wandb.log({"loss_transfer": loss_transfer})
                wandb.log({"train_acc_transfer": train_acc_transfer})
                wandb.log({"val_acc_transfer": val_acc_transfer})
                
            scheduler_transfer.step(val_acc_transfer)
            if stopper_transfer.step(val_acc_transfer): break

        # freeze model layers but fc
        #model.gated_layers.requires_grad_(False)
        #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        #params = sum([np.prod(p.size()) for p in model_parameters])
        #print("Freezing Gated Layers....")
        #print("Number of Trainable Model Params: {}".format(params))

    t1 = time.time()

    if(dict_train["classifier"]):
        print("# Epoch     Loss         TrainAcc        ValAcc        ValF1")
    else: 
        print("# Epoch     Loss         TrainAcc        ValAcc        ValR2")        
    
    for epoch in range(dict_train['epochs']):
        # train on training set
        if(dict_train["classifier"]):
            loss, train_acc = train_classifier(
                model, 
                feature_names, 
                train_loader, 
                optimizer, 
                device = dict_train["gpu"],
                categories = classif_categories
            )

            # evaluate on validation set
            val_acc, f1_score = evaluate_classifier(
                model, 
                feature_names, 
                val_loader, 
                device = dict_train["gpu"],
                categories = classif_categories
            )

            wandb.log({"acc validation": val_acc})
            wandb.log({"f1 validation": f1_score})
            print(
                "{:5d}   {:12.6e}   {:12.2e}   {:12.6e}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, f1_score
                )
            )
            
        else: 
            loss, train_acc = train(
                model, 
                feature_names, 
                train_loader, 
                optimizer, 
                device = dict_train["gpu"],
                augment=dict_train["augment"]
                )
            # evaluate on validation set
            val_acc = evaluate(
                model, 
                feature_names, 
                val_loader, 
                device = dict_train["gpu"]
                )
            val_r2 = evaluate_r2(
                model, 
                feature_names, 
                val_loader, 
                device = dict_train["gpu"]
                )
            train_r2 = evaluate_r2(
                model, 
                feature_names, 
                train_loader, 
                device = dict_train["gpu"]
                )
        
            wandb.log({"loss": loss})
            wandb.log({"mae_val": val_acc})
            wandb.log({"r2_val": val_r2})
            wandb.log({"r2_val": train_r2})

            print(
                "{:5d}   {:12.6e}   {:12.2e}   {:12.6e}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, val_r2
                )
            )

        # save checkpoint for best performing model
        is_best = val_acc < best
        if is_best:
            best = val_acc
            torch.save(model.state_dict(), "checkpoint.pkl")

        if(dict_train["early_stop"]):
            if stopper.step(val_acc):
                pickle_dump(
                    best, dict_train["save_hyper_params"]
                )  # save results for hyperparam tune
                break
        scheduler.step(val_acc)

    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint)
    
    if(dict_train["classifier"]):
        test_acc, test_f1 = evaluate_classifier(
            model, 
            feature_names, 
            test_loader, 
            device = dict_train["gpu"],
            categories = classif_categories
        )

        wandb.log({"acc validation": test_acc})
        wandb.log({"f1 validation": test_f1})
        print("Test Acc: {:12.6e}".format(test_acc))
        print("Test F1: {:12.6e}".format(test_f1))

    else: 
        test_acc = evaluate(model, feature_names, test_loader, device = dict_train["gpu"])
        wandb.log({"mae_test": test_acc})
        print("Test MAE: {:12.6e}".format(test_acc))
    
    t2 = time.time()
    print("Time to Training: {:5.1f} seconds".format(float(t2 - t1)))
    run.finish()


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