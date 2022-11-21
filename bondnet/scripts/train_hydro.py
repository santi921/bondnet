import time, wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt 
from bondnet.model.metric import EarlyStopping
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch, pickle_dump, parse_settings
from bondnet.model.training_utils import (
    evaluate, 
    evaluate_classifier, 
    train, 
    train_classifier, 
    load_model, 
    evaluate_r2, 
    get_grapher,
    evaluate_breakdown
)
seed_torch()
import torch


def train_hydro(
    settings_file = "settings.txt", 
    device = None, 
    dataset = None):
    
    best = 1e10
    feature_names = ["atom", "bond", "global"]
    dict_train = parse_settings(settings_file)

    if(dict_train["classifier"]):
        classif_categories = dict_train["categories"]
        run = wandb.init(project="project_hydro_class", reinit=True)
    else:
        classif_categories = None
        run = wandb.init(project="project_hydro", reinit=True)

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
            grapher=get_grapher(dict_train["extra_features"]), 
            file=dict_train["dataset_loc"], 
            out_file="./", 
            target = 'dG_sp', 
            classifier = dict_train["classifier"], 
            classif_categories=classif_categories, 
            filter_species = dict_train["filter_species"],
            filter_outliers=dict_train["filter_outliers"],
            filter_sparse_rxns=dict_train["filter_sparse_rxns"],
            debug = dict_train["debug"],
            device = dict_train["gpu"],
            extra_keys=dict_train["extra_keys"]
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
        optimizer, mode="min", factor=0.8, patience=50, verbose=True)
    stopper = EarlyStopping(patience=100)

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
                weight= dict_train["category_weights"],
                device = dict_train["gpu"],
                categories = classif_categories,
                augment=dict_train["augment"]
            )

            # evaluate on validation set
            val_acc, f1_score = evaluate_classifier(
                model, 
                feature_names, 
                val_loader, 
                device = dict_train["gpu"]
            )
            wandb.log({"loss": loss})
            wandb.log({"acc train": train_acc})
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
            wandb.log({"mae_train": train_acc})
            wandb.log({"mae_val": val_acc})
            wandb.log({"r2_val": val_r2})
            wandb.log({"r2_train": train_r2})

            print(
                "{:5d}   {:12.6e}   {:12.2e}   {:12.6e}   {:.2f}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, val_r2, train_r2
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
            device = dict_train["gpu"]
        )


        wandb.log({"acc test": test_acc})
        wandb.log({"f1 test": test_f1})
        print("Test Acc: {:12.6e}".format(test_acc))
        print("Test F1: {:12.6e}".format(test_f1))

    else: 
        test_acc = evaluate(model, feature_names, test_loader, device = dict_train["gpu"], plot=True)
        #dict_res = evaluate_breakdown(model, feature_names, test_loader, device = dict_train["gpu"])
        #wandb.log({"mae_val_breakdown": dict_res})
        wandb.log({"mae_test": test_acc})
        print("Test MAE: {:12.6e}".format(test_acc))
    
    t2 = time.time()
    print("Time to Training: {:5.1f} seconds".format(float(t2 - t1)))
    
    run.finish()

