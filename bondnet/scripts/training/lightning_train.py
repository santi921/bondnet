import wandb, argparse, json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

import numpy as np 
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch
from bondnet.model.training_utils import (
    get_grapher, 
    LogParameters, 
    load_model_lightning
)

seed_torch()
import torch
torch.set_float32_matmul_precision("high") # might have to disable on older GPUs


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--on_gpu', default=False, action='store_true')
    parser.add_argument('--debug',  default=False, action='store_true')    
    parser.add_argument('-project_name', type=str, default="hydro_lightning")
    parser.add_argument('-dataset_loc', type=str, default="../../dataset/qm_9_merge_3_qtaim.json")
    parser.add_argument('-log_save_dir', type=str, default="./logs_lightning/")
    parser.add_argument("-config", type=str, default="./settings.json")

    args = parser.parse_args()

    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    config = args.config

    config = json.load(open(config, "r"))
    precision = config["precision"]
    if precision == "16" or precision == "32":
        precision = int(precision)

    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    extra_keys = config["extra_features"]

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(extra_keys), 
        file=dataset_loc, 
        out_file="./", 
        target = config["target_var"], 
        classifier = False, 
        classif_categories=3, 
        filter_species = [3, 5],
        filter_outliers=True,
        filter_sparse_rxns=False,
        debug = debug,
        device = device,
        extra_keys=extra_keys,
        extra_info=config["extra_info"]
        )
     
    
    dict_for_model = {
        "extra_features": extra_keys,
        "classifier": False,
        "classif_categories": 3,
        "filter_species": [3, 5],
        "filter_outliers": True,
        "filter_sparse_rxns": False,
        "debug": debug,
        "in_feats": dataset.feature_size
    }

    config.update(dict_for_model)
    
    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )
    
    print(">"*40 + "config_settings" + "<"*40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))

    print(">"*40 + "config_settings" + "<"*40)

    val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
    test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)
    train_loader = DataLoaderReactionNetwork(
        trainset, 
        batch_size=config['batch_size'], 
        shuffle=True)   
    
    
    model = load_model_lightning(config, device=device, load_dir=log_save_dir)

    if config["transfer"]:
        with wandb.init(project=project_name+"_transfer") as run_transfer:
            
            dataset_transfer = ReactionNetworkDatasetGraphs(
                grapher=get_grapher(extra_keys), 
                file=dataset_loc, 
                out_file="./", 
                target = config["target_var_transfer"], 
                classifier = False, 
                classif_categories=3, 
                filter_species = [3, 5],
                filter_outliers=True,
                filter_sparse_rxns=False,
                debug = debug,
                device = device,
                extra_keys=extra_keys,
                extra_info=config["extra_info"]
            )   
            trainset_transfer, valset_transfer, _ = train_validation_test_split(
                dataset_transfer, validation=0.15, test=0.0  )
            val_transfer_loader = DataLoaderReactionNetwork(valset_transfer, batch_size=len(valset), shuffle=False)
            train_loader_loader = DataLoaderReactionNetwork(
                    trainset_transfer, 
                    batch_size=config['batch_size'], 
                    shuffle=True)
        
            log_parameters = LogParameters()
            logger_tb_transfer = TensorBoardLogger(log_save_dir, name="test_logs_transfer")
            logger_wb_transfer = WandbLogger(project=project_name, name="test_logs_transfer")
            lr_monitor_transfer = LearningRateMonitor(logging_interval='step')
            
            checkpoint_callback_transfer = ModelCheckpoint(
                dirpath=log_save_dir, 
                filename='model_lightning_transfer_{epoch:02d}-{val_l1:.2f}',
                monitor='val_l1',
                mode='min',
                auto_insert_metric_name=True,
                save_last=True
            )
            
            early_stopping_callback_transfer = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.00,
                    patience=500,
                    verbose=False,
                    mode='min'
                )
            
            trainer_transfer = pl.Trainer(
                max_epochs=config["max_epochs_transfer"], 
                accelerator='gpu', 
                devices = [0],
                accumulate_grad_batches=5, 
                enable_progress_bar=True,
                gradient_clip_val=1.0,
                callbacks=[
                    early_stopping_callback_transfer,
                    lr_monitor_transfer, 
                    log_parameters, 
                    checkpoint_callback_transfer],
                enable_checkpointing=True,
                default_root_dir=log_save_dir,
                logger=[
                    logger_tb_transfer, 
                    logger_wb_transfer],
                precision=precision
            )

            trainer_transfer.fit(
                model, 
                train_loader_loader, 
                val_transfer_loader 
                )
            
            # freezing logic
            model_parameters_prior = filter(lambda p: p.requires_grad, model.parameters())
            params_prior = sum([np.prod(p.size()) for p in model_parameters_prior])    
            if(config["freeze"]): model.gated_layers.requires_grad_(False)
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(">"*25 + "Freezing Module" + "<"*25)
            print("Freezing Gated Layers....")
            print("Number of Trainable Model Params Prior: {}".format(params_prior))
            print("Number of Trainable Model Params: {}".format(params))

        run_transfer.finish()


    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
        logger_wb = WandbLogger(project=project_name, name="test_logs")
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=log_save_dir, 
            filename='model_lightning_{epoch:02d}-{val_l1:.2f}',
            monitor='val_l1',
            mode='min',
            auto_insert_metric_name=True,
            save_last=True
        ) 

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=500,
            verbose=False,
            mode='min'
            )
        
        trainer = pl.Trainer(
            max_epochs=config["max_epochs"], 
            accelerator='gpu', 
            devices = [0],
            accumulate_grad_batches=5, 
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            callbacks=[
                early_stopping_callback,
                lr_monitor, 
                log_parameters, 
                checkpoint_callback],
            enable_checkpointing=True,
            default_root_dir=log_save_dir,
            logger=[logger_tb, logger_wb],
            precision=precision
        )

        trainer.fit(
            model, 
            train_loader, 
            val_loader
            )

        trainer.test(
            model, 
            test_loader)
            
    run.finish()
