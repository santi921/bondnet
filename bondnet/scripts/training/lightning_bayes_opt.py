import wandb, argparse, torch, json
import numpy as np 

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch
from bondnet.model.training_utils import get_grapher, LogParameters, load_model_lightning
from bondnet.model.gated_reaction_network_lightning import GatedGCNReactionNetworkLightning

seed_torch()
torch.set_float32_matmul_precision("high") # might have to disable on older GPUs


class TrainingObject:
    def __init__(self, config, log_save_dir, project_name, dataset_loc): 
        
        self.config = config
        self.log_save_dir = log_save_dir
        self.wandb_name = project_name
        self.dataset_loc = dataset_loc

        if self.config["parameters"]["on_gpu"]:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.extra_keys = self.config["parameters"]["extra_features"]["values"][0]
        print("extra keys: ", self.extra_keys)
        print("debug value: ", self.config["parameters"]["debug"]["values"])
        print("target value: ", self.config["parameters"]["target_var"]["values"][0])

        self.dataset = ReactionNetworkDatasetGraphs(
            grapher=get_grapher(self.extra_keys), 
            file=self.dataset_loc, 
            out_file="./", 
            target = self.config["parameters"]["target_var"]["values"][0], 
            classifier = self.config["parameters"]["classifier"]["values"][0], 
            classif_categories=self.config["parameters"]["classif_categories"]["values"][0], 
            filter_species = self.config["parameters"]["filter_species"]["values"][0],
            filter_outliers=self.config["parameters"]["filter_outliers"]["values"][0],
            filter_sparse_rxns=self.config["parameters"]["filter_sparse_rxns"]["values"][0],
            debug = self.config["parameters"]["debug"]["values"][0],
            device = self.device,
            extra_keys=self.extra_keys,
            extra_info=None,
        )
        self.in_feats = self.dataset.feature_size
        
        trainset, valset, testset = train_validation_test_split(
            self.dataset, validation=0.15, test=0.15
        )
        self.trainset = trainset
        self.val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
        self.test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)


        if self.config["parameters"]["transfer"]["values"]: 
            self.dataset_transfer = ReactionNetworkDatasetGraphs(
                grapher=get_grapher(self.extra_keys), 
                file=dataset_loc, 
                out_file="./", 
                target = self.config["parameters"]["target_var_transfer"]["values"], 
                classifier = self.config["parameters"]["classifier"]["values"][0], 
                classif_categories=self.config["parameters"]["classif_categories"]["values"][0], 
                filter_species = self.config["parameters"]["filter_species"]["values"][0],
                filter_outliers=self.config["parameters"]["filter_outliers"]["values"][0],
                filter_sparse_rxns=self.config["parameters"]["filter_sparse_rxns"]["values"][0],
                debug = self.config["parameters"]["debug"]["values"][0],
                device = self.device,
                extra_keys=self.extra_keys,
            )

            trainset, valset, _ = train_validation_test_split(
                self.dataset_transfer, validation=0.15, test=0.0
            )
            self.trainset_transfer = trainset
            self.val_loader_transfer = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
            


        
    def make_model(self, config):
        config['in_feats'] = self.in_feats
        model = load_model_lightning(config, device=self.device, load_dir=self.log_save_dir)
        return model


    def train(self):  
        with wandb.init(project=self.wandb_name) as run:
        
            config = wandb.config
            model = self.make_model(config)
        
            if config["transfer"]:
                
                train_loader_transfer = DataLoaderReactionNetwork(
                    self.trainset_transfer, 
                    batch_size=config['batch_size'], 
                    shuffle=True)
            

                log_parameters_transfer = LogParameters()
                logger_tb_transfer = TensorBoardLogger(log_save_dir, name=self.log_save_dir+"_transfer")
                logger_wb_transfer = WandbLogger(project=wandb_project_name, name=self.log_save_dir+"_transfer")
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
                print(config["precision"]*10)

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
                        log_parameters_transfer, 
                        checkpoint_callback_transfer],
                    enable_checkpointing=True,
                    default_root_dir=log_save_dir,
                    logger=[
                        logger_tb_transfer, 
                        logger_wb_transfer],
                    precision=config["precision"]
                )


                trainer_transfer.fit(
                    model, 
                    train_loader_transfer, 
                    self.val_loader_transfer 
                    )
                

                if(config["freeze"]): model.gated_layers.requires_grad_(False)
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                print("Freezing Gated Layers....")
                print("Number of Trainable Model Params: {}".format(params))


            log_parameters = LogParameters()
            logger_tb = TensorBoardLogger(self.log_save_dir, name=self.wandb_name)
            logger_wb = WandbLogger(project=self.wandb_name, name=self.wandb_name)
            lr_monitor = LearningRateMonitor(logging_interval='step')
            
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.log_save_dir, 
                filename='model_lightning_{epoch:02d}-{val_l1:.2f}',
                monitor='val_l1',
                mode='min',
                auto_insert_metric_name=True,
                save_last=True
            )

            train_loader = DataLoaderReactionNetwork(
                self.trainset, 
                batch_size=config['batch_size'], 
                shuffle=True)
            

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
                default_root_dir=self.log_save_dir,
                logger=[logger_tb, logger_wb],
                precision=config["precision"],
            )


            trainer.fit(
                model, 
                train_loader, 
                self.val_loader
                )


            trainer.test(
                model, 
                self.test_loader)


        run.finish()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', type=str, default="bayes")
    parser.add_argument('--on_gpu', default=False, action='store_true')
    parser.add_argument('--debug',  default=False, action='store_true')
    parser.add_argument('--augment', default=False, action='store_true')

    parser.add_argument('-dataset_loc', type=str, default="../../dataset/qm_9_merge_3_qtaim.json")
    parser.add_argument('-log_save_dir', type=str, default="./logs_lightning/")
    parser.add_argument('-project_name', type=str, default="hydro_lightning")
    parser.add_argument('-sweep_config', type=str, default="./sweep_config.json")

    args = parser.parse_args()
    method = str(args.method)
    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    
    
    augment = bool(args.augment)
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    wandb_project_name =  args.project_name
    sweep_config_loc = args.sweep_config
    sweep_config = {}
    sweep_params = json.load(open(sweep_config_loc, "r"))
    sweep_params["debug"] = {"values": [debug]}
    sweep_config["parameters"] = sweep_params
    

    if method == "bayes":
        sweep_config["method"] = method
        sweep_config["metric"] = {"name": "val_l1", "goal": "minimize"}
    
    # wandb loop
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    #print(sweep_config)
    training_obj = TrainingObject(  
        sweep_config,
        log_save_dir,
        dataset_loc=dataset_loc,
        project_name=wandb_project_name)
    
    print("method: {}".format(method))
    print("on_gpu: {}".format(on_gpu))
    print("debug: {}".format(debug))
    print("augment: {}".format(augment))
    print("dataset_loc: {}".format(dataset_loc))
    print("log_save_dir: {}".format(log_save_dir))
    print("wandb_project_name: {}".format(wandb_project_name))
    print("sweep_config_loc: {}".format(sweep_config_loc))

    wandb.agent(sweep_id, function=training_obj.train, count=300)