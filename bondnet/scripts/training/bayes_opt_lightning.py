import wandb, argparse, torch, json

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
    def __init__(self, dataset, device, dict_for_model, log_save_dir): 
        self.dataset = dataset
        self.dict_for_model = dict_for_model
        self.device = device
        self.in_feats = dataset.feature_size

        self.log_save_dir = log_save_dir
        
        trainset, valset, testset = train_validation_test_split(
            dataset, validation=0.15, test=0.15
        )
        self.trainset = trainset
        self.val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
        self.test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)


    def make_model(self, config):
        config['in_feats'] = self.in_feats
        model = load_model_lightning(config, device=self.device, load_dir=self.log_save_dir)
        return model


    def train(self):
        
        with wandb.init(project="hydro_lightning") as run:
            log_parameters = LogParameters()
            logger_tb = TensorBoardLogger("./logs_lightning/", name="test_logs")
            logger_wb = WandbLogger(project="hydro_lightning", name="test_logs")
            lr_monitor = LearningRateMonitor(logging_interval='step')
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.log_save_dir, 
                filename='model_lightning_{epoch:02d}-{val_l1:.2f}',
                monitor='val_l1',
                mode='min',
                auto_insert_metric_name=True,
                save_last=True
            )
            config = wandb.config
            #run.config.update(dict_for_model)
            # log dict_for_model 
            wandb.log(dict_for_model)
            self.train_loader = DataLoaderReactionNetwork(self.trainset, batch_size=config['batch_size'], shuffle=True)
            model = self.make_model(config)

            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=500,
                verbose=False,
                mode='min'
                )
            
            trainer = pl.Trainer(
                max_epochs=2000, 
                #max_epochs = 20, 
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
                precision="bf16"
            )
            trainer.fit(
                model, 
                self.train_loader, 
                self.val_loader
                )

            trainer.test(
                model, 
                self.test_loader)
            
        run.finish()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', type=str, default="bayes")
    parser.add_argument('-on_gpu', type=bool, default=True)
    parser.add_argument('-debug', type=bool, default=True)
    parser.add_argument('-project_name', type=str, default="bayes_hydro_lightning")
    parser.add_argument('-dataset_loc', type=str, default="../../dataset/qm_9_merge_3_qtaim.json")
    parser.add_argument('-log_save_dir', type=str, default="./logs_lightning/")
    parser.add_argument('-project_name', type=str, default="hydro_lightning")
    parser.add_argument('-sweep_config', type=str, default='sweep_config.yaml')

    args = parser.parse_args()

    method = args.method
    on_gpu = args.on_gpu
    debug = args.debug
    augment = args.augment
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    wandb_project_name = args.wandb_project
    sweep_config = args.sweep_config

    config = json.load(open(sweep_config, "r"))
    extra_keys = config["extra_features"]
    sweep_params = json.load(open(sweep_config, "r"))["parameters"]

    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        

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
        )
    
    sweep_config["parameters"] = sweep_params
    dict_for_model = {
        "extra_features": extra_keys,
        "classifier": False,
        "classif_categories": 3,
        "filter_species": [3, 5],
        "filter_outliers": True,
        "filter_sparse_rxns": False,
        "debug": debug,
    }

    if method == "bayes":
        sweep_config["method"] = method
        sweep_config["metric"] = {"name": "val_l1", "goal": "minimize"}
    
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    training_obj = TrainingObject(
        dataset, 
        device, 
        dict_for_model, 
        log_save_dir,
        project_name=wandb_project_name)
    wandb.agent(sweep_id, function=training_obj.train, count=300)