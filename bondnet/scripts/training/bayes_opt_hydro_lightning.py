import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch, parse_settings
from bondnet.model.training_utils import get_grapher, LogParameters
from bondnet.model.gated_reaction_network_lightning import GatedGCNReactionNetworkLightning

seed_torch()
import torch
torch.set_float32_matmul_precision("high") # might have to disable on older GPUs

sweep_params = {
    "batch_size": {"values": [128, 256]}, 
    "weight_decay": {"values": [0.0001, 0.00]},
    "augment": {"values": [False]},
    "restore": {"values": [False]},
    "on_gpu": {"values": [True]},
    "restore": {"values": [False]},
    "embedding_size": {"values": [8, 12, 16]},
    "gated_dropout": {"values": [0.0, 0.1, 0.2]},
    "gated_num_layers": {"values": [1, 2, 3]},
    #"gated_hidden_size": {"values": [64, 128, 256]},
    "gated_activation": {"values": ["ReLU"]},
    "gated_batch_norm": {"values": [True, False]},
    "gated_graph_norm": {"values": [True, False]},
    "gated_num_fc_layers": {"values": [1, 2, 3]},
    "gated_residual": {"values": [True, False]},
    "num_lstm_iters": {"values": [9, 11, 13, 15]},
    "num_lstm_layers": {"values": [1, 2, 3]},
    "fc_dropout": {"values": [0.0, 0.1, 0.2]},
    "fc_batch_norm": {"values": [True, False]},
    "fc_num_layers": {"values": [1, 2, 3]},
    "epochs": {"values": [100]},
    #"fc_hidden_size": {"values": [64, 128, 256]},
    "fc_activation": {"values": ["ReLU"]},
    "loss": {"values": ["mse", "huber", "mae"]},
    "extra_features": {"values": [["bond_length"]]},
    "gated_hidden_size_1": {"values":[512, 1024]},
    "gated_hidden_size_shape": {"values":["flat", "cone"]},
    "fc_hidden_size_1": {"values":[512, 1024]},
    "fc_hidden_size_shape": {"values":["flat", "cone"]},
    "learning_rate": {"values": [0.0001, 0.00001, 0.000001]},
}


def load_model_lightning(dict_train, device=None, load_dir=None): 
    """
    returns model and optimizer from dict of parameters
        
    Args:
        dict_train(dict): dictionary
    Returns: 
        model (pytorch model): model to train
        optimizer (pytorch optimizer obj): optimizer
    """

    if(device == None):
        if dict_train["on_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dict_train["gpu"] = device
        else:
            device = torch.device("cpu")
            dict_train["gpu"] = "cpu"
    else: dict_train["gpu"] = device

    if dict_train["restore"]: 
        print(":::RESTORING MODEL FROM EXISTING FILE:::")
        
        if load_dir == None:
            load_dir = "./"
        
        try: 
            
            model = GatedGCNReactionNetworkLightning.load_from_checkpoint(
                checkpoint_path=load_dir + "/last.ckpt")
            model.to(device)
            print(":::MODEL LOADED:::")
            return model
        
        except: 
            print(":::NO MODEL FOUND LOADING FRESH MODEL:::")
    
    shape_fc = dict_train["fc_hidden_size_shape"]
    shape_gat = dict_train["gated_hidden_size_shape"]
    base_fc = dict_train["fc_hidden_size_1"]
    base_gat = dict_train["gated_hidden_size_1"]

    if(shape_fc == "flat"):
        fc_layers = [base_fc for i in range(dict_train["fc_num_layers"])]
    else:
        fc_layers = [int(base_fc/(2**i)) for i in range(dict_train["fc_num_layers"])]

    if(shape_gat == "flat"):
        gat_layers = [base_gat for i in range(dict_train["gated_num_layers"])]
    else:
        gat_layers = [int(base_gat/(2**i)) for i in range(dict_train["gated_num_layers"])]

    model = GatedGCNReactionNetworkLightning(
            in_feats=dict_train['in_feats'],
            embedding_size=dict_train['embedding_size'],
            gated_dropout=dict_train["gated_dropout"],
            gated_num_layers=len(gat_layers),
            gated_hidden_size=gat_layers,
            gated_activation=dict_train['gated_activation'],
            gated_batch_norm=dict_train["gated_batch_norm"],
            gated_graph_norm=dict_train["gated_graph_norm"],
            gated_num_fc_layers=dict_train["gated_num_fc_layers"],
            gated_residual=dict_train["gated_residual"],
            num_lstm_iters=dict_train["num_lstm_iters"],
            num_lstm_layers=dict_train["num_lstm_layers"],
            fc_dropout=dict_train["fc_dropout"],
            fc_batch_norm=dict_train['fc_batch_norm'],
            fc_num_layers=len(fc_layers),
            fc_hidden_size=fc_layers,
            fc_activation=dict_train['fc_activation'],
            learning_rate=dict_train['learning_rate'],
            weight_decay=dict_train['weight_decay'],
            scheduler_name="reduce_on_plateau",
            warmup_epochs=10, 
            max_epochs = dict_train["epochs"],
            eta_min=1e-6,
            loss_fn=dict_train["loss"],
            augment=dict_train["augment"],
            device=device
    )
    model.to(device)
    
    return model

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
    
    method = "bayes"
    sweep_config = {}
    on_gpu = True
    debug = True
    project_name = "hydro_lightning"
    extra_features = ["bond_length"]
    dataset_loc = "../../dataset/qm_9_merge_3_qtaim.json"
    log_save_dir = "./logs_lightning/"

    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(extra_features), 
        file=dataset_loc, 
        out_file="./", 
        target = 'dG_sp', 
        classifier = False, 
        classif_categories=3, 
        filter_species = [3, 5],
        filter_outliers=True,
        filter_sparse_rxns=False,
        debug = debug,
        device = device,
        extra_keys=extra_features,
        )
    
    sweep_config["parameters"] = sweep_params
    dict_for_model = {
        "extra_features": ["bond_length"],
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
    
    sweep_id = wandb.sweep(sweep_config, project="project_name")
    training_obj = TrainingObject(dataset, device, dict_for_model, log_save_dir)
    wandb.agent(sweep_id, function=training_obj.train, count=300)