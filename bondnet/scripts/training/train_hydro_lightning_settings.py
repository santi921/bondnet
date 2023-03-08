import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch
from bondnet.model.training_utils import get_grapher, LogParameters, load_model_lightning   

seed_torch()
import torch
torch.set_float32_matmul_precision("high") # might have to disable on older GPUs

config = {
    "batch_size": 128, 
    "weight_decay": 0.000,
    "augment": False,
    "restore": False,
    "on_gpu": True,
    "restore": False,
    "embedding_size": 12,
    "gated_dropout": 0.1,
    "gated_num_layers": 2,
    "gated_activation": "ReLU",
    "gated_batch_norm": False,
    "gated_graph_norm": False,
    "gated_num_fc_layers": 1,
    "gated_residual": True,
    "num_lstm_iters": 3,
    "num_lstm_layers": 1,
    "fc_dropout": 0.2,
    "fc_batch_norm": False,
    "fc_num_layers": 1,
    "epochs": 100,
    "fc_activation": "ReLU",
    "loss": "mae",
    "extra_features": ["bond_length"],
    "gated_hidden_size_1": 512,
    "gated_hidden_size_shape": "flat",
    "fc_hidden_size_1": 256,
    "fc_hidden_size_shape": "flat",
    "learning_rate": 0.001,
}


if __name__ == "__main__": 
    
    method = "bayes"
    sweep_config = {}
    on_gpu = True
    debug = True
    project_name = "hydro_lightning"
    dataset_loc = "../../dataset/qm_9_merge_3_qtaim.json"
    log_save_dir = "./logs_lightning/"

    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(["bond_length"]), 
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
        extra_keys=["bond_length"],
        )
    
    dict_for_model = {
        "extra_features": ["bond_length"],
        "classifier": False,
        "classif_categories": 3,
        "filter_species": [3, 5],
        "filter_outliers": True,
        "filter_sparse_rxns": False,
        "debug": debug,
    }

    config.update(dict_for_model)
    in_feats = dataset.feature_size
    config['in_feats'] = in_feats

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )
    
    trainset = trainset
    val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
    test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)

    
    model = load_model_lightning(config, device=device, load_dir=log_save_dir)


    with wandb.init(project="hydro_lightning") as run:
            log_parameters = LogParameters()
            logger_tb = TensorBoardLogger("./logs_lightning/", name="test_logs")
            logger_wb = WandbLogger(project="hydro_lightning", name="test_logs")
            lr_monitor = LearningRateMonitor(logging_interval='step')
            checkpoint_callback = ModelCheckpoint(
                dirpath=log_save_dir, 
                filename='model_lightning_{epoch:02d}-{val_l1:.2f}',
                monitor='val_l1',
                mode='min',
                auto_insert_metric_name=True,
                save_last=True
            )
            #config = wandb.config
            #run.config.update(dict_for_model)
            # log dict_for_model 
            wandb.log(dict_for_model)
            train_loader = DataLoaderReactionNetwork(trainset, batch_size=config['batch_size'], shuffle=True)
            model = load_model_lightning(config)

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
                default_root_dir=log_save_dir,
                logger=[logger_tb, logger_wb],
                precision="bf16"
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