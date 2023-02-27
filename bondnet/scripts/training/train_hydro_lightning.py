import wandb
                                 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch, parse_settings
from bondnet.model.training_utils import get_grapher, load_model_lightning, LogParameters

seed_torch()
import torch
torch.set_float32_matmul_precision("high") # might have to disable on older GPUs

def main():

    dataset = None
    settings_file = "settings_lightning.txt" 
    log_save_dir = "./logs_lightning/" + settings_file.split(".")[0]

    dict_train = parse_settings(settings_file)
    if dict_train["on_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_train["gpu"] = device
    else:
        device = torch.device("cpu")
        dict_train["gpu"] = "cpu"

    print("train on device: {}".format(dict_train["gpu"]))

    if dict_train["wandb"]:
        run = wandb.init(project="hydro_lightning", reinit=True)
        wandb.config.update(dict_train)
    
        
    if(dataset == None):
        dataset = ReactionNetworkDatasetGraphs(
            grapher=get_grapher(dict_train["extra_features"]), 
            file=dict_train["dataset_loc"], 
            out_file="./", 
            target = 'dG_sp', 
            classifier = dict_train["classifier"], 
            classif_categories=3, 
            filter_species = dict_train["filter_species"],
            filter_outliers=dict_train["filter_outliers"],
            filter_sparse_rxns=dict_train["filter_sparse_rxns"],
            debug = dict_train["debug"],
            device = dict_train["gpu"],
            extra_keys=dict_train["extra_features"],
            )
    
    dict_train['in_feats'] = dataset.feature_size

    model = load_model_lightning(dict_train, device=device, load_dir=log_save_dir)

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )
    
    train_loader = DataLoaderReactionNetwork(trainset, batch_size=dict_train['batch_size'], shuffle=True)
    val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
    test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)
    log_parameters = LogParameters()
    logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
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

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min'
        )
    
    trainer = pl.Trainer(
        #max_epochs=dict_train["epochs"], 
        max_epochs = 20, 
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
    print(dataset.labels[0].keys())
    for i in dataset.labels: 
        print(i["value"], i["value_rev"])
    
    trainer.test(
        model, 
        test_loader)
    trainer.test(
        model, 
        train_loader
    )
        
    if dict_train["wandb"]:
        run.finish()

main()