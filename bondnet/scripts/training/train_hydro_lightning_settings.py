import wandb, argparse, json
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



if __name__ == "__main__": 
    # add argparse to get these parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('-method', type=str, default="bayes")
    parser.add_argument('-on_gpu', type=bool, default=True)
    parser.add_argument('-debug', type=bool, default=True)
    parser.add_argument('-precision', type=str, default=16)
    parser.add_argument('-project_name', type=str, default="hydro_lightning")
    parser.add_argument('-dataset_loc', type=str, default="../../dataset/qm_9_merge_3_qtaim.json")
    parser.add_argument('-log_save_dir', type=str, default="./logs_lightning/")
    parser.add_argument('-target_var', type=str, default="dG_sp")
    parser.add_argument("-config", type=str, default="./settings.json")

    args = parser.parse_args()

    method = args.method
    on_gpu = args.on_gpu
    debug = args.debug
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    precision = args.precision
    config = args.config
    config = json.load(open(config, "r"))
    target_var = args.target_var

    if precision == "16" or precision == "32":
        precision = int(precision)

    if on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(["bond_length"]), 
        file=dataset_loc, 
        out_file="./", 
        target = target_var, 
        classifier = False, 
        classif_categories=3, 
        filter_species = [3, 5],
        filter_outliers=True,
        filter_sparse_rxns=False,
        debug = debug,
        device = device,
        extra_keys=["bond_length"],
        extra_info=["functional_group_reacted"]
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