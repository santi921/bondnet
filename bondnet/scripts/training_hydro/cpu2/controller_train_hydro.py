import torch
import os 
from glob import glob
from bondnet.scripts.train_hydro import train_transfer_hydro
from bondnet.model.training_utils import get_grapher
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.utils import parse_settings

def main():

    #path_mg_data = "../dataset/mg_dataset/20220826_mpreact_reactions.json"
    files = glob("settings*.txt")
    #print(files)
    first_setting = files[0]
    dict_train = parse_settings(first_setting)
    
    if(dict_train["classifier"]):
        classif_categories = dict_train["categories"]
    else:classif_categories = None
    
    #if(device == None):
    if dict_train["on_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_train["gpu"] = device
    else:
        device = torch.device("cpu")
        dict_train["gpu"] = "cpu"
    
    # NOTE YOU WANT TO USE SAME FEATURIZER/DEVICE ON ALL RUNS
    # IN THIS FOLDER B/C THIS MAKES IT SO YOU ONLY HAVE TO GEN 
    # DATASET ONCE
    featurizer_xyz = dict_train["featurizer_xyz"] 

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(featurizer_xyz), 
        file=dict_train["dataset_loc"], 
        out_file="./", 
        target = 'dG_sp', 
        classifier = dict_train["classifier"], 
        classif_categories=classif_categories, 
        filter_species = dict_train["filter_species"],
        filter_sparse_rxns=dict_train["filter_sparse_rxns"],
        filter_outliers=dict_train["filter_outliers"],
        debug = dict_train["debug"],
        device = dict_train["gpu"] 
    )

    for ind, file in enumerate(files):
        train_transfer_hydro(file, 
                dataset = dataset, 
                device = dict_train["gpu"]
            )
        os.rename(file, str(ind) + "_done.txt")
main()
