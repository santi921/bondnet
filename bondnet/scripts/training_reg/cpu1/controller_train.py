import torch
from glob import glob
from bondnet.scripts.train_transfer import train_transfer
from bondnet.model.training_utils import get_grapher
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.utils import parse_settings

def main():

    path_mg_data = "../../../dataset/mg_dataset/20220613_reaction_data.json"
    files = glob("settings*.txt")
    first_setting = files[0]
    dict_train = parse_settings(first_setting)
    
    if(dict_train["classifier"]):classif_categories = 5 # update this later
    else:classif_categories = None
    if(device == None):
        if dict_train["on_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dict_train["gpu"] = device
        else:
            device = torch.device("cpu")
            dict_train["gpu"] = "cpu"
    else: 
        dict_train["gpu"] = device

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), 
        file=path_mg_data, 
        out_file="./", 
        target = 'ts', 
        classifier = dict_train["classifier"], 
        classif_categories=classif_categories, 
        debug = dict_train["debug"],
        device = dict_train["gpu"] 
    )
    dataset_transfer = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), file=path_mg_data, out_file="./", 
        target = 'diff', 
        classifier = dict_train["classifier"], 
        classif_categories=classif_categories, 
        debug = dict_train["debug"],
        device = dict_train["gpu"]
    )

    for file in files:
        try:
        #print(file)
            train_transfer(file, 
                dataset = dataset, 
                dataset_transfer = dataset_transfer, 
                device = device["gpu"]
            )
        except:
            pass
main()


# Todo implement a job for each device but more on that later
#devices = [torch.device("cuda:" + str(i)) for i in range(torch.cuda.device_count())]
