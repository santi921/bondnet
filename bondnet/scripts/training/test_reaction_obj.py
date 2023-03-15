import argparse, json

from bondnet.data.dataset import ReactionNetworkDatasetGraphs

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
        debug = True,
        device = device,
        extra_keys=["bond_length"],
        extra_info=["functional_group_reacted"]
        )
    
    print(dataset[0])    
