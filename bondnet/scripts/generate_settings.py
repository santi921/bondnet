
from random import choice 
import numpy as np 
import os 

def write_one(dictionary_in, target, hydro = False, perlmutter = False):
    """
    write all key-value pairs of a dictionary to a file
    """
    with open(target, "w") as f:
        for key, value in dictionary_in.items():
            f.write("{} {}\n".format(key, value))

def put_file_in_every_subfolder(folder, file):
    """
    put a file in every subfolder of a folder
    """
    for subfolder in os.listdir(folder):
        copy_file(file, os.path.join(folder, subfolder, file))

def copy_file(src, dst):
    """
    copy a file from src to dst
    """
    with open(src, "r") as f:
        content = f.read()  
    with open(dst, "w") as f:
        f.write(content)

def check_folder(folder):
    """
    check if folder exists, if not create it
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def generate_and_write(options):

    dictionary_categories = \
    {
        "categories": [3, 5],
        "category_weights_3": [[1.0, 1.5, 2.0], [1.0, 3.0, 4.0], [1.0, 5.0, 5.0]],
        "category_weights_5": [[2.0, 1.0, 2.0, 1.5, 1.0], [4.0, 1.0, 4.0, 3.0, 1.0], [5.0, 1.0, 5.0, 3.0, 1.0]]
    }

    dictionary_archi = \
    {
        "gated_num_layers": [1,2,3,4],
        "fc_num_layers": [1,2,3],
        "gated_hidden_size_1": [128, 256, 512, 1024],
        "gated_hidden_size_shape": ["flat", "cone"],
        "fc_hidden_size_1": [64, 128, 256],
        "fc_hidden_size_shape": ["flat", "cone"]
    }


    dictionary_values_options = \
    {
        "filter_outliers": [True],
        "filter_sparse_rxns": [True],
        "filter_species": [[1, 2],[2, 4],[3, 6]],
        "augment": [False, True],
        "debug": [False],
        "test": [False],
        "batch_size": [128, 256],
        "embedding_size": [24, 32, 64],
        "epochs": [1000,1500,2000],
        "fc_activation": ["ReLU"],
        "fc_batch_norm": [False, True],
        "freeze": [False, True],
        "gated_activation": ["ReLU"],
        "gated_num_fc_layers": [1, 2, 3, 4],
        "lr": [0.001, 0.0001],
        "output_file": ["results.pkl"],
        "start_epoch": [0],
        "early_stop": [True],
        "scheduler": [False, True],
        "transfer_epochs": [500, 1000, 1500],
        "transfer": [True],
        "freeze" : [True, False],
        "loss": ["mse", "huber"] 
    }

    for i in range(options["num"]):

        dictionary_write = {}

        if(options["hydro"] == True):
            featurizer_dict = {
                "choice_1":{
                    "xyz_featurizer": True,
                    "featurizer_electronic_bond": False,
                    "electronic_featurizer": False,
                    "featurizer_filter":False
                },
                "choice_2":{
                    "xyz_featurizer": False,
                    "featurizer_electronic_bond": False,
                    "electronic_featurizer": False,
                    "featurizer_filter":False
                }
            }
        else: 
            featurizer_dict = {
                "choice_1":{
                    "xyz_featurizer": True,
                    "featurizer_electronic_bond": False,
                    "electronic_featurizer": False,
                    "featurizer_filter":False
                },
                "choice_2":{
                    "xyz_featurizer": True,
                    "featurizer_electronic_bond": False,
                    "electronic_featurizer": False,
                    "featurizer_filter":False
                },
                "choice_3":{
                    "xyz_featurizer": True,
                    "featurizer_electronic_bond": False,
                    "electronic_featurizer": False,
                    "featurizer_filter":False
                }
            }

        featurizer_settings = choice(list(featurizer_dict.keys()))
        featurizer_settings = featurizer_dict[featurizer_settings]
        dictionary_write.update(featurizer_settings)

        if(options["class_cats"] == 3):
            dictionary_write["categories"] = 3
            dictionary_write["category_weights"] = choice(dictionary_categories["category_weights_3"])
        else: 
            dictionary_write["categories"] = 5
            dictionary_write["category_weights"] = choice(dictionary_categories["category_weights_5"])
            
        base_fc = choice(dictionary_archi["gated_hidden_size_1"])
        base_gat = choice(dictionary_archi["fc_hidden_size_1"])
        shape_fc = choice(dictionary_archi["fc_hidden_size_shape"])
        shape_gat = choice(dictionary_archi["gated_hidden_size_shape"])

        if(shape_fc == "flat"):
            fc_layers = [base_fc for i in range(choice(dictionary_archi["fc_num_layers"]))]
        else:
            fc_layers = [int(base_fc/(2**i)) for i in range(choice(dictionary_archi["fc_num_layers"]))]

        if(shape_gat == "flat"):
            gat_layers = [base_gat for i in range(choice(dictionary_archi["gated_num_layers"]))]
        else:
            gat_layers = [int(base_gat/(2**i)) for i in range(choice(dictionary_archi["gated_num_layers"]))]
        
        
        dictionary_write["gated_hidden_size"] = gat_layers
        dictionary_write["fc_hidden_size"] = fc_layers
        dictionary_write["dataset_loc"] = options["dataset_loc"]
        dictionary_write["on_gpu"] = options["gpu"]
        dictionary_write["classifier"] = options["classifier"]
        for k, v in dictionary_values_options.items():
            dictionary_write[k] = choice(v)

        if(options["hydro"]):
            folder = "./hydro_training"
        else: 
            folder = "./mg_training"

        if(options["gpu"]): folder += "_gpu"
        else: folder += "_cpu"

        if(options["classifier"]):
            folder += "_classifier/"
        else:
            folder += "_regressor/"

        check_folder(folder)

        if(options["gpu"]): target = folder + "gpu_"
        else: target = folder + "cpu_"

    
        target += str(int(np.floor(i / options["per_folder"])))
        check_folder(target)
        target += "/settings" + str(i) + ".txt"
        
        write_one(dictionary_write, target, hydro=options["hydro"], perlmutter = options["perlmutter"])

    if(options["hydro"]):
        controller_file = "controller_train_hydro_source.py"
    else:
        controller_file = "controller_train_mg_source.py"

    if(options["perlmutter"] == False): 
        if(options["gpu"]):
            slurm_file = "./xsede_gpu.sh"
        else: 
            slurm_file = "./xsede_cpu.sh"

    else: 
        if(options["gpu"]):
            slurm_file = "./perlmutter_gpu.sh"

        else: 
            slurm_file = "./perlmutter_cpu.sh"
        
    put_file_in_every_subfolder(folder, controller_file)
    put_file_in_every_subfolder(folder, slurm_file)

def main():

    classifier = False
    class_cats = 5
    hydro = False
    num = 50 
    per_folder = 5
    gpu = False
    perlmutter = False

    if hydro:
        dataset_loc = "../../../dataset/mg_dataset/merged_mg.json"
    else: 
        dataset_loc = "../../../dataset/dataset/rev_corrected_bonds_qm_9_hydro_training.json"

    options_dict = {
        "dataset_loc": dataset_loc,
        "classifier": classifier,
        "class_cats": class_cats, 
        "hydro": hydro, 
        "num": num,  
        "per_folder": per_folder,
        "gpu": gpu,
        "perlmutter": perlmutter
    }

    generate_and_write(options_dict)

    
main()
