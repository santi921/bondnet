
from random import choice 
import numpy as np 
import os, argparse, json


def write_one(dictionary_in, target):
    """
    write all key-value pairs of a dictionary to a json file
    """
    # write write to json file
    with open(target, "w") as f:
        json.dump(dictionary_in, f, indent=4)


def put_file_slurm_in_every_subfolder(folder, project_name, file_loc, gpu=True):
    """
    put a file in every subfolder of a folder
    """
    for subfolder in os.listdir(folder):
        with open(os.path.join(folder, subfolder, "submit.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH -N=1\n")
            f.write("#SBATCH -C gpu\n")
            f.write("#SBATCH -G 1\n")
            f.write("#SBATCH -q regular\n")
            f.write("#SBATCH --mail-user=santiagovargas921@gmail.com\n")
            f.write("#SBATCH --mail-type=ALL\n")
            f.write("#SBATCH -t 12:00:00\n")
            f.write("#SBATCH -A jcesr_g\n\n")

            f.write("export OMP_NUM_THREADS=1\n")
            f.write("export OMP_PLACES=threads\n")
            f.write("export export OMP_PROC_BIND=true\n")

            f.write("module load cudatoolkit/11.5\n")
            f.write("module load pytorch/1.11\n")

            script = "lightning_controller.py"

            f.write("srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1  {} -project_name {} ".format(script, project_name))
            

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
        "gated_num_layers": [1,2,3],
        "fc_num_layers": [1,2,3],
        "gated_hidden_size_1": [512, 1024, 2048],
        "gated_hidden_size_shape": ["flat", "cone"],
        "fc_hidden_size_1": [512, 1024],
        "fc_hidden_size_shape": ["flat", "cone"]
    }


    dictionary_values_options = \
    {
        "filter_outliers": [True],
        "filter_sparse_rxns": [False],
        "filter_species": [[3, 6]],
        "debug": [False],
        "test": [False],
        "batch_size": [128, 256],
        "embedding_size": [8, 10, 12],
        "max_epochs": [500,1000,1500],
        "fc_activation": ["ReLU"],
        "fc_batch_norm": [False],
        "freeze": [False, True],
        "gated_activation": ["ReLU"],
        "gated_num_fc_layers": [1, 2, 3],
        "learning_rate": [0.02, 0.01, 0.001],
        "fc_dropout": [0.0, 0.1, 0.25],
        "gated_dropout": [0.0, 0.1, 0.25],
        "output_file": ["results.pkl"],
        "start_epoch": [0],
        "early_stop": [True],
        "scheduler": [False],
        "max_epochs_transfer": [250, 500, 1000],
        "transfer": [False, True],
        "freeze" : [True, False],
        "loss": ["huber", "mse"],
        "weight_decay": [0.0, 0.0001, 0.00001],
        "num_lstm_iters": [7, 9, 11],
        "num_lstm_layers": [1, 2],
        "gated_batch_norm": [0, 1],
        "gated_graph_norm":[0,1],
        "gated_residual":[True],
        "fc_batch_norm":[False, True]
    }   
    
    if(options["hydro"]):
        dictionary_values_options["augment"] = [False]
        dictionary_values_options["transfer"] = [False]
        dictionary_values_options["freeze"] = [False]
        dictionary_values_options["filter_sparse_rxns"] = [False]

    else: 
        dictionary_values_options["augment"] = [False, True]
        dictionary_values_options["filter_sparse_rxns"] = [False]


    for i in range(options["num"]):

        dictionary_write = {}
        
        if(options["hydro"] == True):
            featurizer_dict = {
                #"choice_3":{
                #    "extra_features": ["bond_length", 'Lagrangian_K', 'e_density', 'lap_e_density', 
                #            'e_loc_func', 'ave_loc_ion_E', 'delta_g_promolecular', 'delta_g_hirsh', 'esp_nuc', 
                #            'esp_e', 'esp_total', 'grad_norm', 'lap_norm', 'eig_hess', 
                #            'det_hessian', 'ellip_e_dens', 'eta'],
                #    "feature_filter": True
                #},
                #"choice_4":{
                #    "extra_features": ["bond_length", 'esp_total', 'Lagrangian_K', 'ellip_e_dens'], 
                #    "feature_filter": True
                #},
                "choice_5":{
                    "extra_features": ['esp_total'], 
                    "feature_filter": True
                },
                "choice_6":{
                    "extra_features": ["bond_length"], 
                    "feature_filter": True
                },
                #"choice_7":{
                #    "feature_filter": False
                #},
            }

        else: 
            featurizer_dict = {
                '''
                "choice_1":{
                    "extra_features": [
                        "1_s", "2_s", "1_p", "2_p", "1_d", "2_d", "1_f", "2_f", 
                        "1_polar", "2_polar", "occ_nbo", "valence_electrons", "total_electrons", 
                        "partial_charges_nbo", "partial_charges_mulliken", 
                        "partial_charges_resp", "indices_nbo"], 
                    "feature_filter": True
                },
                "choice_2":{
                    "extra_features": [       
                        "esp_nuc", "esp_e", "esp_total",
                        "ellip_e_dens", "indices_qtaim"
                        ],
                    "feature_filter": True
                },
                "choice_3": {
                        "extra_features": [     
                        "esp_nuc", "esp_e", "esp_total", "ellip_e_dens", "bond_length", 
                        ],
                    "feature_filter": True
                }, 
                '''

                "choice_4":{
                    "extra_features": ["bond_length", "esp_total"], 
                    "feature_filter": True
                },
                "choice_5":{
                    "extra_features": ["bond_length"], 
                    "feature_filter": False
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


        dictionary_write["fc_num_layers"] = choice(dictionary_archi["fc_num_layers"])
        dictionary_write["gated_num_layers"] = choice(dictionary_archi["gated_num_layers"])
        dictionary_write["gated_hidden_size_1"] = base_gat
        dictionary_write["fc_hidden_size_1"] = base_fc
        dictionary_write["fc_hidden_size_shape"] = shape_fc
        dictionary_write["gated_hidden_size_shape"] = shape_gat
        dictionary_write["dataset_loc"] = options["dataset_loc"]
        dictionary_write["on_gpu"] = options["gpu"]
        dictionary_write["classifier"] = options["classifier"]
        dictionary_write["restore"] = False
        dictionary_write["precision"] = "32"
        dictionary_write["restore_path"] = None
        
        if(options["hydro"]):
            dictionary_write["target_var"] = "dG_sp"
            dictionary_write["extra_info"] = ["functional_group_reacted"]
        else:
            dictionary_write["target_var"] = "ts"
            dictionary_write["target_var_transfer"] = "diff"
            dictionary_write["extra_info"] = []
            
        for k, v in dictionary_values_options.items():
            dictionary_write[k] = choice(v)

        if(options["hydro"]):
            folder = "./hydro_lightning"
            
        
        else: 
            folder = "./mg_lightning"

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
        target += "/settings_" + str(i) + ".json"
        # sort keys
        dictionary_write = dict(sorted(dictionary_write.items()))
        write_one(dictionary_write, target)
            
    put_file_slurm_in_every_subfolder(
        folder=folder, 
        project_name=options["project_name"],
        file_loc=options["dataset_loc"], 
        gpu=options["gpu"]
    )


def main():
    # create argparse 
    parser = argparse.ArgumentParser(description='Create settings files for training')
    parser.add_argument('--perlmutter', action='store_true', help='Use perlmutter')
    parser.add_argument('--gpu', action='store_true', help='Use gpu')
    parser.add_argument('--hydro', action='store_true', help='Use hydro')
    parser.add_argument('--classifier', action='store_true', help='Use classifier')
    parser.add_argument('--class_cats', type=int, default=3, help='Number of categories')
    parser.add_argument('--project_name', type=str, default="mg_lightning", help='Project name')
    parser.add_argument('--num', type=int, default=50, help='Number of runs')
    parser.add_argument('--per_folder', type=int, default=50, help='Number of runs per folder')
    args = parser.parse_args()
    options = vars(args)

    classifier = options["classifier"]
    class_cats = options["class_cats"]
    gpu = options["gpu"]
    hydro = options["hydro"]
    perlmutter = options["perlmutter"]
    num = options["num"]
    per_folder = options["per_folder"]
    project_name = options["project_name"]

    
    if(classifier and class_cats != 3 and class_cats != 5):
        raise ValueError("Must have 3 or 5 categories for classifier")

    if hydro:
        options["dataset_loc"] = "../../../../dataset/qm_9_merge_3_qtaim.json"
    
    else: 
        options["dataset_loc"] = "../../../../dataset/mg_dataset/mg_qtaim_complete.json"
    
    generate_and_write(options)

    
main()

