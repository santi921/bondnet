import chemprop
import pandas as pd

from pymatgen.core import Molecule
from rdkit.Chem import rdChemReactions
from rdkit import Chem

from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA


arguments = [
    "--epochs",
    "100",
    "--ffn_num_layers",
    "2",
    "--hidden_size",
    "1800",
    "--dropout",
    "0.05",
    "--depth",
    "6",
    "--batch_size",
    "512",
    "--reaction",
    "--gpu",
    "1",
    "--explicit_h",
    "--reaction_mode",
    "prod_diff",
    "--save_smiles_splits",
    "--data_path",
    "./hydro_train.csv",
    "--dataset_type",
    "regression",
    "--save_dir",
    "test_checkpoints_reaction",
    
]

args = chemprop.args.TrainArgs().parse_args(arguments)

mean_score, std_score = chemprop.train.cross_validate(
    args=args, train_func=chemprop.train.run_training
)