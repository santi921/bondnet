#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --mail-user=santiagovargas921@gmail.com
#SBATCH --mail-type=ALL
module load anaconda3
conda activate bondnet
python controller_train_hydro.py

