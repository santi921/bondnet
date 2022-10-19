#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 20:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --mail-user=santiagovargas921@gmail.com
#SBATCH --mail-type=ALL
module load anaconda3
conda activate bondnet
python controller_train.py
