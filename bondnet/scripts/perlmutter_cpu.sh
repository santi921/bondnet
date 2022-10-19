#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=santiagovargas921@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1
module load cudatoolkit/11.5
module load pytorch/1.11
# srun conda activate bondnet
srun -n 1 -c 256 --cpu_bind=cores python controller_train.py
