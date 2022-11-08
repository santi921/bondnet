#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH --mail-user=santiagovargas921@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 20:00:00
#SBATCH -A jcesr_g

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1
module load cudatoolkit/11.5
module load pytorch/1.11
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1  python controller_train.py
