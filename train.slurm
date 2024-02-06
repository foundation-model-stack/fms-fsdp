#!/bin/bash

#SBATCH --nodes=16
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
##SBATCH --contiguous

srun train.sh
