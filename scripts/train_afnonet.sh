#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --output=./slurmlogs/train-afnonet-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/afnonet.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

srun python src/train.py model=afnonet paths=forecast_hpc datamodule=ncforecast_z500 test=True task_name=afnonet

