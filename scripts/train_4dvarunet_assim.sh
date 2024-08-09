#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/train-fdvarunet_woscale_assim-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/fdvarunet_woscale_assim.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

srun python src/train.py model=fdvarunet_woscale paths=assim_hpc datamodule=ncassimilate_z500 datamodule.batch_size=128 trainer.accumulate_grad_batches=1 trainer.max_epochs=50 test=False task_name=fdvarunet_woscale_assim

