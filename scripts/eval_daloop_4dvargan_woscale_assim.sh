#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/eval-daloop-4dvargan_woscale_assim-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/4dvargan_woscale_assim.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvargan_woscale --obs_partial=0.2
srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvargan_woscale --obs_partial=0.15
srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvargan_woscale --obs_partial=0.1         
srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvargan_woscale --obs_partial=0.05
