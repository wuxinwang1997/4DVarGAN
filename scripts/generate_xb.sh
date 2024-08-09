#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --output=./slurmlogs/generate-background-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/gen_xb.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

srun python src/data_factory/generate_background_nc.py --lead_time=120
