#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/tune-daloop-4dvar_assim-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/tune_4dvar_assim.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.0 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.1 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.2 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.3 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.4 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.5 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.6 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.7 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.8 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.9 --maxIter=1 
srun python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=1.0 --maxIter=1 