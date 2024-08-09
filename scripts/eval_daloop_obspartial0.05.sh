#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/eval_daloop_obspartial5%-%j.out

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=vit --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarnet --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=cyclegan_wscale --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarunet_woscale --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarunet_wscale --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvargan_woscale --obs_partial=0.05  

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvargan_wscale --obs_partial=0.05        

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_woscale --obs_partial=0.05         

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_scalein --obs_partial=0.05

srun python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvar --inflation=0.5 --maxIter=1 --obs_partial=0.05
