#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/eval-daloop-4dvarcyclegan_wscale_assim-diffobserr-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/4dvarcyclegan_wscale_assim_diffobserr.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.015
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.02
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.025
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.03
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.035
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.04
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.045
# srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --obs_err=0.05

srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.015
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.02
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.025
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.03
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.035
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.04
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.045
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --obs_err=0.05

srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.015
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.02
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.025
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.03
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.035
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.04
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.045
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --obs_err=0.05

srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.015
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.02
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.025
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.03
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.035
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.04
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.045
srun python src/evaluate/eval_daloop_diffobserr.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --obs_err=0.05