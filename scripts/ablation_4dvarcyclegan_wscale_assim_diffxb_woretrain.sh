#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_3090
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=./slurmlogs/eval-daloop-4dvarcyclegan_wscale_assim-diffxb-woretrain-%j.out

nvidia-smi dmon -d 30 -s um -o T > ./slurmlogs/4dvarcyclegan_wscale_assim_diffxb_woretrain.log &
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

module load CUDA/12.2.2

# srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=24 
# srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=48      
# srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=72         
# srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=96                 
# srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=120

srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --init_time=24 
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --init_time=48      
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --init_time=72         
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --init_time=96                 
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.15 --init_time=120

srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --init_time=24 
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --init_time=48      
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --init_time=72         
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --init_time=96                 
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 --init_time=120

srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --init_time=24 
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --init_time=48      
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --init_time=72         
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --init_time=96                 
srun python src/evaluate/ablation_daloop_woretrain.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.05 --init_time=120