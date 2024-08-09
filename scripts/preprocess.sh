#!/bin/sh

#SBATCH -N 1
#SBATCH -p qfree
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=2.8deg_preprocess
#SBATCH --mem=50G
#SBATCH --output=./slurmlogs/preprocess_2.8deg-%j.out


module load CUDA/12.2.2

python src/data_factory/nc2h5_equally_era5.py --root_dir=../../project/weatherbench --save_dir=../../project/train_forecast_2.8deg --start_train_year=2000
