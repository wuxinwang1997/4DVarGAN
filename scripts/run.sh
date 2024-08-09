#!/bin/bash

srun -N 1 -p normal --mem=100G python src/data_factory/nc2h5_equally_era5.py --root_dir=../../data/era5 --save_dir=../../data/train_pred
sbatch scripts/train_climax.sh