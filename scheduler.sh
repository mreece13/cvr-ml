#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --signal=SIGUSR1@360
#SBATCH --time=05:30:00

# Set up environment
module load miniforge/24.3.0-0
module load cuda/12.4.0

mamba activate cvr-ml

# Run your application
srun python main_lightning.py --data data/colorado.parquet --batch-size=512 --latent-dims 1 --epochs 20
# srun python main_lightning.py --data data/colorado.parquet --batch-size=512 --latent-dims 2 --epochs 20
srun python main_lightning.py --data data/colorado.parquet --batch-size=512 --eval-only True --latent-dims 1
#srun python main_lightning.py --data data/colorado.parquet --batch-size=512 --eval-only True --latent-dims 2
