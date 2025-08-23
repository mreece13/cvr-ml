#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --signal=SIGUSR1@180
#SBATCH --time=05:30:00

# Set up environment
module load miniforge/24.3.0-0
module load cuda/12.4.0

mamba activate cvr-ml

# Run your application
srun python main_lightning.py --data data/colorado.parquet --epochs 10 --batch-size=512
