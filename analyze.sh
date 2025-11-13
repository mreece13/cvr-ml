#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=128G

# Set up environment
module load miniforge/24.3.0-0
module load cuda/12.4.0

mamba activate cvr-ml

python analyze_embeddings.py --checkpoint "lightning_logs/version_0/checkpoints/epoch=0-step=3209.ckpt"
