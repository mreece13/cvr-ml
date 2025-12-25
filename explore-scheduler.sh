#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH --time=5:59:00

# Set up environment
module load miniforge/24.3.0-0
mamba activate cvr-ml

python explore_embeddings.py
