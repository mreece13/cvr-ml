#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 48
#SBATCH --mem=256G

# Set up environment
module load miniforge/24.3.0-0

mamba activate cvr-ml

Rscript "data/create_sample.R"
