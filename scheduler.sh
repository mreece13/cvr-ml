#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=5:59:00
#SBATCH --signal=SIGUSR1@360

# Set up environment
module load miniforge/24.3.0-0
module load cuda/12.4.0

mamba activate cvr-ml

# Function to handle timeout signal
handle_timeout() {
    echo "Job received timeout signal - will be resubmitted"
    exit 124  # Special exit code for timeout
}

# Trap the timeout signal
trap 'handle_timeout' SIGUSR1

# Run your application
set -e  # Exit on first error

srun python main_lightning.py --data data/combined_sample.parquet --batch-size=512 --latent-dims 1

# If we get here, training completed successfully
echo "Training completed successfully!"
exit 0  # Success - no resubmission needed
