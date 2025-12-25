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

# Parse command line arguments
BATCH_SIZE=${1:-512}
HIDDEN_SIZE=${2:-64}
EMB_DIM=${3:-16}
LR=${4:-1e-3}
N_SAMPLES=${5:-1}

# Run your application
set -e  # Exit on first error

srun python main_lightning.py \
    --data data/combined_sample.parquet \
    --batch-size=$BATCH_SIZE \
    --latent-dims=2 \
    --hidden-size=$HIDDEN_SIZE \
    --emb-dim=$EMB_DIM \
    --lr=$LR \
    --epochs=20 \
    --n-samples=$N_SAMPLES

# If we get here, training completed successfully
echo "Training completed successfully!"
exit 0  # Success - no resubmission needed
