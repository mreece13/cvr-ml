#!/bin/bash

# Hyperparameter Tuning Job Submitter using SLURM Array Jobs
# Distributes hyperparam-grid.txt across array tasks

# SLURM Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=5:59:00
#SBATCH --signal=SIGUSR1@360
#SBATCH -o hyperparam_logs/slurm-%A-%a.out
#SBATCH -a 1-5
#SBATCH --job-name=vae_hyperparam

# Set up environment
module load miniforge/24.3.0-0
module load cuda/12.4.0

mamba activate cvr-ml

# Function to handle timeout signal
handle_timeout() {
    echo "Job received timeout signal - will be resubmitted"
    exit 124
}

# Trap the timeout signal
trap 'handle_timeout' SIGUSR1

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

# Specify Input File
INPUT_FILE=hyperparam-grid.txt

# Count only non-comment, non-empty lines
NUM_LINES=$(grep -v '^#' $INPUT_FILE | grep -v '^[[:space:]]*$' | wc -l)

echo "Total hyperparameter combinations: " $NUM_LINES

# Distribute line numbers across array tasks
MY_LINE_NUMS=( $(seq $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $NUM_LINES) )

echo "This task will process ${#MY_LINE_NUMS[@]} combinations"

# Iterate over assigned line numbers
for LINE_IDX in "${MY_LINE_NUMS[@]}"; do
    
    # Get the LINE_IDX-th non-comment line from INPUT_FILE
    INPUT="$(grep -v '^#' $INPUT_FILE | grep -v '^[[:space:]]*$' | sed "${LINE_IDX}q;d")"
    
    # Parse hyperparameters
    read -r batch_size hidden_size emb_dim lr n_samples <<< "$INPUT"
    
    echo "========================================"
    echo "Processing combination $LINE_IDX:"
    echo "  batch-size: $batch_size"
    echo "  hidden-size: $hidden_size"
    echo "  emb-dim: $emb_dim"
    echo "  lr: $lr"
    echo "  n-samples: $n_samples"
    echo "========================================"
    
    # Run training with these hyperparameters
    set -e
    srun python main_lightning.py \
        --data data/combined_sample.parquet \
        --batch-size=$batch_size \
        --latent-dims=2 \
        --hidden-size=$hidden_size \
        --emb-dim=$emb_dim \
        --lr=$lr \
        --epochs=20 \
        --n-samples=$n_samples
    
    echo "Completed combination $LINE_IDX"
    echo ""
done

echo "Task $SLURM_ARRAY_TASK_ID completed all assigned combinations!"
exit 0
