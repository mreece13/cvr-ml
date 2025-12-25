#!/bin/bash

# Hyperparameter Tuning Job Submitter
# Reads hyperparam-grid.txt and submits a job for each line

GRID_FILE="hyperparam-grid.txt"
LOG_DIR="hyperparam_logs"

# Create log directory
mkdir -p $LOG_DIR

# Counter for jobs
job_count=0
job_ids=()

# Read the grid file and submit jobs (skip comment lines)
while read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    
    # Parse hyperparameters
    read -r batch_size hidden_size emb_dim lr n_samples <<< "$line"
    
    job_count=$((job_count + 1))
    
    # Submit the job with hyperparameters as arguments
    jobid=$(sbatch --parsable \
        --output="${LOG_DIR}/slurm-%j_bs${batch_size}_hs${hidden_size}_ed${emb_dim}_lr${lr}_ns${n_samples}.out" \
        --job-name="cvr_bs${batch_size}_hs${hidden_size}_ed${emb_dim}" \
        scheduler.sh $batch_size $hidden_size $emb_dim $lr $n_samples)
    
    job_ids+=($jobid)
    
    echo "Submitted job $job_count (ID: $jobid): bs=$batch_size hs=$hidden_size ed=$emb_dim lr=$lr ns=$n_samples"
    
done < "$GRID_FILE"

echo ""
echo "========================================"
echo "Submitted $job_count hyperparameter tuning jobs"
echo "Job IDs: ${job_ids[@]}"
echo "========================================"
echo ""
echo "To monitor jobs, use: squeue -u \$USER"
echo "Logs will be saved to: $LOG_DIR/"
