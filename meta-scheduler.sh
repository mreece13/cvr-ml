#!/bin/bash

# Configuration
MAX_SUBMISSIONS=10  # Maximum number of times to resubmit
SCRIPT_PATH="scheduler.sh"
LOG_DIR="slurm_logs"

mkdir -p $LOG_DIR

# Function to submit job with dependency
submit_job() {
    local dep_jobid=$1
    local iteration=$2
    
    if [ -z "$dep_jobid" ]; then
        # First submission (no dependency)
        jobid=$(sbatch --parsable \
                      --output="${LOG_DIR}/slurm-%j.out" \
                      --error="${LOG_DIR}/slurm-%j.err" \
                      --job-name="cvr_iter${iteration}" \
                      $SCRIPT_PATH)
    else
        # Subsequent submission with dependency on previous job
        # afternotok: run only if previous job failed/was killed (non-zero exit)
        # afterany: run after previous job completes (any exit status)
        jobid=$(sbatch --parsable \
                      --output="${LOG_DIR}/slurm-%j.out" \
                      --error="${LOG_DIR}/slurm-%j.err" \
                      --job-name="cvr_iter${iteration}" \
                      --dependency=afternotok:${dep_jobid} \
                      $SCRIPT_PATH)
    fi
    
    echo $jobid
}

# Submit the chain
echo "Starting job chain (max $MAX_SUBMISSIONS submissions)..."
prev_jobid=""

for i in $(seq 1 $MAX_SUBMISSIONS); do
    jobid=$(submit_job "$prev_jobid" $i)
    
    if [ $i -eq 1 ]; then
        echo "Submitted initial job: $jobid"
    else
        echo "Submitted continuation job $i: $jobid (depends on $prev_jobid)"
    fi
    
    prev_jobid=$jobid
done