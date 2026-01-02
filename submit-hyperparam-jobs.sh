#!/bin/bash

# Resume-aware Hyperparameter Job Submitter
# Reads hyperparam-grid.txt and submits only unfinished combos

GRID_FILE="hyperparam-grid.txt"
LOG_DIR="hyperparam_logs"
MAX_SUBMITS=${MAX_SUBMITS:-50}

# Optional positional args: data path and latent dims
DATA_PATH=${1:-"data/combined_sample.parquet"}
LATENT_DIMS=${2:-2}

mkdir -p "$LOG_DIR"

total_jobs=$(( $(squeue --me | wc -l) - 1 ))
submitted_new_jobs=0
hit_cap=0

# Read grid and submit pending combinations
while read -r line; do
    # Skip comments and blank lines
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # Parse hyperparameters: batch_size hidden_size emb_dim lr n_samples
    read -r batch_size hidden_size emb_dim lr n_samples <<< "$line"

    # Compute lr_slug to match Python naming: decimal -> trim zeros/dot -> remove dot
    lr_decimal=$(awk -v v="$lr" 'BEGIN{printf("%.10f", v)}')
    lr_trim=$(echo "$lr_decimal" | sed -E 's/0+$//; s/\.$//')
    [[ -z "$lr_trim" ]] && lr_trim="$lr_decimal"
    lr_slug=${lr_trim//./}

    # Build file_name identical to main_lightning.py
    data_token="data${DATA_PATH#data/}"
    file_name="${data_token}_batch_size${batch_size}_latent_dims${LATENT_DIMS}_hidden_size${hidden_size}_emb_dim${emb_dim}_lr${lr_slug}_n_samples${n_samples}"

    # Completion check: require all key outputs
    vl="outputs/${file_name}_voter_latents.csv"
    ip="outputs/${file_name}_item_parameters.csv"
    vs="outputs/${file_name}_voter_scores.csv"

    if [[ -f "$vl" && -f "$ip" && -f "$vs" ]]; then
        echo "Completed: skip bs=${batch_size} hs=${hidden_size} ed=${emb_dim} lr=${lr} ns=${n_samples}"
        continue
    fi

    # Respect submission cap
    if (( total_jobs >= MAX_SUBMITS)); then
        echo "Reached cap MAX_SUBMITS=$MAX_SUBMITS; stopping submissions."
        hit_cap=1
        break
    fi

    # Submit the job
    jobid=$(sbatch --parsable \
        --output="${LOG_DIR}/slurm-%j_bs${batch_size}_hs${hidden_size}_ed${emb_dim}_ld${LATENT_DIMS}_lr${lr_slug}_ns${n_samples}.out" \
        --job-name="cvr_bs${batch_size}_hs${hidden_size}_ed${emb_dim}_ld${LATENT_DIMS}" \
        scheduler.sh "$batch_size" "$hidden_size" "$emb_dim" "$lr" "$n_samples" "$DATA_PATH" "$LATENT_DIMS")

    echo "Submitted job (ID: $jobid): bs=$batch_size hs=$hidden_size ed=$emb_dim lr=$lr ns=$n_samples"
    submitted_new_jobs=1
    total_jobs=$(( $(squeue --me | wc -l) - 1 ))
done < "$GRID_FILE"

# If we hit the cap and submitted new jobs, resubmit this script with low priority
# to continue the wave when jobs complete
if (( hit_cap && submitted_new_jobs )); then
    echo "Resubmitting script with low priority for next wave..."
    sbatch --output="${LOG_DIR}/slurm-chain-%j.out" \
        --job-name="cvr_chain_submit" \
        submit-hyperparam-jobs.sh "$DATA_PATH" "$LATENT_DIMS"
fi
