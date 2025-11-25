#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 48
#SBATCH --mem=256G

# Set up environment
module load miniforge/24.3.0-0
mamba activate cvr-ml

newmeta="https://docs.google.com/spreadsheets/d/1Pq9sNcCfLVi-qeXfBy7xEi5lPxpMJYy3LVUHQn_uEFI/edit?gid=1814631761#gid=1814631761"

my_key="$(echo "${newmeta}" | cut -d'/' -f6)"
my_sheet="$(echo "${newmeta}" | cut -d'#' -f2 | cut -d '=' -f2)"

curl -fsSL -o metadata/newmeta.csv "https://docs.google.com/spreadsheets/d/${my_key}/export?format=csv&usp=sharing&gid=${my_sheet}"

Rscript "data/create_data.R"
Rscript "data/create_sample.R"
