#!/bin/bash
# submit_extract_roi_betas.sh - Submit roi-beta extraction (single job)
# Usage: bash submit_extract_roi_betas.sh

# 1. Define paths
REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_extract_roi_betas"

# 2. Setup logs
mkdir -p $LOG_DIR
echo "Logs will be saved to: $LOG_DIR"

# 3. Submit single job
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=extract_roi_betas
#SBATCH --output=${LOG_DIR}/extract_%j.out
#SBATCH --error=${LOG_DIR}/extract_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

echo "Running on \$(hostname)"
python $REPO_DIR/D_liu/extract_roi_betas.py
EOT

echo "Submitted. Monitor with: squeue -u \$USER"