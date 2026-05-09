#!/bin/bash
# Usage: bash submit_extract_selective_voxel_counts.sh

REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_extract_selective_voxel_counts"
mkdir -p $LOG_DIR

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=selvox
#SBATCH --output=${LOG_DIR}/selvox_%j.out
#SBATCH --error=${LOG_DIR}/selvox_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

echo "Running on \$(hostname)"
python $REPO_DIR/D_liu/extract_selective_voxel_counts.py
EOT

echo "Submitted. Monitor: squeue -u \$USER"