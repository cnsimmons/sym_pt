#!/bin/bash
# Usage: bash submit_tfce_votc_contrasts.sh

REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_tfce_votc"
mkdir -p $LOG_DIR

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=tfce_votc
#SBATCH --output=${LOG_DIR}/tfce_%j.out
#SBATCH --error=${LOG_DIR}/tfce_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00

echo "Running on \$(hostname)"

# FSL setup — adjust if needed
source /etc/fsl/fsl.sh 2>/dev/null || module load fsl 2>/dev/null || true
which randomise || { echo "ERROR: randomise not found, set up FSL"; exit 1; }

python $REPO_DIR/D_liu/tfce_votc_contrasts.py
EOT

echo "Submitted. Monitor: squeue -u \$USER"
