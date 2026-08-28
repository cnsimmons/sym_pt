#!/bin/bash
# Usage: bash submit_searchlight_distinct.sh [stage]
#   bash submit_searchlight_distinct.sh 1        # searchlight only
#   bash submit_searchlight_distinct.sh 1,2      # searchlight + register
#   bash submit_searchlight_distinct.sh 3,4      # combat + randomise
#   bash submit_searchlight_distinct.sh all      # everything
#
# Single-subject test before committing the whole cohort:
#   bash submit_searchlight_distinct.sh 1 sub-004

REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_searchlight_distinct"
mkdir -p $LOG_DIR

STAGE="${1:-all}"
SUB="${2:-}"

SUBARG=""
if [ -n "$SUB" ]; then
  SUBARG="--sub $SUB"
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=sl_distinct
#SBATCH --output=${LOG_DIR}/sl_%j.out
#SBATCH --error=${LOG_DIR}/sl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00

echo "Running on \\$(hostname)"
echo "stage: ${STAGE}   sub: ${SUB:-all}"

source /etc/fsl/fsl.sh 2>/dev/null || module load fsl-6.0.3 2>/dev/null || true
which flirt || { echo "ERROR: flirt not found, set up FSL"; exit 1; }
which randomise || { echo "ERROR: randomise not found, set up FSL"; exit 1; }

python $REPO_DIR/F_harmonization/combat_07_searchlight_distinctiveness.py \\
    --stage ${STAGE} ${SUBARG}
EOT

echo "Submitted. Monitor: squeue -u \$USER"
