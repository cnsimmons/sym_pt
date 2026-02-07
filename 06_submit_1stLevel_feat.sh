#!/bin/bash
#
# submit_1stLevel_feat.sh - Submit all 1stLevel FEAT jobs with throttling
# Usage: bash submit_1stLevel_feat.sh
#

DATA_DIR="/user_data/csimmon2/sym_pt"
REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_feat"
MAX_RUNNING=12
POLL_INTERVAL=60  # seconds between queue checks

mkdir -p "$LOG_DIR"

# Find all FSF files
FSF_FILES=($(find "$DATA_DIR" -name "1stLevel.fsf" -path "*/derivatives/fsl/loc/run-*" | sort))
TOTAL=${#FSF_FILES[@]}

echo "Found $TOTAL FSF files to submit"
echo "Max concurrent jobs: $MAX_RUNNING"
echo "Logs: $LOG_DIR"
echo ""

submitted=0

for fsf in "${FSF_FILES[@]}"; do
    # Extract sub/ses/run for job naming
    # Path: .../sub-XXX/ses-XX/derivatives/fsl/loc/run-XX/1stLevel.fsf
    sub=$(echo "$fsf" | grep -oP 'sub-\K[^/]+')
    ses=$(echo "$fsf" | grep -oP 'ses-\K[^/]+')
    run=$(echo "$fsf" | grep -oP 'run-\K[^/]+')
    job_name="feat_${sub}_${ses}_${run}"

    # Check if FEAT already ran
    feat_dir="$(dirname $fsf)/1stLevel.feat"
    if [ -f "$feat_dir/stats/cope1.nii.gz" ]; then
        echo "SKIP: $job_name (already complete)"
        continue
    fi

    # Throttle: wait until running jobs drop below limit
    while true; do
        n_running=$(squeue -u $USER -h | grep -c "feat_")
        if [ "$n_running" -lt "$MAX_RUNNING" ]; then
            break
        fi
        echo "  Waiting... ($n_running jobs running)"
        sleep $POLL_INTERVAL
    done

    # Submit
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00

export FSLDIR=/opt/fsl/6.0.3
. \${FSLDIR}/etc/fslconf/fsl.sh
export PATH=\${FSLDIR}/bin:\${PATH}

echo "Running FEAT: $job_name on \$(hostname)"
feat $fsf
EOT

    ((submitted++))
    echo "[$submitted/$TOTAL] Submitted: $job_name"

done

echo ""
echo "Done! Submitted $submitted jobs."
echo "Monitor with: squeue -u \$USER"