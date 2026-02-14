#!/bin/bash
#
# 11_submit_highlevel_feat.sh - Submit HighLevel FEAT jobs with throttling
# Usage: bash 11_submit_highlevel_feat.sh
#

DATA_DIR="/user_data/csimmon2/sym_pt"
REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_highlevel"
MAX_RUNNING=20
POLL_INTERVAL=60

mkdir -p "$LOG_DIR"

FSF_FILES=($(find "$DATA_DIR" -name "HighLevel.fsf" -path "*/derivatives/fsl/loc/*" | sort))
TOTAL=${#FSF_FILES[@]}

echo "Found $TOTAL HighLevel FSF files to submit"
echo "Max concurrent jobs: $MAX_RUNNING"
echo ""

submitted=0

for fsf in "${FSF_FILES[@]}"; do
    sub=$(echo "$fsf" | grep -oP 'sub-\K[^/]+')
    ses=$(echo "$fsf" | grep -oP 'ses-\K[^/]+')
    job_name="hlvl_${sub}_${ses}"

    # Check if already ran
    gfeat_dir="$(dirname $fsf)/HighLevel.gfeat"
    if [ -f "$gfeat_dir/cope1.feat/stats/cope1.nii.gz" ]; then
        echo "SKIP: $job_name (already complete)"
        continue
    fi

    # Throttle
    while true; do
        n_running=$(squeue -u $USER -h | grep -c "hlvl_")
        if [ "$n_running" -lt "$MAX_RUNNING" ]; then
            break
        fi
        echo "  Waiting... ($n_running jobs running)"
        sleep $POLL_INTERVAL
    done

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

echo "Running HighLevel FEAT: $job_name on \$(hostname)"
feat $fsf
EOT

    ((submitted++))
    echo "[$submitted/$TOTAL] Submitted: $job_name"

done

echo ""
echo "Done! Submitted $submitted jobs."