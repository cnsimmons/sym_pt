#!/bin/bash
# submit_anatomy_new.sh - Submit anatomy jobs for new subjects only
# Usage: bash submit_anatomy_new.sh

REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
PROCESSED_DIR="/user_data/csimmon2/sym_pt"
LOG_DIR="$REPO_DIR/logs_anatomy"
MAX_RUNNING=12
POLL_INTERVAL=60

# New subjects to process tonight
NEW_SUBS=(005 039 082)

mkdir -p $LOG_DIR
echo "Processing ${#NEW_SUBS[@]} new subjects"
echo "Logs: $LOG_DIR"
echo ""

submitted=0
for SUB_ID in "${NEW_SUBS[@]}"; do
    SUB_DIR="$PROCESSED_DIR/sub-${SUB_ID}"

    # Skip if T1w_brain already exists
    if [ -f "$SUB_DIR/ses-01/anat/T1w_brain.nii.gz" ]; then
        echo "SKIP: sub-${SUB_ID} (T1w_brain exists)"
        continue
    fi

    # Throttle
    while true; do
        n_running=$(squeue -u $USER -h | grep -c "anat_")
        if [ "$n_running" -lt "$MAX_RUNNING" ]; then
            break
        fi
        echo "  Waiting... ($n_running/$MAX_RUNNING jobs running)"
        sleep $POLL_INTERVAL
    done

    echo "Submitting job for sub-${SUB_ID}..."
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=anat_${SUB_ID}
#SBATCH --output=${LOG_DIR}/${SUB_ID}_%j.out
#SBATCH --error=${LOG_DIR}/${SUB_ID}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

export FSLDIR=/opt/fsl/6.0.3
. \${FSLDIR}/etc/fslconf/fsl.sh
export PATH=\${FSLDIR}/bin:\${PATH}

echo "Processing sub-${SUB_ID} on \$(hostname)"
python $REPO_DIR/A_preprocessing/register_mirror.py --sub $SUB_ID
EOT
    ((submitted++))
    echo "[$submitted] Submitted: anat_${SUB_ID}"
done

echo ""
echo "Done! Submitted $submitted jobs."
echo "Monitor with: squeue -u \$USER"