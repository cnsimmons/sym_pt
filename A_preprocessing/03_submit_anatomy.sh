#!/bin/bash
# submit_anatomy.sh - Submit anatomy jobs with throttling
# Usage: bash submit_anatomy.sh

# 1. Define Paths
REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
PROCESSED_DIR="/user_data/csimmon2/sym_pt"
LOG_DIR="$REPO_DIR/logs_anatomy"
MAX_RUNNING=12
POLL_INTERVAL=60  # seconds between queue checks

# 2. Setup Logs
mkdir -p $LOG_DIR
echo "Logs will be saved to: $LOG_DIR"
echo "Max concurrent jobs: $MAX_RUNNING"
echo ""

# 3. Find subjects
cd $PROCESSED_DIR
SUBJECTS=$(ls -d sub-*)

submitted=0

# 4. Loop and Submit with throttling
for SUB_DIR in $SUBJECTS; do
    SUB_ID=${SUB_DIR#sub-}

    # Throttle: wait until running jobs drop below limit
    while true; do
        n_running=$(squeue -u $USER -h | grep -c "anat_")
        if [ "$n_running" -lt "$MAX_RUNNING" ]; then
            break
        fi
        echo "  Waiting... ($n_running/$MAX_RUNNING jobs running)"
        sleep $POLL_INTERVAL
    done

    echo "Submitting job for $SUB_ID..."

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

# --- LOAD FSL (Crucial on Cluster Nodes) ---
export FSLDIR=/opt/fsl/6.0.3
. \${FSLDIR}/etc/fslconf/fsl.sh
export PATH=\${FSLDIR}/bin:\${PATH}

# --- RUN PYTHON SCRIPT ---
echo "Processing $SUB_ID on \$(hostname)"
python $REPO_DIR/register_mirror.py --sub $SUB_ID
EOT

    ((submitted++))
    echo "[$submitted] Submitted: anat_${SUB_ID}"

done

echo ""
echo "Done! Submitted $submitted jobs."
echo "Monitor with: squeue -u \$USER"