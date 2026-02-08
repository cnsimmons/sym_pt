#!/bin/bash
# submit_anatomy.sh
# Usage: bash submit_anatomy.sh

# 1. Define Paths
REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
PROCESSED_DIR="/user_data/csimmon2/sym_pt"
LOG_DIR="$REPO_DIR/logs_anatomy"

# 2. Setup Logs
mkdir -p $LOG_DIR
echo "Logs will be saved to: $LOG_DIR"

# 3. Find subjects
# We look for folders like 'sub-022' in the processed directory
cd $PROCESSED_DIR
SUBJECTS=$(ls -d sub-*)

# 4. Loop and Submit
for SUB_DIR in $SUBJECTS; do
    # Extract ID (e.g., "sub-022" -> "022")
    SUB_ID=${SUB_DIR#sub-}
    
    echo "Submitting job for $SUB_ID..."
    
    # The HEREDOC below creates the job script on the fly
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

done
