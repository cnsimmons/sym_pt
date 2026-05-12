#!/bin/bash
# Usage: bash submit_sensitivity_analysis.sh
# Runs Tests 1-3 perm tests + Test 5 (WTA composition of TFCE clusters).
# Output: $processed_dir/group_results/sensitivity_liu_overlap/

REPO_DIR="/user_data/csimmon2/git_repos/sym_pt"
LOG_DIR="$REPO_DIR/logs_sensitivity_analysis"
mkdir -p $LOG_DIR

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=sens_perm
#SBATCH --output=${LOG_DIR}/sens_%j.out
#SBATCH --error=${LOG_DIR}/sens_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

echo "Running on \$(hostname)"

python $REPO_DIR/D_liu/sensitivity_analysis.py
EOT

echo "Submitted. Monitor: squeue -u \$USER"ls /user_data/csimmon2/sym_pt/group_results/tfce_votc/*/rand_tfce_corrp_tstat1.nii.gz 2>/dev/null | wc -l
