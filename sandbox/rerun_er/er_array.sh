#!/bin/bash
#SBATCH --job-name=er
#SBATCH --array=0-55%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_er/logs/er_%a.out
#SBATCH --error=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_er/logs/er_%a.err
source /home/csimmon2/anaconda3/etc/profile.d/conda.sh && conda activate fmri
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTENSOR_FLAGS="compiledir=/tmp/pytensor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
python /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_er.py run $SLURM_ARRAY_TASK_ID
rm -rf /tmp/pytensor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
