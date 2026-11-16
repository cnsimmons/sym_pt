#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_grid/logs/grid_%a.out
#SBATCH --error=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_grid/logs/grid_%a.err
source /home/csimmon2/anaconda3/etc/profile.d/conda.sh && conda activate fmri
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_grid.py run $SLURM_ARRAY_TASK_ID
