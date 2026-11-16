#!/bin/bash
#SBATCH --job-name=pkwta
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_more/logs/pkwta.out
#SBATCH --error=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_more/logs/pkwta.err
source /home/csimmon2/anaconda3/etc/profile.d/conda.sh && conda activate fmri
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
python /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_peaks_wta.py all
