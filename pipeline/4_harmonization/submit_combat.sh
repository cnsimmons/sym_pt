#!/bin/bash
#SBATCH --job-name=combat_tfce
#SBATCH --output=/user_data/csimmon2/git_repos/sym_pt/logs_combat/combat_tfce_%j.out
#SBATCH --error=/user_data/csimmon2/git_repos/sym_pt/logs_combat/combat_tfce_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

mkdir -p /user_data/csimmon2/git_repos/sym_pt/logs_combat

# FSL (needed for randomise)
export FSLDIR=/opt/fsl/6.0.3
. ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fmri

python /user_data/csimmon2/git_repos/sym_pt/F_harmonization/combat_03_tfce_harmonized.py