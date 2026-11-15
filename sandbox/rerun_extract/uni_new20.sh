#!/bin/bash
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
set -euo pipefail
/home/csimmon2/anaconda3/envs/fmri/bin/python /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract.py --root /lab_data/behrmannlab/claire job uni_new20 
