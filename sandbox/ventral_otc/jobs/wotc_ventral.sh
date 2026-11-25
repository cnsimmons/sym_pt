#!/bin/bash
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
set -euo pipefail
/home/csimmon2/anaconda3/envs/fmri/bin/python /user_data/csimmon2/git_repos/sym_pt/sandbox/extract_ventral.py job wotc_ventral
