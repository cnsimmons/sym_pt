#!/bin/bash
# One first-level FEAT run: line $SLURM_ARRAY_TASK_ID of first_levels.txt.
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
set -euo pipefail
fsf=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/first_levels.txt)
if [ -e "${fsf%.fsf}.feat" ]; then echo "skip, already run: ${fsf%.fsf}.feat"; exit 0; fi
feat "$fsf"
