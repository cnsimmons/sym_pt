#!/bin/bash
# HighLevel step for one session; runs after its first levels.
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
set -euo pipefail
D=/lab_data/behrmannlab/claire/sym_pt_rerun/sub-025/ses-01/derivatives/fsl/loc_rerun

# 1. copy the record registration into each new first level (no overwrite)
[ -f $D/run-01/1stLevel.feat/stats/cope20.nii.gz ] || { echo "first level incomplete: run-01"; exit 1; }
if [ -e $D/run-01/1stLevel.feat/reg ]; then rmdir $D/run-01/1stLevel.feat/reg || { echo "reg exists: run-01"; exit 1; }; fi
cp -a /user_data/csimmon2/sym_pt/sub-025/ses-01/derivatives/fsl/loc/run-01/1stLevel.feat/reg $D/run-01/1stLevel.feat/reg
[ -f $D/run-02/1stLevel.feat/stats/cope20.nii.gz ] || { echo "first level incomplete: run-02"; exit 1; }
if [ -e $D/run-02/1stLevel.feat/reg ]; then rmdir $D/run-02/1stLevel.feat/reg || { echo "reg exists: run-02"; exit 1; }; fi
cp -a /user_data/csimmon2/sym_pt/sub-025/ses-01/derivatives/fsl/loc/run-02/1stLevel.feat/reg $D/run-02/1stLevel.feat/reg
[ -f $D/run-03/1stLevel.feat/stats/cope20.nii.gz ] || { echo "first level incomplete: run-03"; exit 1; }
if [ -e $D/run-03/1stLevel.feat/reg ]; then rmdir $D/run-03/1stLevel.feat/reg || { echo "reg exists: run-03"; exit 1; }; fi
cp -a /user_data/csimmon2/sym_pt/sub-025/ses-01/derivatives/fsl/loc/run-03/1stLevel.feat/reg $D/run-03/1stLevel.feat/reg

# 2. HighLevel (20 copes)
[ ! -e $D/HighLevel.gfeat ] || { echo "HighLevel.gfeat exists"; exit 1; }
feat $D/HighLevel.fsf

touch $D/RERUN_DONE
echo "sub-025 ses-01 done"
