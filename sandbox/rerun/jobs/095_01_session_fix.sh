#!/bin/bash
# HighLevel for a session with one failed first level (see RERUN_EXCEPTION.txt).
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
set -euo pipefail
D=/lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun

[ -f /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/stats/cope20.nii.gz ] || { echo "first level incomplete: /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat"; exit 1; }
if [ -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/reg ] && [ -z "$(ls -A /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/reg)" ]; then rmdir /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/reg; fi
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/reg ]; then diff -rq /user_data/csimmon2/sym_pt/sub-095/ses-01/derivatives/fsl/loc/run-02/1stLevel.feat/reg /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/reg >/dev/null || { echo "reg differs from record: /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat"; exit 1; }; else cp -a /user_data/csimmon2/sym_pt/sub-095/ses-01/derivatives/fsl/loc/run-02/1stLevel.feat/reg /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.feat/reg; fi
[ -f /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/stats/cope20.nii.gz ] || { echo "first level incomplete: /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat"; exit 1; }
if [ -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/reg ] && [ -z "$(ls -A /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/reg)" ]; then rmdir /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/reg; fi
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/reg ]; then diff -rq /user_data/csimmon2/sym_pt/sub-095/ses-01/derivatives/fsl/loc/run-03/1stLevel.feat/reg /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/reg >/dev/null || { echo "reg differs from record: /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat"; exit 1; }; else cp -a /user_data/csimmon2/sym_pt/sub-095/ses-01/derivatives/fsl/loc/run-03/1stLevel.feat/reg /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.feat/reg; fi

[ ! -e $D/HighLevel.gfeat ] || { echo "HighLevel.gfeat exists"; exit 1; }
feat $D/HighLevel_fix.fsf

touch $D/RERUN_DONE
echo "sub-095 ses-01 done"
