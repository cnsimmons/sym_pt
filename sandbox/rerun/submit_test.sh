#!/bin/bash
# submit_test.sh: submits first levels, then each session's HighLevel step
# after all its runs finish. Skips any session whose loc_rerun outputs exist.
set -uo pipefail
L=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/logs
mkdir -p $L
n=0

# sub-008 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-008/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-008/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-008 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_008_02 --output=$L/rr1_008_02_%j.out --error=$L/rr1_008_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-008/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_008_02 --output=$L/rr1_008_02_%j.out --error=$L/rr1_008_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-008/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_008_02 --output=$L/rr1_008_02_%j.out --error=$L/rr1_008_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-008/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_008_02 --output=$L/rr1_008_02_%j.out --error=$L/rr1_008_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-008/ses-02/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_008_02 --output=$L/rr2_008_02_%j.out --error=$L/rr2_008_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/008_02_session.sh || exit 1
  n=$((n+1))
fi

echo "$n sessions submitted"
