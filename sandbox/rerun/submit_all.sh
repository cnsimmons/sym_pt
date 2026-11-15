#!/bin/bash
# submit_all.sh: submits first levels, then each session's HighLevel step
# after all its runs finish. Skips any session whose loc_rerun outputs exist.
set -uo pipefail
L=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/logs
mkdir -p $L
n=0

# sub-001 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-001 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_001_01 --output=$L/rr1_001_01_%j.out --error=$L/rr1_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_001_01 --output=$L/rr1_001_01_%j.out --error=$L/rr1_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_001_01 --output=$L/rr1_001_01_%j.out --error=$L/rr1_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_001_01 --output=$L/rr1_001_01_%j.out --error=$L/rr1_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_001_01 --output=$L/rr1_001_01_%j.out --error=$L/rr1_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-05/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_001_01 --output=$L/rr1_001_01_%j.out --error=$L/rr1_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-001/ses-01/derivatives/fsl/loc_rerun/run-06/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_001_01 --output=$L/rr2_001_01_%j.out --error=$L/rr2_001_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/001_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-004 ses-06
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-004/ses-06/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-004/ses-06/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-004 ses-06: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_004_06 --output=$L/rr1_004_06_%j.out --error=$L/rr1_004_06_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-004/ses-06/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_004_06 --output=$L/rr1_004_06_%j.out --error=$L/rr1_004_06_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-004/ses-06/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_004_06 --output=$L/rr1_004_06_%j.out --error=$L/rr1_004_06_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-004/ses-06/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_004_06 --output=$L/rr2_004_06_%j.out --error=$L/rr2_004_06_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/004_06_session.sh || exit 1
  n=$((n+1))
fi
# sub-005 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-005 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_005_01 --output=$L/rr1_005_01_%j.out --error=$L/rr1_005_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_005_01 --output=$L/rr1_005_01_%j.out --error=$L/rr1_005_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_005_01 --output=$L/rr1_005_01_%j.out --error=$L/rr1_005_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_005_01 --output=$L/rr1_005_01_%j.out --error=$L/rr1_005_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_005_01 --output=$L/rr1_005_01_%j.out --error=$L/rr1_005_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-005/ses-01/derivatives/fsl/loc_rerun/run-05/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_005_01 --output=$L/rr2_005_01_%j.out --error=$L/rr2_005_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/005_01_session.sh || exit 1
  n=$((n+1))
fi
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
# sub-010 ses-03
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-010/ses-03/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-010/ses-03/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-010 ses-03: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_010_03 --output=$L/rr1_010_03_%j.out --error=$L/rr1_010_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-010/ses-03/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_010_03 --output=$L/rr1_010_03_%j.out --error=$L/rr1_010_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-010/ses-03/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_010_03 --output=$L/rr1_010_03_%j.out --error=$L/rr1_010_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-010/ses-03/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_010_03 --output=$L/rr2_010_03_%j.out --error=$L/rr2_010_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/010_03_session.sh || exit 1
  n=$((n+1))
fi
# sub-016 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-016 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_016_01 --output=$L/rr1_016_01_%j.out --error=$L/rr1_016_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_016_01 --output=$L/rr1_016_01_%j.out --error=$L/rr1_016_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_016_01 --output=$L/rr1_016_01_%j.out --error=$L/rr1_016_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_016_01 --output=$L/rr1_016_01_%j.out --error=$L/rr1_016_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_016_01 --output=$L/rr1_016_01_%j.out --error=$L/rr1_016_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-016/ses-01/derivatives/fsl/loc_rerun/run-05/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_016_01 --output=$L/rr2_016_01_%j.out --error=$L/rr2_016_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/016_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-018 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-018/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-018/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-018 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_018_02 --output=$L/rr1_018_02_%j.out --error=$L/rr1_018_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-018/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_018_02 --output=$L/rr1_018_02_%j.out --error=$L/rr1_018_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-018/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_018_02 --output=$L/rr1_018_02_%j.out --error=$L/rr1_018_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-018/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_018_02 --output=$L/rr1_018_02_%j.out --error=$L/rr1_018_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-018/ses-02/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_018_02 --output=$L/rr2_018_02_%j.out --error=$L/rr2_018_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/018_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-021 ses-03
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-021/ses-03/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-021/ses-03/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-021 ses-03: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_021_03 --output=$L/rr1_021_03_%j.out --error=$L/rr1_021_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-021/ses-03/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_021_03 --output=$L/rr1_021_03_%j.out --error=$L/rr1_021_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-021/ses-03/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_021_03 --output=$L/rr1_021_03_%j.out --error=$L/rr1_021_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-021/ses-03/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_021_03 --output=$L/rr2_021_03_%j.out --error=$L/rr2_021_03_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/021_03_session.sh || exit 1
  n=$((n+1))
fi
# sub-022 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-022/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-022/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-022 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_022_01 --output=$L/rr1_022_01_%j.out --error=$L/rr1_022_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-022/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_022_01 --output=$L/rr1_022_01_%j.out --error=$L/rr1_022_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-022/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_022_01 --output=$L/rr1_022_01_%j.out --error=$L/rr1_022_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-022/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_022_01 --output=$L/rr2_022_01_%j.out --error=$L/rr2_022_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/022_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-023 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-023/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-023/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-023 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_023_01 --output=$L/rr1_023_01_%j.out --error=$L/rr1_023_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-023/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_023_01 --output=$L/rr1_023_01_%j.out --error=$L/rr1_023_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-023/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_023_01 --output=$L/rr1_023_01_%j.out --error=$L/rr1_023_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-023/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_023_01 --output=$L/rr2_023_01_%j.out --error=$L/rr2_023_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/023_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-025 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-025/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-025/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-025 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_025_01 --output=$L/rr1_025_01_%j.out --error=$L/rr1_025_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-025/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_025_01 --output=$L/rr1_025_01_%j.out --error=$L/rr1_025_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-025/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_025_01 --output=$L/rr1_025_01_%j.out --error=$L/rr1_025_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-025/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_025_01 --output=$L/rr2_025_01_%j.out --error=$L/rr2_025_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/025_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-026 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-026/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-026/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-026 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_026_01 --output=$L/rr1_026_01_%j.out --error=$L/rr1_026_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-026/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_026_01 --output=$L/rr1_026_01_%j.out --error=$L/rr1_026_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-026/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_026_01 --output=$L/rr1_026_01_%j.out --error=$L/rr1_026_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-026/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_026_01 --output=$L/rr2_026_01_%j.out --error=$L/rr2_026_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/026_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-027 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-027/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-027/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-027 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_027_02 --output=$L/rr1_027_02_%j.out --error=$L/rr1_027_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-027/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_027_02 --output=$L/rr1_027_02_%j.out --error=$L/rr1_027_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-027/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_027_02 --output=$L/rr1_027_02_%j.out --error=$L/rr1_027_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-027/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_027_02 --output=$L/rr2_027_02_%j.out --error=$L/rr2_027_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/027_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-028 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-028/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-028/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-028 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_028_01 --output=$L/rr1_028_01_%j.out --error=$L/rr1_028_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-028/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_028_01 --output=$L/rr1_028_01_%j.out --error=$L/rr1_028_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-028/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_028_01 --output=$L/rr2_028_01_%j.out --error=$L/rr2_028_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/028_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-029 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-029/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-029/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-029 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_029_01 --output=$L/rr1_029_01_%j.out --error=$L/rr1_029_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-029/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_029_01 --output=$L/rr1_029_01_%j.out --error=$L/rr1_029_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-029/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_029_01 --output=$L/rr1_029_01_%j.out --error=$L/rr1_029_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-029/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_029_01 --output=$L/rr2_029_01_%j.out --error=$L/rr2_029_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/029_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-031 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-031/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-031/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-031 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_031_01 --output=$L/rr1_031_01_%j.out --error=$L/rr1_031_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-031/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_031_01 --output=$L/rr1_031_01_%j.out --error=$L/rr1_031_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-031/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_031_01 --output=$L/rr1_031_01_%j.out --error=$L/rr1_031_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-031/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_031_01 --output=$L/rr2_031_01_%j.out --error=$L/rr2_031_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/031_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-034 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-034/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-034/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-034 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_034_01 --output=$L/rr1_034_01_%j.out --error=$L/rr1_034_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-034/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_034_01 --output=$L/rr1_034_01_%j.out --error=$L/rr1_034_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-034/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_034_01 --output=$L/rr1_034_01_%j.out --error=$L/rr1_034_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-034/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_034_01 --output=$L/rr2_034_01_%j.out --error=$L/rr2_034_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/034_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-035 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-035/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-035/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-035 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_035_01 --output=$L/rr1_035_01_%j.out --error=$L/rr1_035_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-035/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_035_01 --output=$L/rr1_035_01_%j.out --error=$L/rr1_035_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-035/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_035_01 --output=$L/rr1_035_01_%j.out --error=$L/rr1_035_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-035/ses-01/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_035_01 --output=$L/rr2_035_01_%j.out --error=$L/rr2_035_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/035_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-036 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-036 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_036_01 --output=$L/rr1_036_01_%j.out --error=$L/rr1_036_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_036_01 --output=$L/rr1_036_01_%j.out --error=$L/rr1_036_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_036_01 --output=$L/rr1_036_01_%j.out --error=$L/rr1_036_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_036_01 --output=$L/rr1_036_01_%j.out --error=$L/rr1_036_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_036_01 --output=$L/rr1_036_01_%j.out --error=$L/rr1_036_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-036/ses-01/derivatives/fsl/loc_rerun/run-05/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_036_01 --output=$L/rr2_036_01_%j.out --error=$L/rr2_036_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/036_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-037 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-037 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_037_01 --output=$L/rr1_037_01_%j.out --error=$L/rr1_037_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_037_01 --output=$L/rr1_037_01_%j.out --error=$L/rr1_037_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_037_01 --output=$L/rr1_037_01_%j.out --error=$L/rr1_037_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_037_01 --output=$L/rr1_037_01_%j.out --error=$L/rr1_037_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/run-04/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_037_01 --output=$L/rr1_037_01_%j.out --error=$L/rr1_037_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-037/ses-01/derivatives/fsl/loc_rerun/run-05/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_037_01 --output=$L/rr2_037_01_%j.out --error=$L/rr2_037_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/037_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-038 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-038/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-038/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-038 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_038_02 --output=$L/rr1_038_02_%j.out --error=$L/rr1_038_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-038/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_038_02 --output=$L/rr1_038_02_%j.out --error=$L/rr1_038_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-038/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_038_02 --output=$L/rr1_038_02_%j.out --error=$L/rr1_038_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-038/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_038_02 --output=$L/rr2_038_02_%j.out --error=$L/rr2_038_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/038_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-039 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-039/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-039/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-039 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_039_01 --output=$L/rr1_039_01_%j.out --error=$L/rr1_039_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-039/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_039_01 --output=$L/rr1_039_01_%j.out --error=$L/rr1_039_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-039/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_039_01 --output=$L/rr1_039_01_%j.out --error=$L/rr1_039_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-039/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_039_01 --output=$L/rr2_039_01_%j.out --error=$L/rr2_039_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/039_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-040 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-040/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-040/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-040 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_040_01 --output=$L/rr1_040_01_%j.out --error=$L/rr1_040_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-040/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_040_01 --output=$L/rr1_040_01_%j.out --error=$L/rr1_040_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-040/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_040_01 --output=$L/rr1_040_01_%j.out --error=$L/rr1_040_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-040/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_040_01 --output=$L/rr2_040_01_%j.out --error=$L/rr2_040_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/040_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-041 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-041/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-041/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-041 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_041_01 --output=$L/rr1_041_01_%j.out --error=$L/rr1_041_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-041/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_041_01 --output=$L/rr1_041_01_%j.out --error=$L/rr1_041_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-041/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_041_01 --output=$L/rr1_041_01_%j.out --error=$L/rr1_041_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-041/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_041_01 --output=$L/rr2_041_01_%j.out --error=$L/rr2_041_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/041_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-043 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-043/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-043/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-043 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_043_01 --output=$L/rr1_043_01_%j.out --error=$L/rr1_043_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-043/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_043_01 --output=$L/rr1_043_01_%j.out --error=$L/rr1_043_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-043/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_043_01 --output=$L/rr1_043_01_%j.out --error=$L/rr1_043_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-043/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_043_01 --output=$L/rr2_043_01_%j.out --error=$L/rr2_043_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/043_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-044 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-044/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-044/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-044 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_044_01 --output=$L/rr1_044_01_%j.out --error=$L/rr1_044_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-044/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_044_01 --output=$L/rr1_044_01_%j.out --error=$L/rr1_044_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-044/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_044_01 --output=$L/rr1_044_01_%j.out --error=$L/rr1_044_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-044/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_044_01 --output=$L/rr2_044_01_%j.out --error=$L/rr2_044_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/044_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-052 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-052/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-052/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-052 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_052_01 --output=$L/rr1_052_01_%j.out --error=$L/rr1_052_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-052/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_052_01 --output=$L/rr1_052_01_%j.out --error=$L/rr1_052_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-052/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_052_01 --output=$L/rr2_052_01_%j.out --error=$L/rr2_052_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/052_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-054 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-054/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-054/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-054 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_054_01 --output=$L/rr1_054_01_%j.out --error=$L/rr1_054_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-054/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_054_01 --output=$L/rr1_054_01_%j.out --error=$L/rr1_054_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-054/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_054_01 --output=$L/rr1_054_01_%j.out --error=$L/rr1_054_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-054/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_054_01 --output=$L/rr2_054_01_%j.out --error=$L/rr2_054_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/054_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-057 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-057/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-057/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-057 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_057_02 --output=$L/rr1_057_02_%j.out --error=$L/rr1_057_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-057/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_057_02 --output=$L/rr1_057_02_%j.out --error=$L/rr1_057_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-057/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_057_02 --output=$L/rr1_057_02_%j.out --error=$L/rr1_057_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-057/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_057_02 --output=$L/rr2_057_02_%j.out --error=$L/rr2_057_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/057_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-058 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-058/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-058/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-058 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_058_01 --output=$L/rr1_058_01_%j.out --error=$L/rr1_058_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-058/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_058_01 --output=$L/rr1_058_01_%j.out --error=$L/rr1_058_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-058/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_058_01 --output=$L/rr1_058_01_%j.out --error=$L/rr1_058_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-058/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_058_01 --output=$L/rr2_058_01_%j.out --error=$L/rr2_058_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/058_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-059 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-059/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-059/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-059 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_059_02 --output=$L/rr1_059_02_%j.out --error=$L/rr1_059_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-059/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_059_02 --output=$L/rr1_059_02_%j.out --error=$L/rr1_059_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-059/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_059_02 --output=$L/rr1_059_02_%j.out --error=$L/rr1_059_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-059/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_059_02 --output=$L/rr2_059_02_%j.out --error=$L/rr2_059_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/059_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-062 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-062/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-062/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-062 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_062_01 --output=$L/rr1_062_01_%j.out --error=$L/rr1_062_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-062/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_062_01 --output=$L/rr1_062_01_%j.out --error=$L/rr1_062_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-062/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_062_01 --output=$L/rr1_062_01_%j.out --error=$L/rr1_062_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-062/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_062_01 --output=$L/rr2_062_01_%j.out --error=$L/rr2_062_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/062_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-064 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-064/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-064/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-064 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_064_01 --output=$L/rr1_064_01_%j.out --error=$L/rr1_064_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-064/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_064_01 --output=$L/rr1_064_01_%j.out --error=$L/rr1_064_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-064/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_064_01 --output=$L/rr1_064_01_%j.out --error=$L/rr1_064_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-064/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_064_01 --output=$L/rr2_064_01_%j.out --error=$L/rr2_064_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/064_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-065 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-065/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-065/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-065 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_065_01 --output=$L/rr1_065_01_%j.out --error=$L/rr1_065_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-065/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_065_01 --output=$L/rr1_065_01_%j.out --error=$L/rr1_065_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-065/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_065_01 --output=$L/rr1_065_01_%j.out --error=$L/rr1_065_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-065/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_065_01 --output=$L/rr2_065_01_%j.out --error=$L/rr2_065_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/065_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-066 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-066/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-066/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-066 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_066_01 --output=$L/rr1_066_01_%j.out --error=$L/rr1_066_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-066/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_066_01 --output=$L/rr1_066_01_%j.out --error=$L/rr1_066_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-066/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_066_01 --output=$L/rr1_066_01_%j.out --error=$L/rr1_066_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-066/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_066_01 --output=$L/rr2_066_01_%j.out --error=$L/rr2_066_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/066_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-067 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-067/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-067/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-067 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_067_02 --output=$L/rr1_067_02_%j.out --error=$L/rr1_067_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-067/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_067_02 --output=$L/rr1_067_02_%j.out --error=$L/rr1_067_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-067/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_067_02 --output=$L/rr2_067_02_%j.out --error=$L/rr2_067_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/067_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-068 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-068/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-068/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-068 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_068_02 --output=$L/rr1_068_02_%j.out --error=$L/rr1_068_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-068/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_068_02 --output=$L/rr1_068_02_%j.out --error=$L/rr1_068_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-068/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_068_02 --output=$L/rr1_068_02_%j.out --error=$L/rr1_068_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-068/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_068_02 --output=$L/rr2_068_02_%j.out --error=$L/rr2_068_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/068_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-069 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-069/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-069/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-069 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_069_01 --output=$L/rr1_069_01_%j.out --error=$L/rr1_069_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-069/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_069_01 --output=$L/rr1_069_01_%j.out --error=$L/rr1_069_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-069/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_069_01 --output=$L/rr1_069_01_%j.out --error=$L/rr1_069_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-069/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_069_01 --output=$L/rr2_069_01_%j.out --error=$L/rr2_069_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/069_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-071 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-071/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-071/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-071 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_071_01 --output=$L/rr1_071_01_%j.out --error=$L/rr1_071_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-071/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_071_01 --output=$L/rr1_071_01_%j.out --error=$L/rr1_071_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-071/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_071_01 --output=$L/rr2_071_01_%j.out --error=$L/rr2_071_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/071_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-074 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-074/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-074/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-074 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_074_01 --output=$L/rr1_074_01_%j.out --error=$L/rr1_074_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-074/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_074_01 --output=$L/rr1_074_01_%j.out --error=$L/rr1_074_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-074/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_074_01 --output=$L/rr1_074_01_%j.out --error=$L/rr1_074_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-074/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_074_01 --output=$L/rr2_074_01_%j.out --error=$L/rr2_074_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/074_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-075 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-075/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-075/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-075 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_075_01 --output=$L/rr1_075_01_%j.out --error=$L/rr1_075_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-075/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_075_01 --output=$L/rr1_075_01_%j.out --error=$L/rr1_075_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-075/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_075_01 --output=$L/rr1_075_01_%j.out --error=$L/rr1_075_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-075/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_075_01 --output=$L/rr2_075_01_%j.out --error=$L/rr2_075_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/075_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-076 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-076/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-076/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-076 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_076_01 --output=$L/rr1_076_01_%j.out --error=$L/rr1_076_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-076/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_076_01 --output=$L/rr1_076_01_%j.out --error=$L/rr1_076_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-076/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_076_01 --output=$L/rr1_076_01_%j.out --error=$L/rr1_076_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-076/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_076_01 --output=$L/rr2_076_01_%j.out --error=$L/rr2_076_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/076_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-077 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-077/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-077/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-077 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_077_01 --output=$L/rr1_077_01_%j.out --error=$L/rr1_077_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-077/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_077_01 --output=$L/rr1_077_01_%j.out --error=$L/rr1_077_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-077/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_077_01 --output=$L/rr1_077_01_%j.out --error=$L/rr1_077_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-077/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_077_01 --output=$L/rr2_077_01_%j.out --error=$L/rr2_077_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/077_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-078 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-078/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-078/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-078 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_078_01 --output=$L/rr1_078_01_%j.out --error=$L/rr1_078_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-078/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_078_01 --output=$L/rr1_078_01_%j.out --error=$L/rr1_078_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-078/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_078_01 --output=$L/rr1_078_01_%j.out --error=$L/rr1_078_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-078/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_078_01 --output=$L/rr2_078_01_%j.out --error=$L/rr2_078_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/078_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-079 ses-02
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-079/ses-02/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-079/ses-02/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-079 ses-02: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_079_02 --output=$L/rr1_079_02_%j.out --error=$L/rr1_079_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-079/ses-02/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_079_02 --output=$L/rr1_079_02_%j.out --error=$L/rr1_079_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-079/ses-02/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_079_02 --output=$L/rr1_079_02_%j.out --error=$L/rr1_079_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-079/ses-02/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_079_02 --output=$L/rr2_079_02_%j.out --error=$L/rr2_079_02_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/079_02_session.sh || exit 1
  n=$((n+1))
fi
# sub-082 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-082/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-082/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-082 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_082_01 --output=$L/rr1_082_01_%j.out --error=$L/rr1_082_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-082/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_082_01 --output=$L/rr1_082_01_%j.out --error=$L/rr1_082_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-082/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_082_01 --output=$L/rr1_082_01_%j.out --error=$L/rr1_082_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-082/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_082_01 --output=$L/rr2_082_01_%j.out --error=$L/rr2_082_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/082_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-083 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-083/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-083/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-083 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_083_01 --output=$L/rr1_083_01_%j.out --error=$L/rr1_083_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-083/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_083_01 --output=$L/rr1_083_01_%j.out --error=$L/rr1_083_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-083/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_083_01 --output=$L/rr1_083_01_%j.out --error=$L/rr1_083_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-083/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_083_01 --output=$L/rr2_083_01_%j.out --error=$L/rr2_083_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/083_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-084 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-084/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-084/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-084 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_084_01 --output=$L/rr1_084_01_%j.out --error=$L/rr1_084_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-084/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_084_01 --output=$L/rr1_084_01_%j.out --error=$L/rr1_084_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-084/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_084_01 --output=$L/rr1_084_01_%j.out --error=$L/rr1_084_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-084/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_084_01 --output=$L/rr2_084_01_%j.out --error=$L/rr2_084_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/084_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-085 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-085/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-085/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-085 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_085_01 --output=$L/rr1_085_01_%j.out --error=$L/rr1_085_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-085/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_085_01 --output=$L/rr1_085_01_%j.out --error=$L/rr1_085_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-085/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_085_01 --output=$L/rr1_085_01_%j.out --error=$L/rr1_085_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-085/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_085_01 --output=$L/rr2_085_01_%j.out --error=$L/rr2_085_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/085_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-087 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-087/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-087/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-087 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_087_01 --output=$L/rr1_087_01_%j.out --error=$L/rr1_087_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-087/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_087_01 --output=$L/rr1_087_01_%j.out --error=$L/rr1_087_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-087/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_087_01 --output=$L/rr1_087_01_%j.out --error=$L/rr1_087_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-087/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_087_01 --output=$L/rr2_087_01_%j.out --error=$L/rr2_087_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/087_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-088 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-088/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-088/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-088 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_088_01 --output=$L/rr1_088_01_%j.out --error=$L/rr1_088_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-088/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_088_01 --output=$L/rr1_088_01_%j.out --error=$L/rr1_088_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-088/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_088_01 --output=$L/rr1_088_01_%j.out --error=$L/rr1_088_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-088/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_088_01 --output=$L/rr2_088_01_%j.out --error=$L/rr2_088_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/088_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-089 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-089/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-089/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-089 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_089_01 --output=$L/rr1_089_01_%j.out --error=$L/rr1_089_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-089/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_089_01 --output=$L/rr1_089_01_%j.out --error=$L/rr1_089_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-089/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_089_01 --output=$L/rr1_089_01_%j.out --error=$L/rr1_089_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-089/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_089_01 --output=$L/rr2_089_01_%j.out --error=$L/rr2_089_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/089_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-090 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-090/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-090/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-090 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_090_01 --output=$L/rr1_090_01_%j.out --error=$L/rr1_090_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-090/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_090_01 --output=$L/rr1_090_01_%j.out --error=$L/rr1_090_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-090/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_090_01 --output=$L/rr1_090_01_%j.out --error=$L/rr1_090_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-090/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_090_01 --output=$L/rr2_090_01_%j.out --error=$L/rr2_090_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/090_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-091 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-091/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-091/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-091 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_091_01 --output=$L/rr1_091_01_%j.out --error=$L/rr1_091_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-091/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_091_01 --output=$L/rr1_091_01_%j.out --error=$L/rr1_091_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-091/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_091_01 --output=$L/rr1_091_01_%j.out --error=$L/rr1_091_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-091/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_091_01 --output=$L/rr2_091_01_%j.out --error=$L/rr2_091_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/091_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-092 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-092/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-092/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-092 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_092_01 --output=$L/rr1_092_01_%j.out --error=$L/rr1_092_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-092/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_092_01 --output=$L/rr1_092_01_%j.out --error=$L/rr1_092_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-092/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_092_01 --output=$L/rr1_092_01_%j.out --error=$L/rr1_092_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-092/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_092_01 --output=$L/rr2_092_01_%j.out --error=$L/rr2_092_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/092_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-093 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-093/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-093/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-093 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_093_01 --output=$L/rr1_093_01_%j.out --error=$L/rr1_093_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-093/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_093_01 --output=$L/rr1_093_01_%j.out --error=$L/rr1_093_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-093/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_093_01 --output=$L/rr1_093_01_%j.out --error=$L/rr1_093_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-093/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_093_01 --output=$L/rr2_093_01_%j.out --error=$L/rr2_093_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/093_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-094 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-094/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-094/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-094 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_094_01 --output=$L/rr1_094_01_%j.out --error=$L/rr1_094_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-094/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_094_01 --output=$L/rr1_094_01_%j.out --error=$L/rr1_094_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-094/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_094_01 --output=$L/rr1_094_01_%j.out --error=$L/rr1_094_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-094/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_094_01 --output=$L/rr2_094_01_%j.out --error=$L/rr2_094_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/094_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-095 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-095 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_095_01 --output=$L/rr1_095_01_%j.out --error=$L/rr1_095_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_095_01 --output=$L/rr1_095_01_%j.out --error=$L/rr1_095_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_095_01 --output=$L/rr1_095_01_%j.out --error=$L/rr1_095_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-095/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_095_01 --output=$L/rr2_095_01_%j.out --error=$L/rr2_095_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/095_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-096 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-096/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-096/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-096 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_096_01 --output=$L/rr1_096_01_%j.out --error=$L/rr1_096_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-096/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_096_01 --output=$L/rr1_096_01_%j.out --error=$L/rr1_096_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-096/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_096_01 --output=$L/rr1_096_01_%j.out --error=$L/rr1_096_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-096/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_096_01 --output=$L/rr2_096_01_%j.out --error=$L/rr2_096_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/096_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-097 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-097/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-097/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-097 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_097_01 --output=$L/rr1_097_01_%j.out --error=$L/rr1_097_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-097/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_097_01 --output=$L/rr1_097_01_%j.out --error=$L/rr1_097_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-097/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_097_01 --output=$L/rr1_097_01_%j.out --error=$L/rr1_097_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-097/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_097_01 --output=$L/rr2_097_01_%j.out --error=$L/rr2_097_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/097_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-098 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-098/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-098/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-098 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_098_01 --output=$L/rr1_098_01_%j.out --error=$L/rr1_098_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-098/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_098_01 --output=$L/rr1_098_01_%j.out --error=$L/rr1_098_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-098/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_098_01 --output=$L/rr1_098_01_%j.out --error=$L/rr1_098_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-098/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_098_01 --output=$L/rr2_098_01_%j.out --error=$L/rr2_098_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/098_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-099 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-099/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-099/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-099 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_099_01 --output=$L/rr1_099_01_%j.out --error=$L/rr1_099_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-099/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_099_01 --output=$L/rr1_099_01_%j.out --error=$L/rr1_099_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-099/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_099_01 --output=$L/rr1_099_01_%j.out --error=$L/rr1_099_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-099/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_099_01 --output=$L/rr2_099_01_%j.out --error=$L/rr2_099_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/099_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-101 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-101/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-101/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-101 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_101_01 --output=$L/rr1_101_01_%j.out --error=$L/rr1_101_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-101/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_101_01 --output=$L/rr1_101_01_%j.out --error=$L/rr1_101_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-101/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_101_01 --output=$L/rr1_101_01_%j.out --error=$L/rr1_101_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-101/ses-01/derivatives/fsl/loc_rerun/run-03/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_101_01 --output=$L/rr2_101_01_%j.out --error=$L/rr2_101_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/101_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-108 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-108/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-108/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-108 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_108_01 --output=$L/rr1_108_01_%j.out --error=$L/rr1_108_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-108/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_108_01 --output=$L/rr1_108_01_%j.out --error=$L/rr1_108_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-108/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_108_01 --output=$L/rr2_108_01_%j.out --error=$L/rr2_108_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/108_01_session.sh || exit 1
  n=$((n+1))
fi
# sub-109 ses-01
if [ -e /lab_data/behrmannlab/claire/sym_pt_rerun/sub-109/ses-01/derivatives/fsl/loc_rerun/HighLevel.gfeat ] || ls -d /lab_data/behrmannlab/claire/sym_pt_rerun/sub-109/ses-01/derivatives/fsl/loc_rerun/run-*/1stLevel*.feat >/dev/null 2>&1; then
  echo "skip sub-109 ses-01: outputs exist"
else
  ids=""
  j=$(sbatch --parsable --job-name=rr1_109_01 --output=$L/rr1_109_01_%j.out --error=$L/rr1_109_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-109/ses-01/derivatives/fsl/loc_rerun/run-01/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  j=$(sbatch --parsable --job-name=rr1_109_01 --output=$L/rr1_109_01_%j.out --error=$L/rr1_109_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 --wrap="export FSLDIR=/opt/fsl/6.0.3; . \$FSLDIR/etc/fslconf/fsl.sh; export PATH=\$FSLDIR/bin:\$PATH; feat /lab_data/behrmannlab/claire/sym_pt_rerun/sub-109/ses-01/derivatives/fsl/loc_rerun/run-02/1stLevel.fsf") || exit 1; j=${j%%;*}; ids="$ids:$j"
  sbatch --parsable --dependency=afterok$ids --job-name=rr2_109_01 --output=$L/rr2_109_01_%j.out --error=$L/rr2_109_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/109_01_session.sh || exit 1
  n=$((n+1))
fi

echo "$n sessions submitted"
