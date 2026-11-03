LONGITUDINAL PIPELINE - new sessions of existing patients                 2026-11-03
pipeline/longitudinal/: new_session.sh (steps 2-3), session_steps.py (helper),
extract.py (step 4), 09_longitudinal.ipynb (step 5), this file
================================================================================

WHY THIS FOLDER EXISTS
--------------------------------------------------------------------------------
The 1_preprocessing/ scripts were written for the whole cohort. Run as the old run
order runs them, most act on EVERY session in sub_info.csv (read 2026-11-03):
  01_organize          makes folders for every session
  02_extract_confounds every session; skips runs that already have a spikes file
  03_convert_timing    REWRITES every session's timing files
  04_register_mirror   --sub: every session of that subject; skips existing outputs
  05_create_fsf        DELETES and REWRITES every session's 1stLevel.fsf
  06, 11 (submit)      submit any unfinished run / session anywhere in $S
  14, 15 (searchmasks) rewrite FIRST-session masks; never needed for a new session
Already session-scoped: 00a (--sub --ses), 07 (sub-XXX NN), 10 (with XXX-NN),
12 and 13 (sub-XXX NN).

new_session.sh applies the same step logic to ONE session. The Python steps
import and call the original functions (01, 03, 04); the bash steps are the
original per-session code copied unchanged (02, 05, 10). Nothing in
1_preprocessing/ was changed. Tested 2026-11-03 on a mock of the cluster paths
(FSL and SLURM replaced by stand-ins): for sub-101 ses-02, prep wrote the same 35
files as running the original 01-05, with identical 1stLevel.fsf, timing and spikes
content; HighLevel.fsf matched 10's; 'validate' matched files the originals wrote
for another session and caught a deliberate one-line change.

For longitudinal work, do NOT run 01-06, 11, 14 or 15 directly.


SETUP (every new shell, on the cluster, inside screen)
--------------------------------------------------------------------------------
  cd /user_data/csimmon2/git_repos/sym_pt/pipeline && conda activate fmri
  export FSLDIR=/opt/fsl/6.0.3; source $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
The pandas "bottleneck" warning printed by every Python step is harmless.


STAGE 0 - GET THE DATA ONTO THE CLUSTER (manual)
--------------------------------------------------------------------------------
1. Flywheel, Mac browser: tick the session (top-level row) -> Actions -> Download;
   in the dialog tick only "nifti" and "source code" (the .json sidecars). One ZIP.
   Check the nifti count matches the session before downloading. Flywheel's
   session label can be wrong (sub-101's second visit was labeled ses-01_1); the
   files get our ses-NN names below, and a README note records the original names.
2. Timing: the *loc.mat files come from the stimulus PC, not Flywheel. Only
   *loc.mat is needed (not .edf eye tracking, loc_*, loc_blocktimes_*).
3. Mac terminal: put the localizer .nii.gz/.json, the T1 .nii.gz/.json and the
   *loc.mat files in one folder, cd into it (no cd = runs in ~ and matches
   nothing), set SUB/SES, then:
     SUB=101; SES=02; B=sub-${SUB}_ses-${SES}
     mkdir -p ses-$SES/anat ses-$SES/func ses-$SES/covs
     for f in *_func_bold_task_localizer_run_*.nii.gz; do r=$(echo "$f" | sed -E 's/.*_task_localizer_run_([0-9]+)_.*/\1/'); mv "$f" ses-$SES/func/${B}_task-loc_run-${r}_bold.nii.gz; mv "${f%.nii.gz}.json" ses-$SES/func/${B}_task-loc_run-${r}_bold.json; done
     mv *_anat_T1w_acq_mprage_*.nii.gz ses-$SES/anat/${B}_T1w.nii.gz; mv *_anat_T1w_acq_mprage_*.json ses-$SES/anat/${B}_T1w.json
     mv *loc.mat ses-$SES/covs/
     find ses-$SES -type f | sort
     scp -O -r ses-$SES csimmon2@mind.cs.cmu.edu:/lab_data/behrmannlab/hemi/Raw/sub-$SUB/
   Add a README_relabeled note in ses-$SES first if the Flywheel label was wrong.
   scp needs -O: the cluster's older SSH server cannot create new folders in the
   Mac's default (SFTP) scp mode ("path canonicalization failed").
4. sub_info.csv row and EXCLUDE_SES (5 files; run order v5, Stage 0) - still
   needed until the separate longitudinal session list exists (see NOT YET).


STAGES:  bash longitudinal/new_session.sh XXX NN <stage>
--------------------------------------------------------------------------------
  stage0     raw checks (vols 184, TR 2, slices vs first session); .mat checks
             (15 blocks, first block 8.0 s after time0); 00a .mat -> events.tsv
  prep       01 dirs, 02 spikes, 03 timing, 04 anat, 05 1stLevel.fsf, then submits
             FEAT for this session's runs only. Starts the file-guard window.
  qc         after FEAT: checkpoint B (motion, flag > 0.5 mm) and B2 (run-to-run
             agreement) with the verdict: keep FEAT, or run step07
  step07     the committed 07 script; prints the keep/restore rule
  highlevel  HighLevel.fsf (10's logic) and submits HighLevel; refuses if
             HighLevel.gfeat already exists (re-registration is a separate procedure)
  finish     12 symlinks, 13 MNI copies (stops if 13 reports a failure),
             checkpoint C (z range, similarity to first session), file guard
  guard      the file guard alone
  validate   read-only: rebuilds an EXISTING session's 1stLevel.fsf, HighLevel.fsf
             and timing files in /tmp and compares them with what is on disk. Run it
             on an existing session (e.g. 092 02) before trusting the driver on a
             new one, and again whenever a 1_preprocessing script or template changes.
Each stage prints the NEXT command. Rerunning a stage is safe; it only touches
this session.


WHAT EACH STAGE WRITES
--------------------------------------------------------------------------------
  stage0     Raw/sub-XXX/ses-NN/func/*_events.tsv
  prep       $S/sub-XXX/ses-NN/{timing, anat, derivatives/...}; FEAT jobs
             (logs: /user_data/csimmon2/git_repos/sym_pt/logs_feat);
             ~/stage1_start_sub-XXX_ses-NN (guard mark)
  step07     this session's run-*/1stLevel.feat/reg (original kept as reg.bak_DATE)
  highlevel  this session's HighLevel.fsf; HighLevel job (logs: pipeline/logs_highlevel)
  finish     symlinks and *_mni copies inside this session's HighLevel.gfeat
  qc, guard, validate   nothing outside /tmp
  Only possible write elsewhere: 04's ROI staging copies MISSING files into $S/rois.
  The guard would list it.


RULES (deviation log v6)
--------------------------------------------------------------------------------
- Registration: FEAT first. qc's B2: every pair <= 5 mm keep FEAT; any pair > 5 mm
  run step07, keep it only if its QC lines are <= ~1 mm, else restore FEAT and
  flag the session as a registration failure.
- The longitudinal and cross-sectional studies are kept separate; nothing here
  writes cross-sectional outputs. Sessions in BOTH studies (sub-008 ses-02,
  sub-010 ses-03) have .longitudinal copies; do not run this driver on them.
- The driver is for LATER sessions; it refuses a subject's first session.
- Guard window: starts the first time prep runs. To start a new window, delete
  ~/stage1_start_sub-XXX_ses-NN before prep.


STEP 4 - EXTRACTION:  longitudinal/extract.py   (inside screen)
--------------------------------------------------------------------------------
Uses 2_extraction/01-04's own code, unchanged, and writes only to
pipeline/outputs_longitudinal/ (univariate_v1.csv, rsa_v1.csv,
liu_exact_replication_v2.csv, peak_coords_mni.csv). Swapped in: output paths;
which sessions run (each session's rows depend only on that session's files);
and maps - a session with HighLevel.gfeat.longitudinal (sub-008 ses-02, sub-010
ses-03; found automatically) is read from that folder's real zstat1/cope1 files,
because its *_ses01 links point back into HighLevel.gfeat (step 12 makes
absolute links; the folder was renamed on 2026-11-02).

First time only (done 2026-11-03):
  python longitudinal/extract.py seed
      copies what already exists: univariate/rsa from outputs/rereg_20261102
      (extracted 2026-11-02 with the corrected sub-008/010 maps), liu and peak MNI
      from the current cross-sectional files
  python longitudinal/extract.py add 008 02 010 03 092 02 108 02 101 02
      univariate/rsa should report "identical" for the four sessions already in
      the seed (this checks the .longitudinal reading on real data) and "new" for
      101 02; liu/peak MNI "changed" for 008 02 and 010 03, "new" for the others
      (03's file had never been rebuilt for them - why sub-092 had no peak
      locations)
Each new session afterwards (minutes):
  python longitudinal/extract.py add XXX NN
Checks:
  python longitudinal/extract.py validate     # read-only, seconds
      univariate, rsa : identical to outputs/rereg_20261102 for every session there;
                        later sessions appear only in the new files
      liu             : its univariate and RSA columns equal univariate_v1/rsa_v1
                        (03 computes them with the same code as 01 and 02)
      peak MNI        : one converted row per liu row with a native peak
  python ../check_outputs_v2.py --ignore sub-092:2 sub-108:2 | diff outputs/check_restored_20261102.txt -
      # cross-sectional outputs untouched: expect no output
  python longitudinal/extract.py full [uni rsa liu mni]   # rebuild from scratch, if ever needed
Done 2026-11-03: seed; add 008 02 010 03 092 02 108 02 101 02 -> univariate/rsa
identical to 2026-11-02 for the four existing sessions, sub-101 ses-02 new; liu/peak
MNI changed for 008 02, 010 03, 108 02 (108's old rows predated its 2026-10-31
step-07 fix), new for 092 02 and 101 02; cross-sectional check: no output.
Tested 2026-11-03 on a mock with the same rename and absolute links: seed + add
equaled a full rebuild for every session in all four files; the original 01 run
after the rename gave different values for the renamed session.


STEP 5 - LONGITUDINAL NOTEBOOK:  longitudinal/09_longitudinal.ipynb
--------------------------------------------------------------------------------
Copy of 4_analyses/09_longitudinal.ipynb (the original is unchanged) with only the
paths changed: it reads and writes pipeline/outputs_longitudinal/ (univariate, rsa,
peak MNI in; longitudinal_q12_results.csv, otc_rsm_persession.csv, otc_rsm_long/
masks, figures/ out), and its whole-OTC maps come from HighLevel.gfeat.longitudinal
where it exists. Open it in Jupyter as usual, kernel brainiak_env, Restart & Run All.
Run it after step 4 for each new session.


NOT YET (handoff, next steps)
--------------------------------------------------------------------------------
- A separate longitudinal session list, so new sessions no longer need a
  sub_info.csv row or EXCLUDE_SES edits; then move the 092, 108, 101 ses-02 rows
  there (guard: check_outputs must still match check_restored_20261102.txt).
- Stage 0 on the Mac as one script (rename, note, scp).
