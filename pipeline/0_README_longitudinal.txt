LONGITUDINAL PIPELINE - ROAD MAP AND README                            2026-11-05
pipeline/ (to become the longitudinal study's repository)
================================================================================
For every new session of an existing patient, follow the ROAD MAP top to
bottom. The sections after it explain each step.

Where things live:
  pipeline/       all code, settings, logs, records and outputs (this repository)
  $S, $RAW        the imaging data: /user_data/csimmon2/sym_pt and
                  /lab_data/behrmannlab/hemi/Raw (too large for a repository)
  ~/regcheck/     registration-check pictures and keys, copied to the Mac to view
Nothing here writes the cross-sectional study's files in pipeline/ (outputs/,
3_harmonization/, 4_analyses/, 5_figures/).
Files: longitudinal/new_session.sh (road map 3-8), longitudinal/session_steps.py
(its helper), 1_preprocessing/07 (run by step07), longitudinal/extract.py (9),
longitudinal/09_longitudinal.ipynb (11), this file.
Replaces longitudinal/README_longitudinal.txt and, for longitudinal work,
new_sessions_run_order_2026-11-02_v5.txt.


SETUP (every new shell, on the cluster, inside screen)
--------------------------------------------------------------------------------
  cd /user_data/csimmon2/git_repos/sym_pt/pipeline && conda activate fmri
  export FSLDIR=/opt/fsl/6.0.3; source $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
The pandas "bottleneck" warning printed by every Python step is harmless.


ROAD MAP  (XXX = subject number, NN = the new session number; run from pipeline/)
--------------------------------------------------------------------------------
 1. Data onto the cluster: STAGE 0 below.
 2. Add the session's row to sub_info.csv (the file must end in a newline), and
    add (sub-XXX, NN) to EXCLUDE_SES in these five files so the cross-sectional
    study ignores the session (not in longitudinal/09_longitudinal.ipynb):
      3_harmonization/lib_stats.py             4_analyses/02_stats_harmony.py
      4_analyses/08_bf_patient_vs_control.py   5_figures/01_final_figures.ipynb
      5_figures/02_whole_otc_bayes.ipynb
 3. bash longitudinal/new_session.sh XXX NN stage0
      check  every run vols=184 and TR=2; slices as in the first session
             (69 = whole brain; 27 = ROI and OTC analyses only). Take a short
             run out of func/ before prep.
 4. bash longitudinal/new_session.sh XXX NN prep
      wait   until 'squeue -u csimmon2' shows no feat_XXX_NN_ jobs (hours)
 5. bash longitudinal/new_session.sh XXX NN qc
      check  motion: a run over 0.5 mm is flagged
 6. bash longitudinal/new_session.sh XXX NN step07
      check  the rule line (FEAT or v2 for each run) and the baseline line
      rate   if pictures were written: copy that folder from ~/regcheck/ to the
             Mac, rate every run blind (RULES), then open its KEY file. If your
             choice and the rule disagree for any run, stop and record it.
 7. bash longitudinal/new_session.sh XXX NN highlevel
      wait   until squeue shows no hlvl_XXX_NN job
 8. bash longitudinal/new_session.sh XXX NN finish
      check  checkpoint C (any |z| > 20? compare with the first session) and the
             file guard: "none - OK"
 9. python longitudinal/extract.py add XXX NN
    python longitudinal/extract.py validate
      check  the new session is reported "new" in all four files (see OPEN)
10. python ../check_outputs_v2.py --ignore sub-092:2 sub-108:2 | diff outputs/check_restored_20261102.txt -
      check  no output: the cross-sectional outputs are untouched
11. longitudinal/09_longitudinal.ipynb: kernel brainiak_env, Restart & Run All
12. Record and commit: a dated entry in the deviation log (anything flagged,
    step07's rule per run, your ratings), then git commit.


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
4. Then road map step 2 (sub_info.csv row and EXCLUDE_SES) - still needed until
   the separate longitudinal session list exists (see NOT YET).


STAGES:  bash longitudinal/new_session.sh XXX NN <stage>
--------------------------------------------------------------------------------
  stage0     raw checks (vols 184, TR 2, slices vs first session); .mat checks
             (15 blocks, first block 8.0 s after time0); 00a .mat -> events.tsv
  prep       01 dirs, 02 spikes, 03 timing, 04 anat, 05 1stLevel.fsf, then submits
             FEAT for this session's runs only. Starts the file-guard window.
  qc         after FEAT: checkpoint B (motion, flag > 0.5 mm) and B2 (run-to-run
             agreement of FEAT's runs, information only)
  step07     always: step 07 v3 scores FEAT and v2 for each run with FEAT's BBR cost
             and keeps v2 only where its cost is lower by more than 0.01; where the
             two differ it writes pictures with hidden A/B labels - rate them before
             highlevel (RULES)
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
             (logs: pipeline/logs_feat); guard mark in
             pipeline/longitudinal/qc/guard_marks/
  step07     this session's run-*/1stLevel.feat/reg, only for runs that get v2
             (FEAT's original kept as reg.bak_DATE); its log in
             pipeline/longitudinal/qc/logs/; pictures and keys in ~/regcheck/
  highlevel  this session's HighLevel.fsf; HighLevel job (logs: pipeline/logs_highlevel)
  finish     symlinks and *_mni copies inside this session's HighLevel.gfeat
  qc, guard, validate   nothing outside /tmp
  Only possible write elsewhere: 04's ROI staging copies MISSING files into $S/rois.
  The guard would list it.


RULES (deviation log v7, section 15)
--------------------------------------------------------------------------------
- Registration: every run of a later session is registered two ways - FEAT, and
  v2 (FEAT's own BBR started from an alignment to the first session's functional
  scan) - and both are scored with FEAT's BBR cost. A run gets v2 only if its
  cost is lower by more than 0.01; otherwise it keeps FEAT. The 5 mm rule is
  retired. step07 also prints the first session's own costs as the baseline.
- Where the two costs differ, rate step07's pictures before highlevel (Greve &
  Fischl 2009): for each run, which red line follows the edge of the bright
  gray-matter patches more closely, A or B, and is it acceptable next to the
  reference pictures? Then open the key. If your choice and the rule disagree,
  stop and log it.
- The longitudinal and cross-sectional studies are kept separate; nothing here
  writes cross-sectional outputs. Sessions in BOTH studies (sub-008 ses-02,
  sub-010 ses-03) have .longitudinal copies; this driver refuses them, and step
  07 writes them only to reg.longitudinal. Step 07 also refuses any session that
  already has a HighLevel unless it is listed in its LONG_ONLY (092, 108, 101
  ses-02).
- The driver is for LATER sessions; it refuses a subject's first session.
- Guard window: starts the first time prep runs. To start a new window, delete
  longitudinal/qc/guard_marks/stage1_start_sub-XXX_ses-NN before prep.


EXTRACTION (road map 9):  longitudinal/extract.py   (inside screen)
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


NOTEBOOK (road map 11):  longitudinal/09_longitudinal.ipynb
--------------------------------------------------------------------------------
Copy of 4_analyses/09_longitudinal.ipynb (the original is unchanged) with only the
paths changed: it reads and writes pipeline/outputs_longitudinal/ (univariate, rsa,
peak MNI in; longitudinal_q12_results.csv, otc_rsm_persession.csv, otc_rsm_long/
masks, figures/ out), and its whole-OTC maps come from HighLevel.gfeat.longitudinal
where it exists. Open it in Jupyter as usual, kernel brainiak_env, Restart & Run All.
Run it after road map 9-10 for each new session.


OPEN (2026-11-05)
--------------------------------------------------------------------------------
- sub-010 and sub-108: Marlene advised dropping them from the longitudinal study,
  as their provenance cannot be confirmed. For now they are processed like every
  session; whether group analyses drop them is decided later (C.S.).
- extract.py validate compares univariate/rsa with outputs/rereg_20261102 (the
  2026-11-02 extraction), so every session re-registered since then will show
  "differ" until its reference is a snapshot of the longitudinal outputs.
- Two steps still use cross-sectional files: road map 10 (../check_outputs_v2.py,
  outside pipeline/) and extract.py's references (outputs/rereg_20261102 and the
  cross-sectional liu and peak-MNI files). Decide before the repository is split.
- Existing later sessions still need step 07 v3 and a rebuilt HighLevel: 092, 108,
  101 ses-02 (reg/), and 008 ses-02, 010 ses-03 (reg.longitudinal).
- Scripts other than new_session.sh, 07 and extract.py may still hold the fixed
  path to pipeline/. This lists them:
    grep -rl "git_repos/sym_pt/pipeline" --include=*.py --include=*.sh --include=*.ipynb .


NOT YET (handoff, next steps)
--------------------------------------------------------------------------------
- A separate longitudinal session list, so new sessions no longer need a
  sub_info.csv row or EXCLUDE_SES edits; then move the 092, 108, 101 ses-02 rows
  there (guard: check_outputs must still match check_restored_20261102.txt).
- Stage 0 on the Mac as one script (rename, note, scp).
