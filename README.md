# sym_pt — Symmetric Patient fMRI Pipeline

Analysis pipeline for category-selective visual processing in hemispherectomy patients and controls. Localizer task (faces, houses, objects, words, scrambled) with longitudinal and cross-sectional analyses.

## Subjects

- **Patients**: Hemispherectomy patients with 1–5 sessions (sub-004/UD, sub-007/OT, sub-021/TC, and others)
- **Controls**: 24 typically developing controls (9 longitudinal, 15 cross-sectional only)
- **Skipped**: sub-108 (awaiting data)

Subject info in `sub_info.csv` (long format: one row per subject-session).

## Directory Structure

/user_data/csimmon2/sym_pt/          # Processed data
  sub-XXX/ses-XX/
    anat/                            # T1w, skull-stripped brain, brain mask, mirror (patients)
      T1w.nii.gz
      T1w_brain.nii.gz
      T1w_brain_mask.nii.gz
      T1w_brain_mirrored.nii.gz      # Patients only
      T1w_brain_stand.nii.gz         # Registered to MNI
      anat2stand.mat                 # Native -> MNI (FLIRT dof 12)
      mni2anat.mat                   # MNI -> Native (convert_xfm inverse)
      anat2ses01.mat                 # Later sessions only: anat -> ses-01 anat
    timing/                          # FSL 3-column timing files per run/condition
    derivatives/
      fsl/loc/run-XX/
        1stLevel.fsf                 # FEAT design file
        1stLevel.feat/               # FEAT output
          filtered_func_data.nii.gz
          filtered_func_data_reg.nii.gz  # Registered to ses-01 anat
          stats/cope1..cope19.nii.gz
          reg/example_func2standard.mat  # Func -> ses-01 anat
      rois/                          # MNI ROI parcels warped to native space
  rois/                              # Staged MNI-space ROI parcels (split L/R)

/lab_data/behrmannlab/hemi/Raw/      # Raw BIDS data (read-only)
/user_data/csimmon2/git_repos/sym_pt/ # Code repository
## Pipeline Scripts

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_organize.py` | Create directory structure for all subjects |
| 02 | `02_convert_timing.py` | Convert BIDS events.tsv to FSL 3-column timing files |
| 03 | `03_register_mirror.py` | BET skull strip, hemisphere-aware mirroring (patients), FLIRT to MNI, warp ROIs to native. Submitted via `submit_anatomy.sh` |
| 04 | `04_check_registration` | QC: verify anatomy and registration quality |
| 05 | `05_create_fsf.sh` | Generate 1stLevel FSF files from template for all runs |
| 06 | `06_submit_1stLevel_feat.sh` | Submit FEAT 1st-level jobs with throttling (max 12 concurrent) |
| 07 | `07_register_1stlevel.py` | Register filtered_func_data to ses-01 anat using FEAT's registration matrix |
| 08 | `08_register_anat_to_ses01.sh` | Register later session anatomies to first session (multi-session subjects only) |

## FEAT Contrasts (1st Level)

| COPE | Name | Contrast |
|------|------|----------|
| 1 | Face | Face > Object |
| 2 | House | House > Object |
| 3 | Object | Object > Scramble |
| 4 | Word | Word > Object |
| 5 | Scramble | Scramble > mean(all) |
| 6 | Face-all | Face > mean(House+Object+Word+Scramble) |
| 7 | House-all | House > mean(Face+Object+Word+Scramble) |
| 8 | Object-all | Object > mean(Face+House+Word+Scramble) |
| 9 | Word-all | Word > mean(Face+House+Object+Scramble) |
| 10 | Face-scramble | Face > Scramble |
| 11 | House-scramble | House > Scramble |
| 12 | Word-scramble | Word > Scramble |
| 13 | Face-Word | Face > Word |
| 14 | Object-House | Object > House |
| 15 | Face_raw | Face (raw beta) |
| 16 | House_raw | House (raw beta) |
| 17 | Object_raw | Object (raw beta) |
| 18 | Word_raw | Word (raw beta) |
| 19 | Scramble_raw | Scramble (raw beta) |

### COPE usage by analysis

- **ROI definition** (category > scramble): face=10, house=11, object=3, word=12
- **RSA** (category > all others): faces=6, houses=7, objects=8, words=9
- **RSA** (raw betas): face=15, house=16, object=17, word=18, scramble=19
- **Differential/competition**: face=(10,+1), word=(13,−1), object=(3,+1), house=(11,+1)

## Registration Strategy

- **Controls**: Standard FLIRT (dof 12) brain → MNI
- **Patients**: Hemisphere-aware mirroring (intact hemisphere copied to resected side), FLIRT on mirror brain → MNI, inverse via `convert_xfm`
- **FEAT "standard"**: Set to first session's T1w_brain (not MNI), so `example_func2standard.mat` = func → ses-01 anat
- **Inter-session**: Later session anats registered to first session anat (FLIRT dof 6, rigid body)
- **ROIs**: MNI parcels warped to native space via `mni2anat.mat`, binarized

## Configuration

All paths and parameters in `sym_pt_params.py`. Key settings:
- `skip_subs = ['108']`
- `task = 'loc'`
- `conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']`

## Dependencies

- FSL 6.0.3
- Python 3.9+ (conda env: `fmri`)
- nibabel, nilearn, pandas, numpy
