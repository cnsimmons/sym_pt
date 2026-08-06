#!/usr/bin/env python3
"""
extract_selective_voxel_counts.py

Counts of category-selective voxels within VOTC mask, per subject × hemi × category × threshold.

Contrast: category-vs-all-others (FEAT copes 6=face, 7=house, 8=object, 9=word).
These are pre-computed Word>mean(others), Face>mean(others), etc. — no sign flip required
(unlike differential cope 13 which is Face>Word and needs flipping for word).

Mask: Harvard-Oxford VOTC (8 ventral/lateral occipitotemporal labels), hemisphere-split.
Same mask as voxel_allegiance_xs_liu.ipynb. MNI 2mm isotropic.

Z-stats: MNI-registered (zstat1_mni.nii.gz from HighLevel.gfeat, registered via
13_register_zstats_mni.py).

Thresholds: z > 1.96 (~p<.05) and z > 2.33 (~p<.01). Per Ayzenberg 2023 precedent for
hemispherectomy/resection patients.

Subjects: 22 OTC patients + 38 controls. nonOTC excluded. Controls = first session;
patients = first available post-surgery session. Patients use intact hemisphere only;
controls use both.

Output: roi_betas/selective_voxel_counts.csv
  rows = subject × hemi × category × threshold
  columns: subject_id, group, surgery_side, intact_hemi, hemi, category, threshold,
           n_selective_voxels, n_votc_voxels, pct_selective

Reference: Nordt et al. 2021 (Nat Hum Behav) — category-selective voxel volumes within
anatomical VTC partitions. Ayzenberg et al. 2023 (Dev Cogn Neurosci) — z>2.33 threshold
for resection patients.

Usage:
    python extract_selective_voxel_counts.py
"""

import os, sys, time
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import datasets as nl_datasets

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
                           is_patient, get_sub_info, _load_csv)

# ── Configuration ─────────────────────────────────────────────────────────────

# category-vs-all-others copes (from FEAT design.fsf)
CAT_COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
CATEGORIES = ['face', 'house', 'object', 'word']

# Thresholds (z-scores)
THRESHOLDS = [1.96, 2.33]

# Subject exclusions (current cross-sectional cohort: drops the polymicrogyria patient
# sub-017 and the two control exclusions sub-027, sub-084; keeps sub-083 / sub-085).
EXTRA_SKIP = {'sub-017', 'sub-027', 'sub-084'}
PRE_SURGERY_SESSIONS = {
    'sub-021': {'01'}, 'sub-045': {'01'}, 'sub-047': {'01'}, 'sub-049': {'01'},
    'sub-070': {'01'}, 'sub-073': {'01'}, 'sub-081': {'01'}, 'sub-086': {'01'},
    'sub-108': {'02'},
}

OUT_DIR = Path(processed_dir) / 'group_results' / 'roi_betas'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'selective_voxel_counts.csv'


# ── VOTC mask (Harvard-Oxford, hemisphere-split) ─────────────────────────────

def build_votc_masks():
    """Build VOTC masks (LH, RH) from Harvard-Oxford atlas. Matches voxel_allegiance_xs_liu.ipynb."""
    ho_atlas = nl_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_img = ho_atlas.maps if isinstance(ho_atlas.maps, nib.Nifti1Image) else nib.load(ho_atlas.maps)
    ho_data = ho_img.get_fdata()
    ho_labels = ho_atlas.labels

    VOTC_LABEL_NAMES = [
        'Temporal Fusiform Cortex, anterior division',
        'Temporal Fusiform Cortex, posterior division',
        'Temporal Occipital Fusiform Cortex',
        'Parahippocampal Gyrus, anterior division',
        'Parahippocampal Gyrus, posterior division',
        'Lingual Gyrus',
        'Lateral Occipital Cortex, superior division',
        'Lateral Occipital Cortex, inferior division',
    ]
    full = np.zeros(ho_data.shape, dtype=bool)
    for name in VOTC_LABEL_NAMES:
        matches = [i for i, l in enumerate(ho_labels) if name in l]
        if matches:
            full |= (ho_data == matches[0])

    mid_x = ho_data.shape[0] // 2
    mask_lh = full.copy(); mask_lh[:mid_x] = False
    mask_rh = full.copy(); mask_rh[mid_x:] = False
    return mask_lh, mask_rh, ho_img.shape


# ── Subject loader (cross-sectional, OTC + controls only) ────────────────────

def load_subjects():
    df = _load_csv()
    subjects = {}
    for sc in sorted(df['sub_clean'].unique()):
        if sc in skip_subs:
            continue
        sid = f'sub-{sc}'
        sessions = get_sessions(sc)
        if not sessions or not (Path(processed_dir) / sid).exists():
            continue
        info = get_sub_info(sc, sessions[0])
        pt = is_patient(sc)
        intact = info.get('intact_hemi', '')
        group = info.get('group', 'unknown')
        code_str = f"{group}{sc}"
        if code_str in EXTRA_SKIP or sid in EXTRA_SKIP:
            continue
        if group == 'nonOTC':
            continue

        # Pick first post-surgery session (patients) or first session (controls)
        post_sessions = []
        for s in sessions:
            ses_str = f'{s:02d}'
            if ses_str in PRE_SURGERY_SESSIONS.get(sid, set()):
                continue
            post_sessions.append(ses_str)
        if not post_sessions:
            continue

        subjects[sid] = {
            'session': post_sessions[0],
            'first_session': f'{sessions[0]:02d}',
            'group': group,
            'hemi': ('l' if intact == 'left' else 'r') if pt else None,
            'surgery_side': ('right' if intact == 'left' else 'left') if pt else 'na',
            'intact_hemi': intact if pt else 'both',
        }
    return subjects


# ── Z-stat path (MNI space) ──────────────────────────────────────────────────

def get_zstat_mni_path(sid, session, first_session, cope_num):
    """MNI-space z-stat from HighLevel.gfeat. Matches 13_register_zstats_mni.py output."""
    feat_dir = (Path(processed_dir) / sid / f'ses-{session}' / 'derivatives' / 'fsl'
                / 'loc' / 'HighLevel.gfeat' / f'cope{cope_num}.feat' / 'stats')
    # Same naming convention as 08_liu_distinctiveness.py
    if session == first_session:
        return feat_dir / 'zstat1_mni.nii.gz'
    else:
        return feat_dir / f'zstat1_ses{first_session}_mni.nii.gz'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading VOTC mask (Harvard-Oxford, hemisphere-split)...')
    mask_lh, mask_rh, expected_shape = build_votc_masks()
    n_lh, n_rh = int(mask_lh.sum()), int(mask_rh.sum())
    print(f'  LH VOTC: {n_lh:,} voxels')
    print(f'  RH VOTC: {n_rh:,} voxels')

    print('\nLoading subjects...')
    subjects = load_subjects()
    n_pt = sum(1 for v in subjects.values() if v['group'] == 'OTC')
    n_ctrl = sum(1 for v in subjects.values() if v['group'] == 'control')
    print(f'  Controls: {n_ctrl}   OTC patients: {n_pt}')

    rows = []
    skipped = []
    t0 = time.time()

    for idx, (sid, info) in enumerate(sorted(subjects.items())):
        ses = info['session']
        first_ses = info['first_session']
        hemis_to_run = ['l', 'r'] if info['group'] == 'control' else [info['hemi']]

        for cat in CATEGORIES:
            cope = CAT_COPES[cat]
            zpath = get_zstat_mni_path(sid, ses, first_ses, cope)
            if not zpath.exists():
                skipped.append((sid, ses, cat, f'no zstat: {zpath.name}'))
                continue

            try:
                z = nib.load(zpath).get_fdata()
            except Exception as e:
                skipped.append((sid, ses, cat, f'load failed: {e}'))
                continue

            if z.shape != expected_shape:
                skipped.append((sid, ses, cat,
                                f'shape mismatch: z={z.shape}, expected={expected_shape}'))
                continue

            for h in hemis_to_run:
                mask = mask_lh if h == 'l' else mask_rh
                n_votc = int(mask.sum())
                z_in_mask = z[mask]
                z_in_mask = z_in_mask[np.isfinite(z_in_mask)]

                for thresh in THRESHOLDS:
                    n_sel = int((z_in_mask > thresh).sum())
                    rows.append({
                        'subject_id':       sid,
                        'session':          ses,
                        'group':            info['group'],
                        'surgery_side':     info['surgery_side'],
                        'intact_hemi':      info['intact_hemi'],
                        'hemi':             h,
                        'category':         cat,
                        'threshold':        thresh,
                        'cope':             cope,
                        'n_selective':      n_sel,
                        'n_votc_voxels':    n_votc,
                        'pct_selective':    100.0 * n_sel / n_votc if n_votc > 0 else np.nan,
                    })

        if (idx + 1) % 10 == 0:
            print(f'  [{idx+1}/{len(subjects)}] {time.time()-t0:.0f}s elapsed')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nDone in {time.time()-t0:.0f}s')
    print(f'Wrote: {OUT_CSV}  ({len(df)} rows)')

    # Quick sanity: counts of zero per category × group at each threshold
    print('\nZero-count diagnostic (n_subjects with n_selective == 0):')
    print(f'{"Group":>10s} {"Cat":>8s} {"Hemi":>5s} {"Thresh":>7s} {"n_zero":>7s} {"n_total":>8s}')
    print('-' * 55)
    for grp in ('control', 'OTC'):
        for cat in CATEGORIES:
            for h in ('l', 'r'):
                for t in THRESHOLDS:
                    sub = df[(df['group']==grp) & (df['category']==cat) &
                             (df['hemi']==h) & (df['threshold']==t)]
                    if len(sub) == 0: continue
                    n_zero = int((sub['n_selective'] == 0).sum())
                    print(f'{grp:>10s} {cat:>8s} {h:>5s} {t:>7.2f} {n_zero:>7d} {len(sub):>8d}')

    if skipped:
        print(f'\nSkipped {len(skipped)}:')
        for s in skipped[:25]:
            print(f'  {s}')
        if len(skipped) > 25:
            print(f'  ... and {len(skipped)-25} more')


if __name__ == '__main__':
    main()