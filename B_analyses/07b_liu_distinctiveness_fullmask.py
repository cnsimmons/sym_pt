#!/usr/bin/env python3
"""
07b_pairwise_searchmask.py - Pairwise correlations within full anatomical searchmasks

Replicates Liu et al. (2025) longitudinal crowding analysis:
  - Uses ALL voxels within each category's anatomical searchmask
    (NO functional thresholding — matches Liu's FG/OTS approach)
  - Extracts RSA beta patterns (copes 15-18) for face/house/object/word
  - Computes all 6 pairwise Fisher-z correlations

Key methodological notes:
  - Liu used the full anatomically-defined FG/OTS region (7000-12000 voxels)
    to track voxel-by-voxel selectivity changes over time
  - face and word share the SAME searchmask (Temporal Fusiform), so
    face-word competition is measured within shared ventral territory
  - No z-threshold applied — all voxels in the atlas-derived mask are used

Output: group_results/liu_distinctiveness/pairwise_searchmask_{cope_set}.csv

Usage:
  python B_analyses/07b_pairwise_searchmask.py
"""

import os, sys, time, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(processed_dir)
OUTPUT_DIR = Path(f'{processed_dir}/group_results/liu_distinctiveness')

# Exclusions (same as 07_liu_distinctiveness.py)
SUBJECTS_TO_SKIP = ['control083', 'control085']

PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
}

# RSA betas: always copes 15-18 regardless of localisation contrast
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}
RSA_CATEGORIES = ['face', 'house', 'object', 'word']

# Searchmask categories to process
# NOTE: face and word use the SAME atlas parcels (Temporal Fusiform)
# so their results will be identical. Both are included for clarity.
SEARCHMASK_CATEGORIES = ['face', 'house', 'object', 'word']

# ── NIfTI Cache ───────────────────────────────────────────────────────────────

_CACHE = {}

def _load(fp):
    k = str(fp)
    if k not in _CACHE:
        _CACHE[k] = nib.load(k)
    return _CACHE[k]

# ── Load Subjects ─────────────────────────────────────────────────────────────

def load_subjects():
    df = _load_csv()
    subjects = {}
    for sub_clean in sorted(df['sub_clean'].unique()):
        if sub_clean in skip_subs:
            continue
        sid      = f'sub-{sub_clean}'
        sessions = get_sessions(sub_clean)
        if not sessions or not (BASE_DIR / sid).exists():
            continue
        info   = get_sub_info(sub_clean, sessions[0])
        pt     = is_patient(sub_clean)
        intact = info.get('intact_hemi', '')
        code   = f"{info.get('group','')}{sub_clean}"
        if code in SUBJECTS_TO_SKIP:
            continue
        subjects[sid] = {
            'code':           code,
            'sessions':       [f'{s:02d}' for s in sessions],
            'hemi':           ('l' if intact == 'left' else 'r') if pt else None,
            'group':          info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi':    intact,
            'surgery_side':   ('right' if intact == 'left' else 'left') if pt else 'na',
        }
    return subjects


# ── Core Functions ────────────────────────────────────────────────────────────

def get_searchmask(subject_id, searchmask_cat, hemi, subs):
    """
    Get the raw anatomical searchmask — NO functional thresholding.
    This matches Liu's approach: all voxels in FG/OTS, not just
    suprathreshold ones.
    
    Returns: (boolean 3D array, affine) or (None, None) if mask not found.
    """
    info      = subs[subject_id]
    first_ses = info['sessions'][0]

    roi_dir = BASE_DIR / subject_id / f'ses-{first_ses}' / 'ROIs'
    mask_file = roi_dir / f'{hemi}_{searchmask_cat}_searchmask.nii.gz'
    if not mask_file.exists():
        return None, None

    mask_img = _load(mask_file)
    mask = mask_img.get_fdata() > 0

    if mask.sum() < 10:
        return None, None

    return mask, mask_img.affine


def extract_betas_in_mask(subject_id, session, bool_mask, subs):
    """
    Extract RSA beta patterns (copes 15-18) within a boolean mask.
    Returns: (n_voxels × 4) array, list of valid categories, or (None, None).
    """
    info      = subs[subject_id]
    first_ses = info['sessions'][0]
    feat      = (BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' /
                 'fsl' / 'loc' / 'HighLevel.gfeat')
    cn = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'

    patterns, valid_cats = [], []
    for cat in RSA_CATEGORIES:
        cf = feat / f'cope{RSA_COPES[cat]}.feat' / 'stats' / cn
        if not cf.exists():
            continue
        betas = _load(cf).get_fdata()[bool_mask]
        betas = betas[np.isfinite(betas)]
        if len(betas) > 0:
            patterns.append(betas)
            valid_cats.append(cat)

    if len(patterns) < 4:
        return None, None

    # Ensure all patterns have same length
    min_v = min(len(b) for b in patterns)
    return np.column_stack([b[:min_v] for b in patterns]), valid_cats


def compute_pairwise(beta_matrix, valid_cats):
    """
    Compute all pairwise Fisher-z correlations.
    Returns dict: 'cat1-cat2' → fisher_z value
    """
    corr   = np.corrcoef(beta_matrix.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))

    pairwise = {}
    for i in range(len(valid_cats)):
        for j in range(i + 1, len(valid_cats)):
            pairwise[f'{valid_cats[i]}-{valid_cats[j]}'] = float(fisher[i, j])

    return pairwise


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(subs):
    print(f'\n{"="*70}')
    print(f'SEARCHMASK PAIRWISE (no functional threshold — full anatomical mask)')
    print(f'  Searchmask categories: {SEARCHMASK_CATEGORIES}')
    print(f'  NOTE: face and word searchmasks are identical (Temporal Fusiform)')
    print(f'{"="*70}')
    t0 = time.time()

    rows = []

    for sub_idx, (sid, info) in enumerate(sorted(subs.items())):
        code = info['code']
        if (sub_idx + 1) % 10 == 0 or sub_idx == 0:
            print(f'  [{sub_idx+1}/{len(subs)}] {code} ({time.time()-t0:.0f}s)')

        for session in info['sessions']:
            if sid in PRE_SURGERY_SESSIONS and session in PRE_SURGERY_SESSIONS[sid]:
                continue

            hemis = ['l', 'r'] if info['patient_status'] == 'control' else [info['hemi']]

            for hemi in hemis:
                for sm_cat in SEARCHMASK_CATEGORIES:

                    # Get raw anatomical searchmask (no thresholding)
                    mask, affine = get_searchmask(sid, sm_cat, hemi, subs)
                    if mask is None:
                        continue

                    n_voxels = int(mask.sum())

                    # Extract RSA betas within the full mask
                    betas, valid_cats = extract_betas_in_mask(
                        sid, session, mask, subs)
                    if betas is None:
                        continue

                    # Compute pairwise correlations
                    pairwise = compute_pairwise(betas, valid_cats)

                    # Hemisphere label
                    if info['patient_status'] == 'patient':
                        hl = ('intact' if (hemi == 'l' and info['intact_hemi'] == 'left') or
                              (hemi == 'r' and info['intact_hemi'] == 'right') else 'lesioned')
                    else:
                        hl = 'left' if hemi == 'l' else 'right'

                    base = {
                        'subject':      code,
                        'subject_id':   sid,
                        'group':        info['group'] if info['patient_status'] == 'patient' else 'control',
                        'status':       info['patient_status'],
                        'surgery_side': info['surgery_side'],
                        'session':      session,
                        'hemi':         hemi,
                        'hemi_label':   hl,
                        'searchmask':   sm_cat,
                        'n_voxels':     n_voxels,
                    }

                    for pair, val in pairwise.items():
                        rows.append({**base, 'pair': pair, 'fisher_r': val})

    print(f'\n  Done: {time.time()-t0:.0f}s, {len(rows)} pairwise measurements')
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subs   = load_subjects()
    #subs = {'sub-108': subs['sub-108']}
    n_pt   = sum(1 for v in subs.values() if v['patient_status'] == 'patient')
    n_ctrl = sum(1 for v in subs.values() if v['patient_status'] == 'control')
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Skipping: {SUBJECTS_TO_SKIP}')

    df = run_pipeline(subs)

    out_file = OUTPUT_DIR / f'pairwise_searchmask_differential.csv'
    df.to_csv(out_file, index=False)
    print(f'Saved: {out_file}')

    # Quick summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    print(f'Subjects: {df["subject_id"].nunique()}')
    print(f'Rows: {len(df)}')
    print(f'\nVoxels per searchmask (full anatomical, no threshold):')
    for sm in SEARCHMASK_CATEGORIES:
        v = df[df['searchmask'] == sm].groupby(
            ['subject_id', 'session', 'hemi'])['n_voxels'].first()
        if len(v) > 0:
            print(f'  {sm}: M={v.mean():.0f} (SD={v.std():.0f}, '
                  f'range={v.min():.0f}-{v.max():.0f})')

    # Verify face = word mask
    face_v = df[df['searchmask'] == 'face'].groupby(
        ['subject_id', 'session', 'hemi'])['n_voxels'].first()
    word_v = df[df['searchmask'] == 'word'].groupby(
        ['subject_id', 'session', 'hemi'])['n_voxels'].first()
    if len(face_v) > 0 and len(word_v) > 0:
        match = (face_v.values == word_v.values).all() if len(face_v) == len(word_v) else False
        print(f'\n  Face and word searchmask voxel counts identical: {match}')

    print('\nDone!')


if __name__ == '__main__':
    main()