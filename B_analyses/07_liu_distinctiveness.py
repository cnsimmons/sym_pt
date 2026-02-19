#!/usr/bin/env python3
"""
08_liu_distinctiveness.py - Compute Liu representational distinctiveness

For each subject × session × hemisphere × category:
  1. Localize ROI using LOC_COPES (dynamic per session)
  2. Build 6mm sphere around peak
  3. Extract raw beta patterns (RSA_COPES 15-18)
  4. Compute mean Fisher-z correlation between preferred and non-preferred categories

Runs all three contrast maps and saves separate CSVs.

Exclusions (documented):
  - control083: pathological RSA beta values in house sphere (|β|>100 in 89%
    of voxels, Fisher-z house/face-house=3.71). Likely GLM scaling artifact.
    Peak coordinates valid — retained in peak_coords.csv. Confirmed 2025-02-19.
  - control085: pathological RSA beta values in house sphere (|β|>100 in 16%
    of voxels, max=536). Fisher-z house/face-house=3.12. Confirmed 2025-02-19.
    Peak coordinates valid — retained in peak_coords.csv.
  - sub-090 (OTC, KT): RSA beta files (copes 15-18) missing from HighLevel.gfeat.
    Localisation zstats present. Excluded until HighLevel is rerun.
    See reprocessing checklist.

Usage:
  python B_analyses/08_liu_distinctiveness.py
  python B_analyses/08_liu_distinctiveness.py --cope-set differential
  python B_analyses/08_liu_distinctiveness.py --cope-set all
"""

import os, sys, time, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import label, center_of_mass

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(processed_dir)
OUTPUT_DIR = Path(f'{processed_dir}/group_results/liu_distinctiveness')

# ── Exclusions ────────────────────────────────────────────────────────────────
# See module docstring for full rationale.
SUBJECTS_TO_SKIP = ['OTC108', 'control083', 'control085']

PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
}

# ── Cope maps: localisation cope per category × contrast set ──────────────────
# ORIGINAL sets unchanged. New sub-ROIs added to each set.
# RSA betas always use copes 15-18 regardless of cope set.
COPE_MAPS = {
    'differential': {
        # original
        'face': 1, 'house': 2, 'object': 3, 'word': 4,
        # house split (required — bimodal Y distribution, GMM k=3 >> k=1)
        'house_PPA': 2, 'house_TOS': 2,
        # face sub-ROIs
        'face_FFA': 1, 'face_STS': 1,
        # object sub-ROIs
        'object_LOC': 3, 'object_pF': 3,
        # word sub-ROIs
        'word_VWFA': 4, 'word_STG': 9,
        # early visual cortex
        'evc': 3,
    },
    'cat_vs_scramble': {
        'face': 10, 'house': 11, 'object': 3, 'word': 12,
        'house_PPA': 11, 'house_TOS': 11,
        'face_FFA':  10, 'face_STS':  10,
        'object_LOC': 3, 'object_pF':  3,
        'word_VWFA': 12, 'word_STG':  12,
        'evc': 3,
    },
    'hybrid': {
        'face': 1, 'house': 2, 'object': 3, 'word': 12,
        'house_PPA': 2,  'house_TOS': 2,
        'face_FFA':  1,  'face_STS':  1,
        'object_LOC': 3, 'object_pF': 3,
        'word_VWFA': 12, 'word_STG': 12,
        'evc': 3,
    },
}

# RSA beta copes (raw condition betas — same for all cope sets)
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

# All categories (used for RSA beta extraction within each sphere)
CATEGORIES = [
    'face', 'house', 'object', 'word',
    'house_PPA', 'house_TOS',
    'face_FFA', 'face_STS',
    'object_LOC', 'object_pF',
    'word_VWFA', 'word_STG',
    'evc',
]

# Bilateral categories (both hemispheres analyzed for controls and patients)
BILATERAL_CATEGORIES = [
    'object', 'house', 'house_PPA', 'house_TOS',
    'object_LOC', 'object_pF', 'evc',
]

# Preferred hemisphere per category (used for roi_status labelling)
PREFERRED_HEMI = {
    'face':       'r',
    'word':       'l',
    'house':      'both',
    'object':     'both',
    'house_PPA':  'both',
    'house_TOS':  'both',
    'face_FFA':   'r',
    'face_STS':   'r',
    'object_LOC': 'both',
    'object_pF':  'both',
    'word_VWFA':  'l',
    'word_STG':   'l',
    'evc':        'both',
}

THRESHOLD_Z  = 1.96
TOP_PCT      = 0.10
MIN_VOXELS   = 50
SPHERE_RADIUS = 6

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

def extract_roi(subject_id, session, category, hemi, loc_copes, subs):
    info      = subs[subject_id]
    first_ses = info['sessions'][0]
    cope_num  = loc_copes[category]

    bm_file = BASE_DIR / subject_id / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    mf = None
    for sd in ['ROIs', os.path.join('derivatives', 'rois')]:
        p = BASE_DIR / subject_id / f'ses-{first_ses}' / sd / f'{hemi}_{category}_searchmask.nii.gz'
        if p.exists():
            mf = p; break
    if mf is None:
        return None

    mi     = _load(mf)
    mask   = mi.get_fdata() > 0
    affine = mi.affine

    feat  = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf    = feat / f'cope{cope_num}.feat' / 'stats' / zname
    if not zf.exists():
        return None

    z = _load(zf).get_fdata().copy()
    if bm is not None:
        z[~bm] = 0

    supra = (z > THRESHOLD_Z) & mask
    ns    = supra.sum()
    if ns < MIN_VOXELS:
        return None

    top_n  = max(MIN_VOXELS, int(ns * TOP_PCT))
    top_n  = min(top_n, ns)
    thresh = np.sort(z[supra])[-top_n]
    top    = (z >= thresh) & supra

    labeled, nc = label(top)
    if nc == 0:
        return None

    sizes = [(labeled == i).sum() for i in range(1, nc + 1)]
    li    = np.argmax(sizes) + 1
    roi   = (labeled == li)
    pidx  = np.unravel_index(np.argmax(z * roi), z.shape)

    return {
        'n_voxels':    sizes[li - 1],
        'peak_z':      z[pidx],
        'peak_coord':  nib.affines.apply_affine(affine, np.array(pidx)),
        'centroid':    nib.affines.apply_affine(affine, np.array(center_of_mass(roi))),
        'affine':      affine,
        'brain_shape': z.shape,
    }


def create_sphere(peak_coord, affine, brain_shape, radius=SPHERE_RADIUS):
    grid  = np.array(np.meshgrid(
        np.arange(brain_shape[0]), np.arange(brain_shape[1]),
        np.arange(brain_shape[2]), indexing='ij'
    )).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    mask  = np.zeros(brain_shape, dtype=bool)
    for c in grid[dists <= radius]:
        mask[c[0], c[1], c[2]] = True
    return mask


def extract_sphere_betas(subject_id, session, sphere_mask, subs):
    info      = subs[subject_id]
    first_ses = info['sessions'][0]
    feat      = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn        = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'

    patterns, valid_cats = [], []
    for cat in ['face', 'house', 'object', 'word']:   # RSA always uses 4 base categories
        cf = feat / f'cope{RSA_COPES[cat]}.feat' / 'stats' / cn
        if not cf.exists():
            continue
        betas = _load(cf).get_fdata()[sphere_mask]
        betas = betas[np.isfinite(betas)]
        if len(betas) > 0:
            patterns.append(betas)
            valid_cats.append(cat)

    if len(patterns) < 4:
        return None, None

    min_v = min(len(b) for b in patterns)
    return np.column_stack([b[:min_v] for b in patterns]), valid_cats


def compute_liu(beta_matrix, valid_cats, roi_category):
    """Compute Liu distinctiveness for the ROI category.
    Uses the four base categories (face/house/object/word) for RSA
    regardless of which sub-ROI was used for localisation."""
    # Map sub-ROI category to its parent base category for RSA lookup
    parent_map = {
        'house_PPA':  'house', 'house_TOS':  'house',
        'face_FFA':   'face',  'face_STS':   'face',
        'object_LOC': 'object','object_pF':  'object',
        'word_VWFA':  'word',  'word_STG':   'word',
        'evc':        'object',  # EVC: use object as reference (least category-specific)
    }
    rsa_cat = parent_map.get(roi_category, roi_category)

    if valid_cats is None or len(valid_cats) < 4 or rsa_cat not in valid_cats:
        return None, None

    corr   = np.corrcoef(beta_matrix.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))
    pi     = valid_cats.index(rsa_cat)
    nonp   = [i for i in range(len(valid_cats)) if i != pi]
    liu_val = float(np.mean(fisher[pi, nonp]))

    pairwise = {}
    for i in range(len(valid_cats)):
        for j in range(i + 1, len(valid_cats)):
            pairwise[f'{valid_cats[i]}-{valid_cats[j]}'] = float(fisher[i, j])

    return liu_val, pairwise

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(loc_copes, cope_set_name, subs):
    print(f'\n{"="*70}')
    print(f'COPE SET: {cope_set_name} → {loc_copes}')
    print(f'{"="*70}')
    t0 = time.time()

    liu_rows      = []
    pairwise_rows = []

    for sub_idx, (sid, info) in enumerate(sorted(subs.items())):
        code = info['code']
        print(f'  [{sub_idx+1}/{len(subs)}] {code} ({time.time()-t0:.0f}s)', end='\r')

        for session in info['sessions']:
            if sid in PRE_SURGERY_SESSIONS and session in PRE_SURGERY_SESSIONS[sid]:
                continue

            hemis = ['l', 'r'] if info['patient_status'] == 'control' else [info['hemi']]

            for hemi in hemis:
                for category in loc_copes.keys():
                    roi = extract_roi(sid, session, category, hemi, loc_copes, subs)
                    if roi is None:
                        continue

                    sphere = create_sphere(roi['peak_coord'], roi['affine'], roi['brain_shape'])
                    betas, vcats = extract_sphere_betas(sid, session, sphere, subs)
                    if betas is None:
                        continue

                    liu_val, pairwise = compute_liu(betas, vcats, category)
                    if liu_val is None:
                        continue

                    # Hemisphere label
                    if info['patient_status'] == 'patient':
                        hl = ('intact' if (hemi == 'l' and info['intact_hemi'] == 'left') or
                              (hemi == 'r' and info['intact_hemi'] == 'right') else 'lesioned')
                    else:
                        hl = 'left' if hemi == 'l' else 'right'

                    cat_type = 'bilateral' if category in BILATERAL_CATEGORIES else 'unilateral'

                    # ROI reorganization status
                    if info['patient_status'] == 'patient' and cat_type == 'unilateral':
                        pref_h = PREFERRED_HEMI.get(category, 'both')
                        actual_h = 'left' if hemi == 'l' else 'right'
                        if info['surgery_side'] == 'left':
                            roi_status = 'reorganized' if category in (
                                'word', 'word_VWFA', 'word_STG') else 'typical'
                        else:
                            roi_status = 'reorganized' if category in (
                                'face', 'face_FFA', 'face_STS') else 'typical'
                    else:
                        roi_status = 'control' if info['patient_status'] == 'control' else 'bilateral'

                    base = {
                        'subject':      code,
                        'subject_id':   sid,
                        'group':        info['group'] if info['patient_status'] == 'patient' else 'control',
                        'status':       info['patient_status'],
                        'surgery_side': info['surgery_side'],
                        'session':      session,
                        'hemi':         hemi,
                        'hemi_label':   hl,
                        'category':     category,
                        'cat_type':     cat_type,
                        'roi_status':   roi_status,
                        'cope_set':     cope_set_name,
                    }

                    liu_rows.append({**base,
                                     'liu_distinctiveness': liu_val,
                                     'peak_z':              roi['peak_z'],
                                     'n_voxels':            roi['n_voxels'],
                                     'sphere_voxels':       sphere.sum()})

                    for pair, val in pairwise.items():
                        pairwise_rows.append({**base, 'pair': pair, 'fisher_r': val})

    print(f'\n  Done: {time.time()-t0:.0f}s, {len(liu_rows)} measurements')
    return pd.DataFrame(liu_rows), pd.DataFrame(pairwise_rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Liu distinctiveness analysis')
    parser.add_argument('--cope-set', type=str, default='all',
                        choices=['differential', 'cat_vs_scramble', 'hybrid', 'all'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subs   = load_subjects()
    n_pt   = sum(1 for v in subs.values() if v['patient_status'] == 'patient')
    n_ctrl = sum(1 for v in subs.values() if v['patient_status'] == 'control')
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Skipping: {SUBJECTS_TO_SKIP}')

    cope_sets = COPE_MAPS if args.cope_set == 'all' else {args.cope_set: COPE_MAPS[args.cope_set]}

    for name, copes in cope_sets.items():
        _CACHE.clear()
        liu_df, pair_df = run_pipeline(copes, name, subs)

        liu_df.to_csv(OUTPUT_DIR / f'liu_distinctiveness_{name}.csv',  index=False)
        pair_df.to_csv(OUTPUT_DIR / f'pairwise_correlations_{name}.csv', index=False)
        print(f'  Saved: liu_distinctiveness_{name}.csv')
        print(f'  Saved: pairwise_correlations_{name}.csv')

    print('\nDone!')


if __name__ == '__main__':
    main()