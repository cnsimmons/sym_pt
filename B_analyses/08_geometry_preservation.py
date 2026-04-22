#!/usr/bin/env python3
"""
09_geometry_preservation.py - Compute representational geometry metrics

For subjects with 2+ post-surgery sessions:
  1. Localize ROI per session using LOC_COPES (dynamic ROI)
  2. Build 6mm sphere at each session's centroid
  3. Extract raw beta patterns (RSA_COPES 15-18) — no circularity
  4. Compute:
     a. Spatial relocation (centroid distance T1→T2)
     b. Geometry preservation (RDM correlation T1→T2)
     c. MDS shift (Procrustes-aligned embedding distance per category)

Runs all three contrast maps and saves separate CSVs.

Exclusions (documented):
  - control083: pathological RSA beta values in house sphere (|β|>100 in 89%
    of voxels, Fisher-z house/face-house=3.71). Likely GLM scaling artifact.
    Peak coordinates valid — retained in peak_coords.csv. Confirmed 2025-02-19.
  - control085: pathological RSA beta values in house sphere (|β|>100 in 16%
    of voxels, max=536). Fisher-z house/face-house=3.12. Confirmed 2025-02-19.
    Peak coordinates valid — retained in peak_coords.csv.
  - sub-090 (OTC, KT): RSA beta files (copes 15-18) missing from HighLevel.gfeat.
    Excluded until HighLevel is rerun. See reprocessing checklist.

Usage:
  python B_analyses/09_geometry_preservation.py
  python B_analyses/09_geometry_preservation.py --cope-set differential
  python B_analyses/09_geometry_preservation.py --cope-set all
"""

import os, sys, time, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import label, center_of_mass
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(processed_dir)
OUTPUT_DIR = Path(f'{processed_dir}/group_results/geometry')

# ── Exclusions ────────────────────────────────────────────────────────────────
# See module docstring for full rationale.
SUBJECTS_TO_SKIP = ['control083', 'control085']

PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
}

# ── Cope maps ─────────────────────────────────────────────────────────────────
# ORIGINAL sets unchanged. New sub-ROIs added to each set.
COPE_MAPS = {
    'differential': {
        # original
        'face': 1, 'house': 2, 'object': 3, 'word': 13,
        # house split (required — bimodal Y distribution confirmed)
        'house_PPA': 2, 'house_TOS': 2,
        # face sub-ROIs
        'face_FFA': 1, 'face_STS': 1,
        # object sub-ROIs
        'object_LOC': 3, 'object_pF': 3,
        # word sub-ROIs
        'word_VWFA': 13, 'word_STG': 13,
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

RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

# All categories processed
CATEGORIES = [
    'face', 'house', 'object', 'word',
    'house_PPA', 'house_TOS',
    'face_FFA', 'face_STS',
    'object_LOC', 'object_pF',
    'word_VWFA', 'word_STG',
    'evc',
]

# RSA beta extraction always uses the four base categories
RSA_CATEGORIES = ['face', 'house', 'object', 'word']

BILATERAL_CATEGORIES = [
    'object', 'house', 'house_PPA', 'house_TOS',
    'object_LOC', 'object_pF', 'evc',
]

THRESHOLD_Z   = 1.96
TOP_PCT       = 0.10
MIN_VOXELS    = 50
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
    zn    = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf    = feat / f'cope{cope_num}.feat' / 'stats' / zn
    if not zf.exists():
        return None

    z = _load(zf).get_fdata().copy()
    if category in ('word', 'word_VWFA', 'word_STG'):
        z = -z
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

    return {
        'n_voxels':    sizes[li - 1],
        'peak_z':      z[np.unravel_index(np.argmax(z * roi), z.shape)],
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
    """RSA betas always extracted for the four base categories."""
    info      = subs[subject_id]
    first_ses = info['sessions'][0]
    feat      = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn        = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'

    patterns, valid_cats = [], []
    for cat in RSA_CATEGORIES:
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


def mds_2d(rdm):
    n = rdm.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (rdm ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(loc_copes, cope_set_name, subs):
    print(f'\n{"="*70}')
    print(f'COPE SET: {cope_set_name} → {loc_copes}')
    print(f'{"="*70}')
    t0 = time.time()

    spatial_rows  = []
    geometry_rows = []
    mds_rows      = []

    for sub_idx, (sid, info) in enumerate(sorted(subs.items())):
        code = info['code']
        print(f'  [{sub_idx+1}/{len(subs)}] {code} ({time.time()-t0:.0f}s)', end='\r')

        post = [s for s in info['sessions']
                if not (sid in PRE_SURGERY_SESSIONS and s in PRE_SURGERY_SESSIONS[sid])]
        if len(post) < 2:
            continue

        s1, s2 = post[0], post[-1]
        hemis  = [info['hemi']] if info['patient_status'] == 'patient' else ['l', 'r']

        for hemi in hemis:
            for category in loc_copes.keys():
                roi_t1 = extract_roi(sid, s1, category, hemi, loc_copes, subs)
                roi_t2 = extract_roi(sid, s2, category, hemi, loc_copes, subs)
                if roi_t1 is None or roi_t2 is None:
                    continue

                affine      = roi_t1['affine']
                brain_shape = roi_t1['brain_shape']

                relocation_mm = float(np.linalg.norm(
                    roi_t1['centroid'] - roi_t2['centroid']))

                sphere_t1 = create_sphere(roi_t1['centroid'], affine, brain_shape)
                sphere_t2 = create_sphere(roi_t2['centroid'], affine, brain_shape)

                betas_t1, cats_t1 = extract_sphere_betas(sid, s1, sphere_t1, subs)
                betas_t2, cats_t2 = extract_sphere_betas(sid, s2, sphere_t2, subs)
                if betas_t1 is None or betas_t2 is None or cats_t1 != cats_t2:
                    continue

                rdm_t1 = 1 - np.corrcoef(betas_t1.T)
                rdm_t2 = 1 - np.corrcoef(betas_t2.T)

                triu      = np.triu_indices(4, k=1)
                r_geom, _ = pearsonr(rdm_t1[triu], rdm_t2[triu])

                # MDS shift (Procrustes-aligned)
                mds_shifts = {}
                try:
                    c1 = mds_2d(rdm_t1)
                    c2 = mds_2d(rdm_t2)
                    R, _        = orthogonal_procrustes(c1, c2)
                    c1_aligned  = c1 @ R
                    for i, cat in enumerate(cats_t1):
                        mds_shifts[cat] = float(np.linalg.norm(c1_aligned[i] - c2[i]))
                except Exception:
                    mds_shifts = {}

                # Metadata
                cat_type = 'bilateral' if category in BILATERAL_CATEGORIES else 'unilateral'
                if info['patient_status'] == 'patient' and cat_type == 'unilateral':
                    if info['surgery_side'] == 'left':
                        roi_status = 'reorganized' if category in (
                            'word', 'word_VWFA', 'word_STG') else 'typical'
                    else:
                        roi_status = 'reorganized' if category in (
                            'face', 'face_FFA', 'face_STS') else 'typical'
                elif info['patient_status'] == 'control':
                    roi_status = 'control'
                else:
                    roi_status = 'bilateral'

                hl = 'intact' if info['patient_status'] == 'patient' else (
                    'left' if hemi == 'l' else 'right')

                base = {
                    'subject':      code,
                    'subject_id':   sid,
                    'group':        info['group'] if info['patient_status'] == 'patient' else 'control',
                    'status':       info['patient_status'],
                    'surgery_side': info['surgery_side'],
                    'hemi':         hemi,
                    'hemi_label':   hl,
                    'category':     category,
                    'cat_type':     cat_type,
                    'roi_status':   roi_status,
                    'session_1':    s1,
                    'session_2':    s2,
                    'cope_set':     cope_set_name,
                }

                spatial_rows.append({**base, 'relocation_mm': relocation_mm})
                geometry_rows.append({**base, 'geometry_preservation': float(r_geom)})

                for mcat, shift in mds_shifts.items():
                    mds_rows.append({
                        **base,
                        'measured_category':  mcat,
                        'measured_cat_type':  'bilateral' if mcat in BILATERAL_CATEGORIES else 'unilateral',
                        'mds_shift':          shift,
                    })

    print(f'\n  Done: {time.time()-t0:.0f}s')
    spatial_df  = pd.DataFrame(spatial_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    mds_df      = pd.DataFrame(mds_rows)
    print(f'  Spatial: {len(spatial_df)}, Geometry: {len(geometry_df)}, MDS: {len(mds_df)}')
    return spatial_df, geometry_df, mds_df

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(spatial_df, geometry_df, mds_df, cope_set_name):
    print(f'\n--- {cope_set_name} SUMMARY ---')
    otc_g  = geometry_df[geometry_df['group'] == 'OTC']
    ctrl_g = geometry_df[geometry_df['group'] == 'control']

    for label_str, grp in [('OTC', otc_g), ('Controls', ctrl_g)]:
        print(f'\nGeometry preservation ({label_str}):')
        for cat in CATEGORIES:
            vals = grp[grp['category'] == cat]['geometry_preservation']
            if len(vals) > 0:
                print(f'  {cat}: {vals.mean():.3f} ± {vals.std():.3f} (n={len(vals)})')

    # Relocation–geometry correlation
    merged = spatial_df.merge(
        geometry_df,
        on=['subject_id', 'hemi', 'category', 'group',
            'cat_type', 'surgery_side', 'status', 'subject'],
        suffixes=('_sp', '_gm'))
    otc_m = merged[merged['group'] == 'OTC']
    if len(otc_m) > 3:
        rho, p = spearmanr(otc_m['relocation_mm'], otc_m['geometry_preservation'])
        print(f'\n  Relocation ↔ geometry (OTC): ρ={rho:.3f}, p={p:.4f}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Geometry preservation analysis')
    parser.add_argument('--cope-set', type=str, default='all',
                        choices=['differential', 'cat_vs_scramble', 'hybrid', 'all'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subs   = load_subjects()
    #subs = {'sub-108': subs['sub-108']}
    n_pt   = sum(1 for v in subs.values() if v['patient_status'] == 'patient')
    n_ctrl = sum(1 for v in subs.values() if v['patient_status'] == 'control')
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Skipping: {SUBJECTS_TO_SKIP}')

    cope_sets = COPE_MAPS if args.cope_set == 'all' else {args.cope_set: COPE_MAPS[args.cope_set]}

    for name, copes in cope_sets.items():
        _CACHE.clear()
        spatial_df, geometry_df, mds_df = run_pipeline(copes, name, subs)

        spatial_df.to_csv(OUTPUT_DIR  / f'spatial_{name}.csv',  index=False)
        geometry_df.to_csv(OUTPUT_DIR / f'geometry_{name}.csv', index=False)
        mds_df.to_csv(OUTPUT_DIR      / f'mds_{name}.csv',      index=False)

        print_summary(spatial_df, geometry_df, mds_df, name)
        print(f'  Saved to {OUTPUT_DIR}/')

    print('\nDone!')


if __name__ == '__main__':
    main()