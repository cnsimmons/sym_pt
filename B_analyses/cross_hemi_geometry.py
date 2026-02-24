#!/usr/bin/env python3
"""
compute_cross_hemisphere_geometry.py
────────────────────────────────────
For each CONTROL subject × category:
  1. Localize category-selective ROI independently in left and right hemispheres
  2. Build 6mm sphere at each hemisphere's peak/centroid
  3. Extract 4-category beta patterns (copes 15-18) within each sphere
  4. Compute 4×4 RDM (1 - Pearson r) for each hemisphere
  5. Correlate upper triangles of left and right RDMs
     → cross-hemisphere RDM similarity

This tests whether symmetric categories (house, object) have more bilaterally
consistent representational geometry than asymmetric categories (face, word).

Rationale:
  The vulnerability argument in the paper claims that symmetric categories
  depend on bilateral representation, so losing one hemisphere's contribution
  (plus competitive pressure from reorganizing categories) disrupts the
  surviving representation's geometry. This analysis provides the empirical
  foundation: if symmetric categories have genuinely more correlated bilateral
  geometry in controls, they have a structural property that makes them
  vulnerable when that bilateral architecture is disrupted.

Output: {processed_dir}/group_results/geometry/cross_hemisphere_rdm.csv

Usage:
  python compute_cross_hemisphere_geometry.py
"""

import os, sys, gc
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy.ndimage import label, center_of_mass

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

BASE_DIR   = Path(processed_dir)
OUTPUT_DIR = BASE_DIR / 'group_results' / 'geometry'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same exclusions as geometry preservation pipeline
SUBJECTS_TO_SKIP = ['OTC108', 'control083', 'control085', 'OTC017']

RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}
LOC_COPES = {'face': 1, 'house': 2, 'object': 3, 'word': 4}
CATEGORIES    = ['face', 'house', 'object', 'word']
BILATERAL     = ['house', 'object']

THRESHOLD_Z   = 1.96
TOP_PCT       = 0.10
MIN_VOXELS    = 50
SPHERE_RADIUS = 6


def load_nii(fp):
    return nib.load(str(fp))


def load_subjects():
    """Load subject info. Returns all subjects (controls will be filtered later)."""
    df = _load_csv()
    subjects = {}
    for sub_clean in sorted(df['sub_clean'].unique()):
        if sub_clean in skip_subs:
            continue
        sid = f'sub-{sub_clean}'
        sessions = get_sessions(sub_clean)
        if not sessions or not (BASE_DIR / sid).exists():
            continue
        info = get_sub_info(sub_clean, sessions[0])
        pt = is_patient(sub_clean)
        intact = info.get('intact_hemi', '')
        code = f"{info.get('group','')}{sub_clean}"
        if code in SUBJECTS_TO_SKIP:
            continue
        subjects[sid] = {
            'code':           code,
            'sessions':       [f'{s:02d}' for s in sessions],
            'hemi':           ('l' if intact == 'left' else 'r') if pt else None,
            'group':          info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi':    intact,
        }
    return subjects


def extract_roi(sub_id, session, category, hemi, first_ses):
    """Localize ROI using HighLevel zstat. Returns dict with centroid info or None.
    Mirrors the geometry preservation pipeline exactly."""
    cope_num = LOC_COPES[category]

    # Find searchmask
    mask_path = None
    for sd in ['ROIs', os.path.join('derivatives', 'rois')]:
        p = BASE_DIR / sub_id / f'ses-{first_ses}' / sd / f'{hemi}_{category}_searchmask.nii.gz'
        if p.exists():
            mask_path = p
            break
    if mask_path is None:
        return None

    mask_img = load_nii(mask_path)
    mask = mask_img.get_fdata() > 0
    affine = mask_img.affine

    bm_file = BASE_DIR / sub_id / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = load_nii(bm_file).get_fdata() > 0 if bm_file.exists() else None

    feat = BASE_DIR / sub_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf = feat / f'cope{cope_num}.feat' / 'stats' / zname
    if not zf.exists():
        return None

    z = load_nii(zf).get_fdata().copy()
    if bm is not None:
        z[~bm] = 0

    supra = (z > THRESHOLD_Z) & mask
    if supra.sum() < MIN_VOXELS:
        return None

    top_n = max(MIN_VOXELS, int(supra.sum() * TOP_PCT))
    top_n = min(top_n, supra.sum())
    thresh = np.sort(z[supra])[-top_n]
    top = (z >= thresh) & supra

    labeled, nc = label(top)
    if nc == 0:
        return None
    sizes = [(labeled == i).sum() for i in range(1, nc + 1)]
    li = np.argmax(sizes) + 1
    roi = (labeled == li)

    return {
        'centroid':    nib.affines.apply_affine(affine, np.array(center_of_mass(roi))),
        'affine':      affine,
        'brain_shape': z.shape,
    }


def create_sphere(peak_coord, affine, brain_shape, radius=SPHERE_RADIUS):
    """Create spherical mask. Matches geometry preservation pipeline."""
    grid = np.array(np.meshgrid(
        np.arange(brain_shape[0]), np.arange(brain_shape[1]),
        np.arange(brain_shape[2]), indexing='ij'
    )).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    mask = np.zeros(brain_shape, dtype=bool)
    for c in grid[dists <= radius]:
        mask[c[0], c[1], c[2]] = True
    return mask


def extract_sphere_betas(sub_id, session, sphere_mask, first_ses):
    """Extract 4-category beta patterns from sphere. Returns (n_voxels × 4) array."""
    feat = BASE_DIR / sub_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'

    patterns = []
    valid_cats = []
    for cat in CATEGORIES:
        cf = feat / f'cope{RSA_COPES[cat]}.feat' / 'stats' / cn
        if not cf.exists():
            return None, None
        betas = load_nii(cf).get_fdata()[sphere_mask]
        betas = betas[np.isfinite(betas)]
        if len(betas) == 0:
            return None, None
        patterns.append(betas)
        valid_cats.append(cat)

    if len(patterns) < 4:
        return None, None

    min_v = min(len(b) for b in patterns)
    return np.column_stack([b[:min_v] for b in patterns]), valid_cats


def compute_rdm(betas):
    """Compute 4×4 representational dissimilarity matrix (1 - Pearson r)."""
    n_cats = betas.shape[1]
    rdm = np.zeros((n_cats, n_cats))
    for i in range(n_cats):
        for j in range(n_cats):
            if i == j:
                rdm[i, j] = 0.0
            else:
                r, _ = pearsonr(betas[:, i], betas[:, j])
                rdm[i, j] = 1 - r
    return rdm


def rdm_correlation(rdm1, rdm2):
    """Correlate upper triangles of two RDMs."""
    triu = np.triu_indices(rdm1.shape[0], k=1)
    v1 = rdm1[triu]
    v2 = rdm2[triu]
    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan
    r, _ = pearsonr(v1, v2)
    return r


def main():
    subjects = load_subjects()

    # Filter to controls only (patients don't have two hemispheres)
    controls = {sid: info for sid, info in subjects.items()
                if info['patient_status'] == 'control'}
    print(f'Controls: {len(controls)}')

    rows = []

    for sid, info in sorted(controls.items()):
        first_ses = info['sessions'][0]
        session = first_ses  # Use first session for cross-hemisphere comparison

        print(f'\n{info["code"]} ses-{session}:')

        for category in CATEGORIES:
            # Extract ROI in both hemispheres independently
            roi_l = extract_roi(sid, session, category, 'l', first_ses)
            roi_r = extract_roi(sid, session, category, 'r', first_ses)

            if roi_l is None or roi_r is None:
                print(f'  {category}: missing ROI (L={roi_l is not None}, R={roi_r is not None})')
                continue

            # Build spheres
            sphere_l = create_sphere(roi_l['centroid'], roi_l['affine'], roi_l['brain_shape'])
            sphere_r = create_sphere(roi_r['centroid'], roi_r['affine'], roi_r['brain_shape'])

            # Extract beta patterns
            betas_l, cats_l = extract_sphere_betas(sid, session, sphere_l, first_ses)
            betas_r, cats_r = extract_sphere_betas(sid, session, sphere_r, first_ses)

            if betas_l is None or betas_r is None:
                print(f'  {category}: beta extraction failed')
                continue

            if cats_l != cats_r:
                print(f'  {category}: category mismatch L vs R')
                continue

            # Compute RDMs
            rdm_l = compute_rdm(betas_l)
            rdm_r = compute_rdm(betas_r)

            # Cross-hemisphere RDM correlation
            r_cross = rdm_correlation(rdm_l, rdm_r)

            cat_type = 'symmetric' if category in BILATERAL else 'asymmetric'

            rows.append({
                'subject':       info['code'],
                'subject_id':    sid,
                'session':       session,
                'category':      category,
                'cat_type':      cat_type,
                'cross_hemi_r':  r_cross,
                'n_voxels_l':    int(sphere_l.sum()),
                'n_voxels_r':    int(sphere_r.sum()),
            })

            r_str = f'{r_cross:.3f}' if np.isfinite(r_cross) else 'nan'
            print(f'  {category} ({cat_type}): cross-hemi RDM r={r_str}  '
                  f'(L:{sphere_l.sum()} vox, R:{sphere_r.sum()} vox)')

        gc.collect()

    df = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / 'cross_hemisphere_rdm.csv'
    df.to_csv(out_file, index=False)
    print(f'\nSaved: {out_file}')
    print(f'Total rows: {len(df)}')
    print(f'Subjects: {df["subject"].nunique()}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY: Cross-hemisphere RDM similarity by category type')
    print('='*70)

    for ct in ['symmetric', 'asymmetric']:
        vals = df[df['cat_type'] == ct]['cross_hemi_r'].dropna()
        if len(vals) > 0:
            print(f'  {ct}: M={vals.mean():.3f}, SD={vals.std():.3f}, n={len(vals)}')

    print(f'\nBy category:')
    for cat in CATEGORIES:
        vals = df[df['category'] == cat]['cross_hemi_r'].dropna()
        if len(vals) > 0:
            ct = 'symmetric' if cat in BILATERAL else 'asymmetric'
            print(f'  {cat} ({ct}): M={vals.mean():.3f}, SD={vals.std():.3f}, n={len(vals)}')

    # Symmetric - asymmetric difference per subject
    print(f'\nPer-subject symmetric - asymmetric difference:')
    subj_diffs = []
    for sid in df['subject_id'].unique():
        sdf = df[df['subject_id'] == sid]
        sym_mean  = sdf[sdf['cat_type'] == 'symmetric']['cross_hemi_r'].mean()
        asym_mean = sdf[sdf['cat_type'] == 'asymmetric']['cross_hemi_r'].mean()
        if np.isfinite(sym_mean) and np.isfinite(asym_mean):
            diff = sym_mean - asym_mean
            subj_diffs.append(diff)
            print(f'  {sdf.iloc[0]["subject"]}: sym={sym_mean:.3f}, asym={asym_mean:.3f}, diff={diff:+.3f}')

    if subj_diffs:
        diffs = np.array(subj_diffs)
        print(f'\n  Mean difference: {diffs.mean():.3f} (SD={diffs.std():.3f})')
        print(f'  Subjects with sym > asym: {(diffs > 0).sum()}/{len(diffs)}')

        # Quick bootstrap CI
        rng = np.random.default_rng(42)
        boot_means = [diffs[rng.choice(len(diffs), len(diffs), replace=True)].mean()
                      for _ in range(100000)]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f'  Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')


if __name__ == '__main__':
    main()