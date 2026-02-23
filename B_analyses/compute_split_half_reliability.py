#!/usr/bin/env python3
"""
compute_split_half_reliability.py
─────────────────────────────────
For each subject × session × category × hemisphere:
  1. Load sphere ROI (same as geometry/liu pipeline)
  2. Extract beta patterns from copes 15-18 in run-1 and run-2
  3. Correlate within-category patterns across runs → split-half reliability

Output: {processed_dir}/group_results/geometry/split_half_reliability.csv

Usage:
  python compute_split_half_reliability.py
"""

import os, sys, argparse
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

SUBJECTS_TO_SKIP = ['OTC108', 'control083', 'control085', 'OTC017']

PRE_SURGERY_SESSIONS = {
    'sub-017': ['01'], 'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'],
    'sub-049': ['01'], 'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'],
    'sub-086': ['01'],
}

# RSA betas — same copes as geometry pipeline
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

# Localizer copes for finding ROI peaks
LOC_COPES = {'face': 1, 'house': 2, 'object': 3, 'word': 4}

CATEGORIES = ['face', 'house', 'object', 'word']
BILATERAL  = ['house', 'object']

THRESHOLD_Z   = 1.96
TOP_PCT       = 0.10
MIN_VOXELS    = 50
SPHERE_RADIUS = 6

_CACHE = {}
def _load(fp):
    k = str(fp)
    if k not in _CACHE:
        _CACHE[k] = nib.load(k)
    return _CACHE[k]


def load_subjects():
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
            'surgery_side':   ('right' if intact == 'left' else 'left') if pt else 'na',
        }
    return subjects


def find_runs(sub_id, session):
    """Find available run directories for a subject/session."""
    loc_dir = BASE_DIR / sub_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc'
    runs = []
    for d in sorted(loc_dir.iterdir()) if loc_dir.exists() else []:
        if d.name.startswith('run-') and (d / '1stLevel.feat' / 'stats').exists():
            runs.append(d.name)  # e.g. 'run-01'
    return runs


def extract_roi(sub_id, session, category, hemi, first_ses):
    """Localize ROI using HighLevel zstat (same as geometry pipeline).
    Returns sphere mask and peak info, or None."""
    cope_num = LOC_COPES[category]
    
    # Searchmask
    roi_dir = BASE_DIR / sub_id / f'ses-{first_ses}' / 'ROIs'
    mask_path = roi_dir / f'{hemi}_{category}_searchmask.nii.gz'
    if not mask_path.exists():
        return None

    mask_img = _load(mask_path)
    mask = mask_img.get_fdata() > 0
    affine = mask_img.affine

    # Brain mask
    bm_file = BASE_DIR / sub_id / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    # Zstat from HighLevel for localization
    feat = BASE_DIR / sub_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf = feat / f'cope{cope_num}.feat' / 'stats' / zname
    if not zf.exists():
        return None

    z = _load(zf).get_fdata().copy()
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
    pidx = np.unravel_index(np.argmax(z * roi), z.shape)
    peak_coord = nib.affines.apply_affine(affine, np.array(pidx))

    # Build sphere
    grid = np.array(np.meshgrid(
        np.arange(z.shape[0]), np.arange(z.shape[1]),
        np.arange(z.shape[2]), indexing='ij'
    )).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    sphere = np.zeros(z.shape, dtype=bool)
    for c in grid[dists <= SPHERE_RADIUS]:
        sphere[c[0], c[1], c[2]] = True

    return sphere


def extract_run_pattern(sub_id, session, run_name, sphere, first_ses):
    """Extract beta pattern (copes 15-18) from a single run within the sphere."""
    run_stats = (BASE_DIR / sub_id / f'ses-{session}' / 'derivatives' / 
                 'fsl' / 'loc' / run_name / '1stLevel.feat' / 'stats')
    
    patterns = []
    valid_cats = []
    for cat in CATEGORIES:
        cope_file = run_stats / f'cope{RSA_COPES[cat]}.nii.gz'
        if not cope_file.exists():
            return None, None
        betas = _load(cope_file).get_fdata()[sphere]
        betas = betas[np.isfinite(betas)]
        if len(betas) == 0:
            return None, None
        patterns.append(betas)
        valid_cats.append(cat)

    if len(patterns) != 4:
        return None, None

    min_v = min(len(b) for b in patterns)
    return np.column_stack([b[:min_v] for b in patterns]), valid_cats


def compute_split_half(pattern_run1, pattern_run2, cat_idx):
    """Correlate within-category voxel patterns across two runs."""
    p1 = pattern_run1[:, cat_idx]
    p2 = pattern_run2[:, cat_idx]
    if len(p1) < 3 or np.std(p1) == 0 or np.std(p2) == 0:
        return np.nan
    r, _ = pearsonr(p1, p2)
    return r


def main():
    subjects = load_subjects()
    print(f'Loaded {len(subjects)} subjects')
    
    rows = []
    
    for sid, info in subjects.items():
        sessions = info['sessions']
        
        # Skip pre-surgical sessions
        pre = PRE_SURGERY_SESSIONS.get(sid, [])
        sessions = [s for s in sessions if s not in pre]
        if not sessions:
            continue
        
        # Use first valid session
        session = sessions[0]
        first_ses = info['sessions'][0]
        if first_ses in pre and len(info['sessions']) > 1:
            first_ses = [s for s in info['sessions'] if s not in pre][0]
        
        # Determine hemispheres
        if info['patient_status'] == 'patient':
            hemis = [info['hemi']]
            hemi_labels = ['intact']
        else:
            hemis = ['l', 'r']
            hemi_labels = ['left', 'right']
        
        # Find runs
        runs = find_runs(sid, session)
        if len(runs) < 2:
            print(f'  {sid} ses-{session}: <2 runs, skipping')
            continue
        
        print(f'{info["code"]} ses-{session}: {len(runs)} runs')
        
        for hemi, hl in zip(hemis, hemi_labels):
            for category in CATEGORIES:
                # Localize ROI
                sphere = extract_roi(sid, session, category, hemi, first_ses)
                if sphere is None:
                    print(f'  {category} {hemi}: no ROI')
                    continue
                
                # Extract patterns from first two runs
                p1, cats1 = extract_run_pattern(sid, session, runs[0], sphere, first_ses)
                p2, cats2 = extract_run_pattern(sid, session, runs[1], sphere, first_ses)
                
                if p1 is None or p2 is None:
                    print(f'  {category} {hemi}: pattern extraction failed')
                    continue
                
                cat_idx = CATEGORIES.index(category)
                r = compute_split_half(p1, p2, cat_idx)
                
                cat_type = 'symmetric' if category in BILATERAL else 'asymmetric'
                
                rows.append({
                    'subject':      info['code'],
                    'subject_id':   sid,
                    'group':        info['group'] if info['patient_status'] == 'patient' else 'control',
                    'status':       info['patient_status'],
                    'surgery_side': info['surgery_side'],
                    'session':      session,
                    'hemi':         'left' if hemi == 'l' else 'right',
                    'hemi_label':   hl,
                    'category':     category,
                    'cat_type':     cat_type,
                    'split_half_r': r,
                    'n_voxels':     sphere.sum(),
                })
                
                r_str = f'{r:.3f}' if np.isfinite(r) else 'nan'
                print(f'  {category} {hemi}: r={r_str} ({sphere.sum()} voxels)')
    
    df = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / 'split_half_reliability.csv'
    df.to_csv(out_file, index=False)
    print(f'\nSaved: {out_file}')
    print(f'Total rows: {len(df)}')
    print(f'Patients: {df[df["status"]=="patient"]["subject"].nunique()}')
    print(f'Controls: {df[df["status"]=="control"]["subject"].nunique()}')


if __name__ == '__main__':
    main()