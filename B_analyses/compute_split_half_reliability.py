#!/usr/bin/env python3
"""
compute_split_half_reliability.py
─────────────────────────────────
For each subject × session × category × hemisphere:
  1. Load sphere ROI (same as geometry/liu pipeline)
  2. Extract beta patterns from copes 15-18 in run-1 and run-2
  3. Correlate within-category patterns across runs → split-half reliability

UPDATE: Now loops over ALL available post-surgical sessions (not just session 1).
  This is critical because the geometry preservation analysis compares the first
  and last post-surgical sessions. If split-half reliability is equivalent at both
  timepoints, the argument that the geometry deficit reflects temporal change
  (rather than noise) is substantially stronger.

Output: {processed_dir}/group_results/geometry/split_half_reliability.csv

Usage:
  python compute_split_half_reliability.py
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

SUBJECTS_TO_SKIP = ['OTC108', 'control083', 'control085', 'OTC017']

PRE_SURGERY_SESSIONS = {
    'sub-017': ['01'], 'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'],
    'sub-049': ['01'], 'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'],
    'sub-086': ['01'],
}

RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}
LOC_COPES = {'face': 1, 'house': 2, 'object': 3, 'word': 4}
CATEGORIES = ['face', 'house', 'object', 'word']
BILATERAL  = ['house', 'object']

THRESHOLD_Z   = 1.96
TOP_PCT       = 0.10
MIN_VOXELS    = 50
SPHERE_RADIUS = 6


def load_nii(fp):
    """Load NIfTI without caching."""
    return nib.load(str(fp))


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
        if d.name.startswith('run-') and (d / '1stLevel.feat' / 'reg_standard' / 'stats').exists():
            runs.append(d.name)
    return runs


def extract_roi(sub_id, session, category, hemi, first_ses):
    """Localize ROI using HighLevel zstat. Returns sphere mask or None."""
    cope_num = LOC_COPES[category]

    roi_dir = BASE_DIR / sub_id / f'ses-{first_ses}' / 'ROIs'
    mask_path = roi_dir / f'{hemi}_{category}_searchmask.nii.gz'
    if not mask_path.exists():
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
    pidx = np.unravel_index(np.argmax(z * roi), z.shape)
    peak_coord = nib.affines.apply_affine(affine, np.array(pidx))

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
    """Extract beta pattern (copes 15-18) from a single run within the sphere.
    Uses reg_standard/stats for registered cope maps."""
    run_stats = (BASE_DIR / sub_id / f'ses-{session}' / 'derivatives' /
                 'fsl' / 'loc' / run_name / '1stLevel.feat' / 'reg_standard' / 'stats')

    patterns = []
    for cat in CATEGORIES:
        cope_file = run_stats / f'cope{RSA_COPES[cat]}.nii.gz'
        if not cope_file.exists():
            return None, None
        betas = load_nii(cope_file).get_fdata()[sphere]
        betas = betas[np.isfinite(betas)]
        if len(betas) == 0:
            return None, None
        patterns.append(betas)

    if len(patterns) != 4:
        return None, None

    min_v = min(len(b) for b in patterns)
    return np.column_stack([b[:min_v] for b in patterns]), CATEGORIES


def compute_split_half(pattern_run1, pattern_run2, cat_idx):
    """Correlate within-category voxel patterns across two runs."""
    p1 = pattern_run1[:, cat_idx]
    p2 = pattern_run2[:, cat_idx]
    if len(p1) < 3 or np.std(p1) == 0 or np.std(p2) == 0:
        return np.nan
    r, _ = pearsonr(p1, p2)
    return r


def get_post_surgical_sessions(sid, info):
    """Return list of post-surgical sessions for a subject."""
    pre = PRE_SURGERY_SESSIONS.get(sid, [])
    return [s for s in info['sessions'] if s not in pre]


def determine_first_ses(sid, info):
    """Determine the first session to use for ROI masks (handles pre-surgical exclusion)."""
    pre = PRE_SURGERY_SESSIONS.get(sid, [])
    first_ses = info['sessions'][0]
    if first_ses in pre and len(info['sessions']) > 1:
        first_ses = [s for s in info['sessions'] if s not in pre][0]
    return first_ses


def main():
    subjects = load_subjects()
    print(f'Loaded {len(subjects)} subjects')

    rows = []

    for sid, info in sorted(subjects.items()):
        # Get all post-surgical sessions
        post_sessions = get_post_surgical_sessions(sid, info)
        if not post_sessions:
            continue

        first_ses = determine_first_ses(sid, info)

        # For controls, use all sessions; for patients, use post-surgical only
        sessions_to_process = post_sessions

        if info['patient_status'] == 'patient':
            hemis = [info['hemi']]
            hemi_labels = ['intact']
        else:
            hemis = ['l', 'r']
            hemi_labels = ['left', 'right']

        for session in sessions_to_process:
            runs = find_runs(sid, session)
            if len(runs) < 2:
                continue

            # Determine if this is the first or last post-surgical session
            # (for tagging in output — helps match to geometry analysis)
            session_rank = post_sessions.index(session)
            is_first_post = (session_rank == 0)
            is_last_post  = (session_rank == len(post_sessions) - 1)

            if is_first_post:
                session_label = 'first_post'
            elif is_last_post:
                session_label = 'last_post'
            else:
                session_label = 'intermediate'

            print(f'{info["code"]} ses-{session} ({session_label}): {len(runs)} runs')

            for hemi, hl in zip(hemis, hemi_labels):
                for category in CATEGORIES:
                    sphere = extract_roi(sid, session, category, hemi, first_ses)
                    if sphere is None:
                        continue

                    # Compute split-half for all available run pairs
                    # Primary: run-1 vs run-2 (most comparable to standard split-half)
                    p1, _ = extract_run_pattern(sid, session, runs[0], sphere, first_ses)
                    p2, _ = extract_run_pattern(sid, session, runs[1], sphere, first_ses)

                    if p1 is None or p2 is None:
                        continue

                    cat_idx = CATEGORIES.index(category)
                    r_12 = compute_split_half(p1, p2, cat_idx)

                    cat_type = 'symmetric' if category in BILATERAL else 'asymmetric'

                    row_base = {
                        'subject':        info['code'],
                        'subject_id':     sid,
                        'group':          info['group'] if info['patient_status'] == 'patient' else 'control',
                        'status':         info['patient_status'],
                        'surgery_side':   info['surgery_side'],
                        'session':        session,
                        'session_label':  session_label,
                        'session_rank':   session_rank,
                        'n_post_sessions': len(post_sessions),
                        'hemi':           'left' if hemi == 'l' else 'right',
                        'hemi_label':     hl,
                        'category':       category,
                        'cat_type':       cat_type,
                        'n_voxels':       int(sphere.sum()),
                    }

                    # Run 1 vs Run 2
                    rows.append({
                        **row_base,
                        'run_pair':      f'{runs[0]}_vs_{runs[1]}',
                        'split_half_r':  r_12,
                    })

                    r_str = f'{r_12:.3f}' if np.isfinite(r_12) else 'nan'
                    print(f'  {category} {hemi} {runs[0]}v{runs[1]}: r={r_str} ({sphere.sum()} vox)')

                    # If 3+ runs available, also compute run-1 vs run-3 and run-2 vs run-3
                    # for additional robustness (and average across pairs)
                    if len(runs) >= 3:
                        p3, _ = extract_run_pattern(sid, session, runs[2], sphere, first_ses)
                        if p3 is not None:
                            r_13 = compute_split_half(p1, p3, cat_idx)
                            r_23 = compute_split_half(p2, p3, cat_idx)

                            rows.append({**row_base, 'run_pair': f'{runs[0]}_vs_{runs[2]}', 'split_half_r': r_13})
                            rows.append({**row_base, 'run_pair': f'{runs[1]}_vs_{runs[2]}', 'split_half_r': r_23})

                            # Also store the mean across all pairs
                            pair_rs = [r for r in [r_12, r_13, r_23] if np.isfinite(r)]
                            if pair_rs:
                                rows.append({**row_base, 'run_pair': 'mean_all_pairs', 'split_half_r': np.mean(pair_rs)})

        gc.collect()

    df = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / 'split_half_reliability.csv'
    df.to_csv(out_file, index=False)
    print(f'\nSaved: {out_file}')
    print(f'Total rows: {len(df)}')
    print(f'Patients: {df[df["status"]=="patient"]["subject"].nunique()}')
    print(f'Controls: {df[df["status"]=="control"]["subject"].nunique()}')

    # ── Summary statistics ────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY: Split-half reliability by category type')
    print('='*70)

    # Filter to primary run pair (run-1 vs run-2) for clean summary
    primary = df[df['run_pair'].str.contains('run-01_vs_run-02') | 
                 df['run_pair'].str.contains('run-1_vs_run-2') |
                 (~df['run_pair'].str.contains('mean'))]
    # Safer: just use the first run pair per subject/session/category
    primary = df.groupby(['subject_id', 'session', 'hemi', 'category']).first().reset_index()

    for status in ['patient', 'control']:
        grp = primary[primary['status'] == status]
        if grp.empty:
            continue
        print(f'\n--- {status.upper()} ---')
        for ct in ['symmetric', 'asymmetric']:
            vals = grp[grp['cat_type'] == ct]['split_half_r'].dropna()
            if len(vals) > 0:
                print(f'  {ct}: M={vals.mean():.3f}, SD={vals.std():.3f}, n={len(vals)}')

        # By session label (for patients)
        if status == 'patient':
            print(f'\n  By session timepoint:')
            for sl in ['first_post', 'last_post', 'intermediate']:
                sl_grp = grp[grp['session_label'] == sl]
                if sl_grp.empty:
                    continue
                for ct in ['symmetric', 'asymmetric']:
                    vals = sl_grp[sl_grp['cat_type'] == ct]['split_half_r'].dropna()
                    if len(vals) > 0:
                        print(f'    {sl} / {ct}: M={vals.mean():.3f}, SD={vals.std():.3f}, n={len(vals)}')


if __name__ == '__main__':
    main()