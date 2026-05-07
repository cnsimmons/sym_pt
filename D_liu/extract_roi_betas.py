#!/usr/bin/env python3
"""
extract_roi_betas.py

Extracts run-level betas within primary ROI spheres (FFA, PPA, LOC, VWFA).
Cross-sectional sample: controls = first session, OTC patients = last session.
Native (ses-01 anat) space, 7mm spheres around peaks ∩ anatomical searchmasks.

Outputs (in OUT_DIR):
  - roi_per_condition_responses.csv : 1 row per (sub, ROI, hemi, condition)
      Used for: rank-order test (does FFA still respond face > others?)
  - roi_split_half_loro.csv : 1 row per (sub, ROI, hemi, c1, c2)
      Used for: distinctiveness decomposition (within vs between split-half r)

Usage:
    python extract_roi_betas.py
    python extract_roi_betas.py --peak-csv /path/to/alt.csv
"""
import os, sys, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
                           is_patient, get_sub_info, _load_csv, get_runs)

# --- Constants -----------------------------------------------------------------
PRIMARY_ROIS = {                          # ROI label : anatomical searchmask category
    'face_FFA':   'face',
    'house_PPA':  'house',
    'object_LOC': 'object',
    'word_VWFA':  'word',
}
RAW_BETA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}
CATEGORIES = ['face', 'house', 'object', 'word']
SPHERE_RADIUS_MM = 7.0

EXTRA_SKIP = {'sub-017', 'control083', 'control085'}
EXCLUDE_SES = {('sub-108', 2)}            # match cross-sectional.ipynb

OUT_DIR = Path(processed_dir) / 'group_results' / 'roi_betas'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Subject selection ---------------------------------------------------------
def load_subjects():
    """Controls: first session. OTC patients: last session. NonOTC excluded."""
    df = _load_csv()
    subjects = {}
    for sc in sorted(df['sub_clean'].unique()):
        if sc in skip_subs: continue
        sid = f'sub-{sc}'
        if sid in EXTRA_SKIP: continue
        sessions = get_sessions(sc)
        if not sessions or not os.path.exists(os.path.join(processed_dir, sid)):
            continue
        sessions = [s for s in sessions if (sid, s) not in EXCLUDE_SES]
        if not sessions: continue

        info0 = get_sub_info(sc, sessions[0])
        group = info0.get('group', 'unknown')
        if group == 'nonOTC': continue
        pt = is_patient(sc)

        pick = sessions[-1] if pt else sessions[0]   # last for patients, first for controls
        ses_str = f'{pick:02d}'
        info = get_sub_info(sc, pick)
        intact = info.get('intact_hemi', '')

        subjects[sid] = {
            'session': ses_str,
            'group': group,
            'is_patient': pt,
            'intact_hemi': intact,
            'hemis': ['l', 'r'] if not pt else [('l' if intact == 'left' else 'r')],
        }
    return subjects


# --- Path helpers --------------------------------------------------------------
def get_run_beta_path(sub_id, ses, run, cope_num):
    return os.path.join(processed_dir, sub_id, f'ses-{ses}',
                        'derivatives', 'fsl', 'loc', f'run-{run:02d}',
                        '1stLevel.feat', 'reg_standard', 'stats',
                        f'cope{cope_num}.nii.gz')

def get_searchmask_path(sub_id, ses, hemi, roi_label):
    """ROI-specific searchmask, e.g. l_face_FFA_searchmask.nii.gz"""
    return os.path.join(processed_dir, sub_id, f'ses-{ses}', 'ROIs',
                        f'{hemi}_{roi_label}_searchmask.nii.gz')


# --- Sphere construction -------------------------------------------------------
def build_sphere_mask(shape, affine, peak_mm, radius_mm):
    """Boolean 3D mask of voxels whose centers lie within radius_mm of peak_mm.
    peak_mm is (x, y, z) in world (mm) coords. Uses affine to convert each voxel
    center to mm and compute Euclidean distance."""
    nx, ny, nz = shape
    i, j, k = np.indices((nx, ny, nz))
    coords = np.stack([i.ravel(), j.ravel(), k.ravel(),
                       np.ones(i.size)], axis=0)
    mm = (affine @ coords).T[:, :3]
    diffs = mm - np.asarray(peak_mm).reshape(1, 3)
    inside = (diffs ** 2).sum(axis=1) <= radius_mm ** 2
    return inside.reshape(shape)


# --- Main ----------------------------------------------------------------------
def main(peak_csv):
    print(f'Peak CSV: {peak_csv}')
    coords = pd.read_csv(peak_csv)
    coords = coords[coords['category'].isin(PRIMARY_ROIS.keys())].copy()
    coords['ses_str'] = coords['session'].astype(int).astype(str).str.zfill(2)
    print(f'  Filtered to primary ROIs: {len(coords)} rows')

    subjects = load_subjects()
    print(f'Loaded {len(subjects)} subjects')

    rows_response, rows_split, skipped = [], [], []

    for sid, info in subjects.items():
        ses = info['session']
        sub_clean = sid.replace('sub-', '')
        runs = get_runs(sid, int(ses))
        if len(runs) < 2:
            skipped.append((sid, ses, 'fewer than 2 runs'))
            continue

        # Pre-load all run × condition copes (no resampling — assumes cope and
        # searchmask are in the same native anat space, matching voxel_split_half_native_liu.ipynb).
        run_copes = {}
        cope_shape, cope_affine = None, None
        ok = True
        for r in runs:
            run_copes[r] = {}
            for cat, cope_num in RAW_BETA_COPES.items():
                p = get_run_beta_path(sid, ses, r, cope_num)
                if not os.path.exists(p):
                    skipped.append((sid, ses, f'missing run-{r:02d} cope{cope_num}'))
                    ok = False; break
                img = nib.load(p)
                run_copes[r][cat] = img.get_fdata()
                if cope_shape is None:
                    cope_shape, cope_affine = img.shape, img.affine
            if not ok: break
        if not ok: continue

        for h in info['hemis']:
            for roi_label, anat_cat in PRIMARY_ROIS.items():
                row = coords[(coords['subject_id'] == sid) &
                             (coords['ses_str'] == ses) &
                             (coords['hemi'] == h) &
                             (coords['category'] == roi_label)]
                if not len(row):
                    skipped.append((sid, ses, h, roi_label, 'no peak in CSV'))
                    continue
                peak = (row['peak_x_native'].iloc[0],
                        row['peak_y_native'].iloc[0],
                        row['peak_z_native'].iloc[0])
                if any(np.isnan(p) for p in peak):
                    skipped.append((sid, ses, h, roi_label, 'peak NaN'))
                    continue

                sm_path = get_searchmask_path(sid, ses, h, roi_label)
                if not os.path.exists(sm_path):
                    skipped.append((sid, ses, h, roi_label, 'no searchmask'))
                    continue
                sm = nib.load(sm_path).get_fdata() > 0
                if sm.shape != cope_shape:
                    skipped.append((sid, ses, h, roi_label,
                                    f'shape mismatch sm={sm.shape} cope={cope_shape}'))
                    continue

                sphere = build_sphere_mask(cope_shape, cope_affine, peak, SPHERE_RADIUS_MM)
                mask = sm & sphere
                n_vox = int(mask.sum())
                if n_vox < 5:
                    skipped.append((sid, ses, h, roi_label, f'mask too small: {n_vox}'))
                    continue

                # ---- Output 1: per-condition mean response (avg across all runs) ----
                # Compute per-run mean within mask, then average across runs.
                for cat in CATEGORIES:
                    per_run_means = [run_copes[r][cat][mask].mean() for r in runs]
                    rows_response.append({
                        'subject_id': sid, 'session': ses, 'group': info['group'],
                        'is_patient': info['is_patient'],
                        'intact_hemi': info['intact_hemi'],
                        'hemi': h, 'roi': roi_label, 'condition': cat,
                        'mean_response': float(np.mean(per_run_means)),
                        'sd_across_runs': float(np.std(per_run_means, ddof=1)) if len(runs) > 1 else np.nan,
                        'n_voxels': n_vox, 'n_runs': len(runs),
                    })

                # ---- Output 2: LORO split-half within & between correlations ----
                # Cocktail-blank demean per half (standard for multivariate split-half).
                loro = {(c1, c2): [] for c1 in CATEGORIES for c2 in CATEGORIES}
                for r_held in runs:
                    others = [r for r in runs if r != r_held]
                    h1 = {c: run_copes[r_held][c][mask] for c in CATEGORIES}
                    h2 = {c: np.mean([run_copes[r][c][mask] for r in others], axis=0)
                          for c in CATEGORIES}
                    for halves in (h1, h2):
                        mat = np.column_stack([halves[c] for c in CATEGORIES])
                        mat -= mat.mean(axis=1, keepdims=True)
                        for i_c, c in enumerate(CATEGORIES):
                            halves[c] = mat[:, i_c]
                    for c1 in CATEGORIES:
                        for c2 in CATEGORIES:
                            v1, v2 = h1[c1], h2[c2]
                            if v1.std() == 0 or v2.std() == 0:
                                loro[(c1, c2)].append(np.nan)
                            else:
                                loro[(c1, c2)].append(np.corrcoef(v1, v2)[0, 1])
                for c1 in CATEGORIES:
                    for c2 in CATEGORIES:
                        vals = np.array(loro[(c1, c2)])
                        rows_split.append({
                            'subject_id': sid, 'session': ses, 'group': info['group'],
                            'is_patient': info['is_patient'],
                            'intact_hemi': info['intact_hemi'],
                            'hemi': h, 'roi': roi_label,
                            'cat_1': c1, 'cat_2': c2,
                            'pair_type': 'within' if c1 == c2 else 'between',
                            'r_loro_mean': float(np.nanmean(vals)),
                            'n_loro_iters': int(np.sum(~np.isnan(vals))),
                            'n_voxels': n_vox, 'n_runs': len(runs),
                        })

    response_df = pd.DataFrame(rows_response)
    split_df = pd.DataFrame(rows_split)
    out_resp = OUT_DIR / 'roi_per_condition_responses.csv'
    out_split = OUT_DIR / 'roi_split_half_loro.csv'
    response_df.to_csv(out_resp, index=False)
    split_df.to_csv(out_split, index=False)

    print(f'\nWrote: {out_resp}  ({len(response_df)} rows)')
    print(f'Wrote: {out_split}  ({len(split_df)} rows)')
    if skipped:
        print(f'\nSkipped {len(skipped)} entries (first 25):')
        for s in skipped[:25]: print(f'  {s}')
        if len(skipped) > 25: print(f'  ... and {len(skipped)-25} more')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--peak-csv',
        default='/user_data/csimmon2/git_repos/sym_pt/D_liu/liu_exact_replication_v2.csv',
        help='Source of peak coordinates.'
    )
    args = parser.parse_args()
    main(args.peak_csv)