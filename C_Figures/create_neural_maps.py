#!/usr/bin/env python3
"""
create_neural_maps.py — Generate 2D neural heatmaps for Ayzenberg-style figure.

Adapted from Ayzenberg et al. hemispace repo for sym_pt directory structure.

Steps:
  1. create_sub_map()   — per-subject 2D projection of zstat within ROI mask
  2. create_group_map() — sum binary maps across controls → proportion map

Output:
  {processed_dir}/group_results/neural_map/{cond}_binary.npy  (group heatmap)
  {sub_dir}/derivatives/neural_map/{cond}_func.npy            (per-subject)
  {sub_dir}/derivatives/neural_map/{cond}_binary.npy          (per-subject binary)

Usage:
  python B_analyses/create_neural_maps.py
  python B_analyses/create_neural_maps.py --sub 004
"""

import os, sys, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import image

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions, get_sub_info, _load_csv

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR    = Path(processed_dir)
OUTPUT_DIR  = BASE_DIR / 'group_results' / 'neural_map'
THRESH_Z    = 2.3

# Category → (cope number, ROI type)
COND_INFO = {
    'face':   {'cope': 1, 'roi': 'ventral'},
    'word':   {'cope': 4, 'roi': 'ventral'},
    'house':  {'cope': 2, 'roi': 'ventral'},
    'object': {'cope': 3, 'roi': 'ventral'},
}

# Ventral ROI mask — update path if different
VENTRAL_ROI = Path('/opt/fsl/6.0.3/data/atlases/MNI/MNI-maxprob-thr25-2mm.nii.gz')
# Fall back to MNI brain mask if ventral ROI not available
MNI_MASK    = Path('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')


def get_roi_mask():
    """Load ventral ROI if available, otherwise use full MNI brain mask."""
    if VENTRAL_ROI.exists():
        roi = image.load_img(str(VENTRAL_ROI))
        return image.math_img('img > 0', img=roi)
    elif MNI_MASK.exists():
        print('  WARNING: ventral ROI not found, using full MNI brain mask')
        return image.load_img(str(MNI_MASK))
    else:
        raise FileNotFoundError('No ROI mask found — check VENTRAL_ROI and MNI_MASK paths')


def get_zstat_path(sub_id, ses, first_ses, cope):
    """Return path to registered zstat for a subject/session/cope."""
    feat = (BASE_DIR / sub_id / f'ses-{ses}' / 'derivatives' / 'fsl' /
            'loc' / 'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats')
    # Registered to first session space
    for name in [f'zstat1_ses{first_ses}.nii.gz', 'zstat1.nii.gz']:
        p = feat / name
        if p.exists():
            return p
    return None


def create_sub_map(subjects=None):
    """Create per-subject 2D neural maps (func + binary) for each condition."""
    print('\nCreating individual subject maps...')
    df       = _load_csv()
    roi_mask = get_roi_mask()

    if subjects is None:
        subjects = sorted(
            d.replace('sub-', '') for d in os.listdir(BASE_DIR)
            if d.startswith('sub-') and d.replace('sub-', '') not in skip_subs
        )

    for sub_clean in subjects:
        sub_id   = f'sub-{sub_clean}'
        sessions = get_sessions(sub_clean)
        if not sessions:
            continue
        first_ses = f'{sessions[0]:02d}'
        # Use first post-surgical session
        ses       = first_ses

        out_dir = BASE_DIR / sub_id / f'ses-{ses}' / 'derivatives' / 'neural_map'
        out_dir.mkdir(parents=True, exist_ok=True)

        for cond, info in COND_INFO.items():
            zstat_path = get_zstat_path(sub_id, ses, first_ses, info['cope'])
            if zstat_path is None:
                print(f'  SKIP {sub_clean} {cond}: zstat not found')
                continue

            print(f'  {sub_clean} {cond}')

            # Load and threshold zstat
            zstat    = image.load_img(str(zstat_path))
            zstat_thr = image.threshold_img(zstat, threshold=THRESH_Z,
                                             two_sided=False)

            # Resample ROI mask to zstat space and apply
            roi_res  = image.resample_to_img(roi_mask, zstat_thr,
                                              interpolation='nearest')
            zstat_roi = image.math_img('img1 * img2',
                                        img1=zstat_thr, img2=roi_res)

            func_np  = zstat_roi.get_fdata()

            # 2D projection: max across z-axis, then transpose → (y, x)
            func_2d  = np.transpose(np.max(func_np, axis=2))

            # Binary version
            binary_2d        = np.zeros_like(func_2d)
            binary_2d[func_2d > 0] = 1

            np.save(str(out_dir / f'{cond}_func.npy'),   func_2d)
            np.save(str(out_dir / f'{cond}_binary.npy'), binary_2d)

    print('  Done: individual maps')


def create_group_map():
    """Sum binary maps across controls → group proportion heatmap."""
    print('\nCreating group maps...')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df           = _load_csv()
    control_subs = df[df['group'] == 'control']['sub_clean'].unique()

    for cond in COND_INFO:
        func_list   = []
        binary_list = []

        for sub_clean in sorted(control_subs):
            sub_id    = f'sub-{sub_clean}'
            sessions  = get_sessions(sub_clean)
            if not sessions:
                continue
            ses     = f'{sessions[0]:02d}'
            map_dir = BASE_DIR / sub_id / f'ses-{ses}' / 'derivatives' / 'neural_map'
            fp      = map_dir / f'{cond}_func.npy'
            bp      = map_dir / f'{cond}_binary.npy'

            if not fp.exists():
                continue

            func   = np.load(str(fp))
            binary = np.load(str(bp))

            # Normalise func to [0, 1] per subject
            mx = np.max(func)
            if mx > 0:
                func = func / mx
            func_list.append(func)
            binary_list.append(binary)

        if not func_list:
            print(f'  SKIP {cond}: no control maps found')
            continue

        func_group   = np.nanmean(func_list,  axis=0)
        binary_group = np.nansum(binary_list, axis=0)

        np.save(str(OUTPUT_DIR / f'{cond}_func.npy'),   func_group)
        np.save(str(OUTPUT_DIR / f'{cond}_binary.npy'), binary_group)
        print(f'  Saved: {cond}_binary.npy  (n={len(binary_list)} controls)')

    print('  Done: group maps')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None,
                        help='Single subject (e.g. 004)')
    parser.add_argument('--group-only', action='store_true',
                        help='Skip sub maps, only recompute group map')
    args = parser.parse_args()

    subs = [args.sub] if args.sub else None

    if not args.group_only:
        create_sub_map(subjects=subs)

    if args.sub is None:  # only create group map when running all subjects
        create_group_map()


if __name__ == '__main__':
    main()