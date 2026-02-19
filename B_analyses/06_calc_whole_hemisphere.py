#!/usr/bin/env python3
"""
06_calc_whole_hemisphere.py - Whole-hemisphere selectivity analysis

Following Ayzenberg et al. (2023) supplementary analysis:
  Measures summed selectivity across the entire hemisphere rather than
  within category-specific searchmasks. Serves as ROI-free check.

Approach:
  - Creates hemisphere masks by splitting T1w_brain_mask at the midline
  - Computes same metrics as 02_calc_summary_vals.py but within hemi masks
  - Same group/hemisphere filtering (patients=intact only, no pre-surgical)

Usage:
  python 06_calc_whole_hemisphere.py
  python 06_calc_whole_hemisphere.py --sub 004
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
                           is_patient, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────

THRESH = 2.3

# 06_calc_whole_hemisphere.py — replace CATEGORY_COPES
CATEGORY_COPES = {
    # original — unchanged
    'face':   1,
    'house':  2,
    'object': 3,
    'word':   4,
    # new: evc as whole-hemisphere baseline
    'evc':    3,
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_hemis_for_subject(sub_clean, ses):
    """Return hemispheres to analyze, group, and intact_hemi."""
    info = get_sub_info(sub_clean, ses)
    group = info.get('group', '')
    intact_hemi = info.get('intact_hemi', '')

    df = _load_csv()
    sub_rows = df[(df['sub_clean'] == sub_clean) & (df['ses_num'] == ses)]
    if not sub_rows.empty:
        pre_post = sub_rows.iloc[0].get('pre_post', 'na')
    else:
        pre_post = 'na'

    if group == 'control':
        return ['l', 'r'], group, 'control'
    if pre_post == 'pre':
        return [], group, intact_hemi
    if intact_hemi == 'left':
        return ['l'], group, intact_hemi
    elif intact_hemi == 'right':
        return ['r'], group, intact_hemi
    return [], group, intact_hemi


def get_or_create_hemi_mask(anat_dir, hemi):
    """
    Get or create a hemisphere brain mask by splitting T1w_brain_mask at midline.
    
    In native space, the midpoint along the x-axis splits hemispheres.
    For radiological convention (common in FSL): x < mid = right, x >= mid = left
    
    Returns path to hemisphere mask, or None if brain mask doesn't exist.
    """
    hemi_label = 'left' if hemi == 'l' else 'right'
    hemi_mask_path = f'{anat_dir}/T1w_brain_mask_{hemi_label}.nii.gz'

    if os.path.exists(hemi_mask_path):
        return hemi_mask_path

    brain_mask_path = f'{anat_dir}/T1w_brain_mask.nii.gz'
    if not os.path.exists(brain_mask_path):
        return None

    print(f'    Creating {hemi_label} hemisphere mask...')
    mask_img = nib.load(brain_mask_path)
    mask_data = mask_img.get_fdata().copy()

    mid_x = mask_data.shape[0] // 2

    hemi_data = np.zeros_like(mask_data)
    if hemi == 'l':
        # Left hemisphere = higher x indices in radiological
        hemi_data[mid_x:, :, :] = mask_data[mid_x:, :, :]
    elif hemi == 'r':
        hemi_data[:mid_x, :, :] = mask_data[:mid_x, :, :]

    out_img = nib.Nifti1Image(hemi_data.astype(np.float32), mask_img.affine)
    nib.save(out_img, hemi_mask_path)

    n_voxels = int(hemi_data.sum())
    print(f'    Saved: {hemi_mask_path} ({n_voxels:,} voxels)')

    return hemi_mask_path


def calc_summary_vals(zstat_path, mask_path, thresh=THRESH):
    """Same metric computation as 02_calc_summary_vals.py"""
    zstat_img = nib.load(zstat_path)
    zstat_data = zstat_img.get_fdata()

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0

    mask_size = int(mask_data.sum())
    if mask_size == 0:
        return mask_size, np.nan, 0.0, 0.0, 0.0

    masked_vals = zstat_data[mask_data]
    supra_vals = masked_vals[masked_vals > thresh]
    n_active = len(supra_vals)

    if n_active == 0:
        return mask_size, np.nan, 0.0, 0.0, 0.0

    vox_size = np.prod(zstat_img.header.get_zooms()[:3])

    mean_act = float(np.mean(supra_vals))
    volume = float(n_active * vox_size)
    sum_selec = float(np.sum(supra_vals))
    sum_selec_norm = float((sum_selec / mask_size) * 1000)

    return mask_size, mean_act, volume, sum_selec, sum_selec_norm


# ── Core ─────────────────────────────────────────────────────────────────────

def process_subject(sub_clean, ses, first_ses, thresh=THRESH, dry_run=False):
    """Compute whole-hemisphere selectivity for all categories."""
    ses_str = f'{ses:02d}'
    first_ses_str = f'{first_ses:02d}'

    base_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}'
    anat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/anat'
    gfeat_dir = f'{base_dir}/derivatives/fsl/loc/HighLevel.gfeat'

    hemis, group, intact_hemi = get_hemis_for_subject(sub_clean, ses)
    if not hemis:
        print(f'  SKIP: pre-surgical or no intact_hemi')
        return []

    results = []

    for category, cope_num in CATEGORY_COPES.items():
        zstat_path = (f'{gfeat_dir}/cope{cope_num}.feat/stats/'
                      f'zstat1_ses{first_ses_str}.nii.gz')
        if not os.path.exists(zstat_path):
            zstat_path = f'{gfeat_dir}/cope{cope_num}.feat/stats/zstat1.nii.gz'
            if not os.path.exists(zstat_path):
                continue

        for hemi in hemis:
            hemi_mask = get_or_create_hemi_mask(anat_dir, hemi)
            if hemi_mask is None:
                print(f'  SKIP {hemi}_{category}: no brain mask')
                continue

            if dry_run:
                print(f'  WOULD COMPUTE: {hemi}_{category} (whole hemi)')
                continue

            mask_size, mean_act, volume, sum_selec, sum_selec_norm = \
                calc_summary_vals(zstat_path, hemi_mask, thresh)

            hemi_full = 'left' if hemi == 'l' else 'right'

            results.append({
                'sub': f'sub-{sub_clean}',
                'ses': ses_str,
                'group': group,
                'intact_hemi': intact_hemi,
                'hemi': hemi_full,
                'category': category,
                'cope': cope_num,
                'mask_type': 'whole_hemi',
                'mask_size': mask_size,
                'mean_act': mean_act,
                'volume': volume,
                'sum_selec': sum_selec,
                'sum_selec_norm': sum_selec_norm,
            })

            if not np.isnan(mean_act):
                n_active = int(volume / np.prod(nib.load(zstat_path).header.get_zooms()[:3]))
                print(f'  {hemi}_{category}: {n_active} active voxels, '
                      f'sum_selec_norm={sum_selec_norm:.2f}')
            else:
                print(f'  {hemi}_{category}: no suprathreshold voxels')

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Whole-hemisphere selectivity')
    parser.add_argument('--sub', type=str, help='Single subject')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--threshold', type=float, default=THRESH)
    parser.add_argument('--suffix', type=str, default='_wholehemi')
    args = parser.parse_args()

    thresh = args.threshold

    print('=' * 60)
    print('WHOLE-HEMISPHERE SELECTIVITY ANALYSIS')
    print('=' * 60)
    print(f'Threshold: z > {thresh}')
    print(f'Mask: full hemisphere (split at midline)')
    print()

    if args.sub:
        subjects = [args.sub.replace('sub-', '')]
    else:
        subjects = sorted(
            d.replace('sub-', '')
            for d in os.listdir(processed_dir)
            if d.startswith('sub-') and d.replace('sub-', '') not in skip_subs
        )

    print(f'Processing {len(subjects)} subjects\n')
    all_results = []

    for sub_clean in subjects:
        sessions = get_sessions(sub_clean)
        if not sessions:
            continue
        first_ses = sessions[0]

        for ses in sessions:
            print(f'=== sub-{sub_clean} ses-{ses:02d} ===')
            results = process_subject(sub_clean, ses, first_ses, thresh, args.dry_run)
            all_results.extend(results)

    if not args.dry_run and all_results:
        df = pd.DataFrame(all_results)

        out_dir = f'{processed_dir}/group_results/selectivity'
        os.makedirs(out_dir, exist_ok=True)

        out_file = f'{out_dir}/selectivity_summary{args.suffix}.csv'
        df.to_csv(out_file, index=False)

        n_pt = df[df['group'] != 'control']['sub'].nunique()
        n_ctrl = df[df['group'] == 'control']['sub'].nunique()

        print()
        print('=' * 60)
        print(f'Saved: {out_file}')
        print(f'Patients: {n_pt} | Controls: {n_ctrl}')
        print('=' * 60)
        print()
        print('To bootstrap: python 03_resample_selectivity.py --suffix _wholehemi')

    print('\nDone!')


if __name__ == '__main__':
    main()