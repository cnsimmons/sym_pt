#!/usr/bin/env python3
"""
05_calc_peak_coords.py - Peak coordinate analysis for anatomical location

Following Ayzenberg et al. (2023):
  1. For each subject × category, find peak voxel within searchmask
  2. Convert native-space peak to MNI coordinates using anat2stand.mat
  3. Compute Euclidean distance between each patient peak and control peaks
  4. Bootstrap: sample 4 controls, compute mean distance to remaining controls
  5. Test if patient distance falls within control distribution

Requires: anat2stand.mat from register_mirror.py (already exists)

Usage:
  python 05_calc_peak_coords.py
  python 05_calc_peak_coords.py --sub 004
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
                           is_patient, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────

THRESH = 2.3

CATEGORY_COPES = {
    'face': 1,
    'house': 2,
    'object': 3,
    'word': 4,
}

N_SUBS = 4
ITER = 10000


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


def native_voxel_to_mni_mm(voxel_ijk, native_img, anat2stand_mat, mni_ref):
    """
    Convert a native-space voxel index to MNI mm coordinates.
    
    Steps:
      1. voxel_ijk → native mm (using native affine)
      2. native mm → MNI mm (using anat2stand.mat from FLIRT)
    
    FLIRT matrices operate on mm coordinates (not voxel indices),
    but are relative to the image's FSL-convention origin.
    We use img2imgcoord for accuracy.
    """
    # Write temp coordinate file
    tmp_in = '/tmp/peak_native_coord.txt'
    tmp_out = '/tmp/peak_mni_coord.txt'

    # img2imgcoord expects voxel coordinates with -vox flag
    with open(tmp_in, 'w') as f:
        f.write(f'{voxel_ijk[0]} {voxel_ijk[1]} {voxel_ijk[2]}\n')

    cmd = (f'img2imgcoord -src {native_img} -dest {mni_ref} '
           f'-xfm {anat2stand_mat} -vox {tmp_in}')

    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        # Parse output — img2imgcoord prints header then coordinates
        lines = result.stdout.strip().split('\n')
        # Last line should have the coordinates
        coords = lines[-1].strip().split()
        mni_xyz = [float(c) for c in coords[:3]]
        return mni_xyz
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f'    WARNING: img2imgcoord failed: {e}')
        return [np.nan, np.nan, np.nan]


def find_peak_in_mask(zstat_path, mask_path, thresh=THRESH):
    """
    Find the peak (max z-value) voxel within a mask above threshold.
    Returns: (peak_ijk, peak_value) or (None, None) if no suprathreshold voxels.
    """
    zstat_data = nib.load(zstat_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata() > 0

    # Mask and threshold
    masked = np.where(mask_data & (zstat_data > thresh), zstat_data, 0)

    if masked.max() == 0:
        return None, None

    peak_idx = np.unravel_index(np.argmax(masked), masked.shape)
    peak_val = float(masked[peak_idx])

    return list(peak_idx), peak_val


# ── Core ─────────────────────────────────────────────────────────────────────

def process_subject(sub_clean, ses, first_ses):
    """Extract peak MNI coordinates for all categories."""
    ses_str = f'{ses:02d}'
    first_ses_str = f'{first_ses:02d}'

    base_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}'
    roi_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/ROIs'
    anat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/anat'
    gfeat_dir = f'{base_dir}/derivatives/fsl/loc/HighLevel.gfeat'

    anat2stand = f'{anat_dir}/anat2stand.mat'
    native_brain = f'{anat_dir}/T1w_brain.nii.gz'
    mni_ref = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'

    if not os.path.exists(anat2stand):
        print(f'  SKIP: anat2stand.mat not found')
        return []

    hemis, group, intact_hemi = get_hemis_for_subject(sub_clean, ses)
    if not hemis:
        print(f'  SKIP: pre-surgical or no intact_hemi')
        return []

    results = []

    for category, cope_num in CATEGORY_COPES.items():
        # Find zstat
        zstat_path = (f'{gfeat_dir}/cope{cope_num}.feat/stats/'
                      f'zstat1_ses{first_ses_str}.nii.gz')
        if not os.path.exists(zstat_path):
            zstat_path = f'{gfeat_dir}/cope{cope_num}.feat/stats/zstat1.nii.gz'
            if not os.path.exists(zstat_path):
                continue

        for hemi in hemis:
            mask_path = f'{roi_dir}/{hemi}_{category}_searchmask.nii.gz'
            if not os.path.exists(mask_path):
                continue

            peak_ijk, peak_val = find_peak_in_mask(zstat_path, mask_path, THRESH)

            if peak_ijk is None:
                print(f'  {hemi}_{category}: no peak')
                continue

            # Convert to MNI
            mni_xyz = native_voxel_to_mni_mm(peak_ijk, native_brain,
                                              anat2stand, mni_ref)

            hemi_full = 'left' if hemi == 'l' else 'right'

            results.append({
                'sub': f'sub-{sub_clean}',
                'ses': ses_str,
                'group': group,
                'intact_hemi': intact_hemi,
                'hemi': hemi_full,
                'category': category,
                'peak_x_mni': mni_xyz[0],
                'peak_y_mni': mni_xyz[1],
                'peak_z_mni': mni_xyz[2],
                'peak_val': peak_val,
            })

            print(f'  {hemi}_{category}: MNI=({mni_xyz[0]:.1f}, {mni_xyz[1]:.1f}, '
                  f'{mni_xyz[2]:.1f}) z={peak_val:.2f}')

    return results


def calc_patient_distances(peak_df, n_subs=N_SUBS, n_iter=ITER):
    """
    For each patient, compute mean Euclidean distance to control peaks.
    Bootstrap control-to-control distances for comparison.
    """
    controls = peak_df[peak_df['group'] == 'control']
    patients = peak_df[peak_df['group'] != 'control']

    # Use first session per control
    controls = controls.copy()
    controls['ses_int'] = controls['ses'].astype(int)
    first_ses = controls.groupby('sub')['ses_int'].min().reset_index()
    first_ses.columns = ['sub', 'first_ses']
    controls = controls.merge(first_ses, on='sub')
    controls = controls[controls['ses_int'] == controls['first_ses']]

    # Same for patients
    patients = patients.copy()
    patients['ses_int'] = patients['ses'].astype(int)
    first_ses_pt = patients.groupby('sub')['ses_int'].min().reset_index()
    first_ses_pt.columns = ['sub', 'first_ses']
    patients = patients.merge(first_ses_pt, on='sub')
    patients = patients[patients['ses_int'] == patients['first_ses']]

    all_results = []
    all_resamples = {}

    categories = peak_df['category'].unique()
    hemis = controls['hemi'].unique()

    for cat in sorted(categories):
        for hemi in sorted(hemis):
            ctrl_sub = controls[(controls['category'] == cat) &
                                (controls['hemi'] == hemi)]

            if len(ctrl_sub) < n_subs + 1:
                continue

            ctrl_coords = ctrl_sub[['peak_x_mni', 'peak_y_mni', 'peak_z_mni']].values
            ctrl_subs_list = ctrl_sub['sub'].values

            # Bootstrap control distances
            rng = np.random.default_rng(seed=42)
            boot_dists = []

            for _ in range(n_iter):
                idx = rng.choice(len(ctrl_coords), size=n_subs, replace=False)
                remaining = np.delete(np.arange(len(ctrl_coords)), idx)

                sample_dists = []
                for i in idx:
                    dists = np.sqrt(np.sum((ctrl_coords[i] - ctrl_coords[remaining])**2, axis=1))
                    sample_dists.append(np.mean(dists))

                boot_dists.append(np.mean(sample_dists))

            boot_dists = np.array(boot_dists)
            col_name = f'{cat}_{hemi}'
            all_resamples[col_name] = boot_dists

            # Patient distances
            pt_sub = patients[(patients['category'] == cat) &
                              (patients['hemi'] == hemi)]

            for _, row in pt_sub.iterrows():
                pt_coord = np.array([row['peak_x_mni'], row['peak_y_mni'], row['peak_z_mni']])

                if np.any(np.isnan(pt_coord)):
                    continue

                dists_to_ctrl = np.sqrt(np.sum((pt_coord - ctrl_coords)**2, axis=1))
                mean_dist = float(np.mean(dists_to_ctrl))

                percentile = float(np.mean(boot_dists <= mean_dist) * 100)

                all_results.append({
                    'sub': row['sub'],
                    'group': row['group'],
                    'intact_hemi': row['intact_hemi'],
                    'hemi': row['hemi'],
                    'category': cat,
                    'mean_dist_mm': mean_dist,
                    'percentile': percentile,
                    'within_ci': percentile <= 97.5,
                })

    return pd.DataFrame(all_results), all_resamples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Peak coordinate analysis')
    parser.add_argument('--sub', type=str, help='Single subject')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    print('=' * 60)
    print('PEAK COORDINATE ANALYSIS')
    print('=' * 60)

    if args.sub:
        sub_clean = args.sub.replace('sub-', '')
        subjects = [sub_clean]
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
            results = process_subject(sub_clean, ses, first_ses)
            all_results.extend(results)

    if not all_results:
        print('No results')
        return

    peak_df = pd.DataFrame(all_results)

    out_dir = f'{processed_dir}/group_results/peak_coords'
    os.makedirs(out_dir, exist_ok=True)

    # Save all peak coordinates
    peak_file = f'{out_dir}/peak_coords{args.suffix}.csv'
    peak_df.to_csv(peak_file, index=False)
    print(f'\nSaved: {peak_file} ({len(peak_df)} peaks)')

    # Run distance analysis if we have both patients and controls
    if peak_df['group'].nunique() > 1:
        print('\nComputing patient distances and bootstrapping...')
        dist_df, resamples = calc_patient_distances(peak_df)

        dist_file = f'{out_dir}/patient_distances{args.suffix}.csv'
        dist_df.to_csv(dist_file, index=False)
        print(f'Saved: {dist_file}')

        # Save bootstrap distributions
        resample_df = pd.DataFrame(resamples)
        resample_file = f'{out_dir}/distance_resamples{args.suffix}.csv'
        resample_df.to_csv(resample_file, index=False)

        # Summary
        print('\n' + '=' * 60)
        print('PATIENT DISTANCE SUMMARY (within 97.5th pctl = normal)')
        print('=' * 60)
        for sub in sorted(dist_df['sub'].unique()):
            sub_data = dist_df[dist_df['sub'] == sub]
            line = f'{sub}: '
            for _, r in sub_data.iterrows():
                marker = '-' if r['within_ci'] else '*'
                line += (f'{r["category"]}_{r["hemi"]}='
                         f'{r["mean_dist_mm"]:.1f}mm({r["percentile"]:.0f}%){marker} ')
            print(line)

    print('\nDone!')


if __name__ == '__main__':
    main()