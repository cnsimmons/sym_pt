#!/usr/bin/env python3
"""
04_calc_confounds.py - Compute tSNR and motion metrics for paper reporting

this can take a while

Following Ayzenberg et al. (2023):
  - Motion: mean absolute rotation (deg) and translation (mm) from MCFLIRT
  - tSNR: mean(timeseries) / std(timeseries) within each searchmask
  - Reports that data quality is comparable across patients and controls

Computes:
  1. Per-run motion (rotation, translation) from MCFLIRT .par files
  2. Per-run tSNR within each category searchmask (registered func data)
  3. Subject-level averages across runs

Outputs:
  - confound_summary.csv: per-subject averages (group, motion, tSNR)
  - confound_by_run.csv: per-run details
  - Terminal summary comparing patients vs controls

Usage:
  python 04_calc_confounds.py              # All subjects
  python 04_calc_confounds.py --sub 004    # Single subject
  

optional slurm:
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=confounds
#SBATCH --output=slurm_out/confounds_%j.out
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -p cpu
#SBATCH --cpus-per-task=1

module load fsl/6.0.3
export FSLDIR=/opt/fsl/6.0.3
. ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

python B_analyses/04_calc_confounds.py
EOT
  
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
                           is_patient, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────

#CATEGORIES = ['face', 'word', 'object', 'house']

CATEGORIES = [
    # original
    'face', 'word', 'object', 'house',
    # new sub-ROIs
    'house_PPA', 'house_TOS',
    'face_FFA', 'face_STS',
    'object_LOC', 'object_pF',
    'word_VWFA', 'word_STG',
    'evc',
]


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
    else:
        return [], group, intact_hemi


# ── Core Functions ───────────────────────────────────────────────────────────

def calc_motion(par_file):
    """
    Extract mean absolute rotation and translation from MCFLIRT .par file.
    FSL .par columns: rot_x rot_y rot_z (radians) trans_x trans_y trans_z (mm)
    Returns: (mean_rot_deg, mean_trans_mm)
    """
    params = np.loadtxt(par_file)

    # Rotations (cols 0-2) in radians → degrees
    rot_deg = np.abs(params[:, :3]) * (180.0 / np.pi)
    trans_mm = np.abs(params[:, 3:])

    mean_rot = float(np.mean(rot_deg))
    mean_trans = float(np.mean(trans_mm))

    return mean_rot, mean_trans


def calc_tsnr(func_path, mask_path):
    """
    Compute tSNR = mean(timeseries) / std(timeseries) within a mask.
    Resamples mask to functional resolution to handle dimension mismatch.
    Returns mean tSNR across voxels in the mask.
    """
    from nilearn.image import resample_to_img, load_img, binarize_img

    func_img = load_img(func_path)
    if func_img.ndim != 4:
        return np.nan

    mask_img = load_img(mask_path)

    # Resample mask to functional data resolution (nearest neighbor)
    mask_resampled = resample_to_img(mask_img, func_img,
                                      interpolation='nearest')
    mask_data = mask_resampled.get_fdata() > 0

    if mask_data.sum() == 0:
        return np.nan

    func_data = func_img.get_fdata()
    masked_ts = func_data[mask_data, :]  # (n_voxels, n_timepoints)

    with np.errstate(divide='ignore', invalid='ignore'):
        voxel_tsnr = np.mean(masked_ts, axis=1) / np.std(masked_ts, axis=1)

    return float(np.nanmean(voxel_tsnr))


def process_subject(sub_clean, ses, first_ses):
    """Compute motion and tSNR for all runs in one subject-session."""
    ses_str = f'{ses:02d}'
    first_ses_str = f'{first_ses:02d}'

    task_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/loc'
    roi_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/ROIs'

    hemis, group, intact_hemi = get_hemis_for_subject(sub_clean, ses)
    if not hemis:
        print(f'  SKIP: pre-surgical or no intact_hemi')
        return []

    # Find completed runs
    feat_dirs = sorted(glob(f'{task_dir}/run-*/1stLevel.feat'))
    if not feat_dirs:
        print(f'  SKIP: no FEAT directories')
        return []

    run_results = []

    for feat_dir in feat_dirs:
        run = feat_dir.split('run-')[1].split('/')[0]

        # ── Motion ──
        par_file = f'{feat_dir}/mc/prefiltered_func_data_mcf.par'
        if os.path.exists(par_file):
            mean_rot, mean_trans = calc_motion(par_file)
        else:
            mean_rot, mean_trans = np.nan, np.nan

        # ── tSNR per searchmask ──
        func_reg = f'{feat_dir}/filtered_func_data_reg.nii.gz'
        if not os.path.exists(func_reg):
            # Fall back to unregistered
            func_reg = f'{feat_dir}/filtered_func_data.nii.gz'

        tsnr_vals = {}
        if os.path.exists(func_reg):
            for hemi in hemis:
                for cat in CATEGORIES:
                    mask_path = f'{roi_dir}/{hemi}_{cat}_searchmask.nii.gz'
                    if os.path.exists(mask_path):
                        tsnr = calc_tsnr(func_reg, mask_path)
                        hemi_full = 'left' if hemi == 'l' else 'right'
                        tsnr_vals[f'tsnr_{cat}_{hemi_full}'] = tsnr

        row = {
            'sub': f'sub-{sub_clean}',
            'ses': ses_str,
            'run': run,
            'group': group,
            'intact_hemi': intact_hemi,
            'mean_rot_deg': mean_rot,
            'mean_trans_mm': mean_trans,
        }
        row.update(tsnr_vals)
        run_results.append(row)

        # Brief output
        tsnr_str = ', '.join(f'{k}={v:.1f}' for k, v in tsnr_vals.items()
                             if not np.isnan(v))
        rot_str = f'{mean_rot:.4f}' if not np.isnan(mean_rot) else 'N/A'
        trans_str = f'{mean_trans:.4f}' if not np.isnan(mean_trans) else 'N/A'
        print(f'  run-{run}: rot={rot_str}° trans={trans_str}mm | {tsnr_str}')

    return run_results


def summarize_by_subject(run_df):
    """Average motion and tSNR across runs per subject-session."""
    # Identify tSNR columns dynamically
    tsnr_cols = [c for c in run_df.columns if c.startswith('tsnr_')]
    agg_dict = {
        'group': 'first',
        'intact_hemi': 'first',
        'mean_rot_deg': 'mean',
        'mean_trans_mm': 'mean',
    }
    for col in tsnr_cols:
        agg_dict[col] = 'mean'

    summary = run_df.groupby(['sub', 'ses']).agg(agg_dict).reset_index()

    # Also compute a mean tSNR across all searchmasks
    if tsnr_cols:
        summary['tsnr_mean'] = summary[tsnr_cols].mean(axis=1)

    return summary


def print_group_summary(summary_df):
    """Print patient vs control comparison for paper reporting."""
    print()
    print('=' * 70)
    print('GROUP SUMMARY (for paper reporting)')
    print('=' * 70)

    tsnr_cols = [c for c in summary_df.columns if c.startswith('tsnr_')]

    for group_label, group_filter in [
        ('Controls', summary_df['group'] == 'control'),
        ('Patients (OTC)', summary_df['group'] == 'OTC'),
        ('Patients (nonOTC)', summary_df['group'] == 'nonOTC'),
        ('All Patients', summary_df['group'].isin(['OTC', 'nonOTC'])),
    ]:
        gdata = summary_df[group_filter]
        if len(gdata) == 0:
            continue

        n = gdata['sub'].nunique()
        rot = gdata['mean_rot_deg']
        trans = gdata['mean_trans_mm']

        print(f'\n{group_label} (n={n}):')
        print(f'  Rotation:    M={rot.mean():.4f}°  SD={rot.std():.4f}  '
              f'range=[{rot.min():.4f}, {rot.max():.4f}]')
        print(f'  Translation: M={trans.mean():.4f}mm  SD={trans.std():.4f}  '
              f'range=[{trans.min():.4f}, {trans.max():.4f}]')

        if 'tsnr_mean' in gdata.columns:
            tsnr = gdata['tsnr_mean']
            print(f'  Mean tSNR:   M={tsnr.mean():.2f}  SD={tsnr.std():.2f}  '
                  f'range=[{tsnr.min():.2f}, {tsnr.max():.2f}]')

    # Flag any subjects with high motion
    print(f'\n{"─" * 70}')
    print('Subjects with mean rotation > 0.25° or translation > 0.20mm:')
    flagged = summary_df[
        (summary_df['mean_rot_deg'] > 0.25) |
        (summary_df['mean_trans_mm'] > 0.20)
    ]
    if len(flagged) == 0:
        print('  None')
    else:
        for _, row in flagged.iterrows():
            print(f'  {row["sub"]} ses-{row["ses"]} ({row["group"]}): '
                  f'rot={row["mean_rot_deg"]:.4f}° '
                  f'trans={row["mean_trans_mm"]:.4f}mm')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compute confound metrics')
    parser.add_argument('--sub', type=str, help='Single subject (e.g., 004)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Output file suffix')
    args = parser.parse_args()

    print('=' * 70)
    print('COMPUTE CONFOUND METRICS (tSNR + Motion)')
    print('=' * 70)
    print()

    if args.sub:
        sub_clean = args.sub.replace('sub-', '')
        subjects = [sub_clean]
    else:
        subjects = sorted(
            d.replace('sub-', '')
            for d in os.listdir(processed_dir)
            if d.startswith('sub-') and d.replace('sub-', '') not in skip_subs
        )

    print(f'Processing {len(subjects)} subjects')
    print()

    all_run_results = []

    for sub_clean in subjects:
        sessions = get_sessions(sub_clean)
        if not sessions:
            continue

        first_ses = sessions[0]

        for ses in sessions:
            print(f'=== sub-{sub_clean} ses-{ses:02d} ===')
            results = process_subject(sub_clean, ses, first_ses)
            all_run_results.extend(results)

    if not all_run_results:
        print('No results to save')
        return

    # Save per-run results
    run_df = pd.DataFrame(all_run_results)
    out_dir = f'{processed_dir}/group_results/confounds'
    os.makedirs(out_dir, exist_ok=True)

    run_file = f'{out_dir}/confound_by_run{args.suffix}.csv'
    run_df.to_csv(run_file, index=False)
    print(f'\nSaved per-run: {run_file} ({len(run_df)} rows)')

    # Compute and save subject-level summary
    summary_df = summarize_by_subject(run_df)
    summary_file = f'{out_dir}/confound_summary{args.suffix}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'Saved summary: {summary_file} ({len(summary_df)} rows)')

    # Print group-level comparison
    print_group_summary(summary_df)

    print('\nDone!')


if __name__ == '__main__':
    main()