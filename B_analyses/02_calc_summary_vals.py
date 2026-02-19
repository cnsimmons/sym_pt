#!/usr/bin/env python3
"""
02_calc_summary_vals.py - Calculate selectivity metrics per subject
Computes mean activation, active volume, summed selectivity, and 
normalized summed selectivity within category-specific searchmasks.

Follows Ayzenberg et al. (2023) selectivity analysis:
  - Threshold zstat at p < .01 uncorrected (z ~ 2.3)
  - Within each searchmask, extract surviving voxels
  - Compute: mean_act, volume (mm³), sum_selec, sum_selec_norm

Key filtering:
  - Excludes pre-surgical sessions (pre_post != 'post' for patients)
  - For patients: only analyzes intact hemisphere
  - For controls: analyzes both hemispheres

Usage:
  python 02_calc_summary_vals.py              # All subjects
  python 02_calc_summary_vals.py --sub 004    # Single subject
  python 02_calc_summary_vals.py --dry-run    # Preview only
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

# Threshold for statistical maps (z-score)
# Ayzenberg used p < .01 uncorrected ≈ z = 2.3
THRESH = 2.3

# ── Contrast Maps ────────────────────────────────────────────────────────────
# Three contrast maps following different approaches in the literature.
#
# 1. "differential" — each category vs. a single other category (Ayzenberg 2023)
# 2. "category_vs_rest" — each category vs. mean of all others
# 3. "liu" — pairwise contrasts used in Liu 2025
#    NOTE: Liu contrasts require COPEs that may not yet exist in your FEAT.
#    If missing, patch the 1stLevel template and re-run FEAT first.

# ── CATEGORY_COPES ─────────────────────────────────────────────────────────────
# Replace the existing CATEGORY_COPES dict in 02_calc_summary_vals.py.
#
# ORIGINAL keys (face, house, object, word) are UNCHANGED.
# New sub-ROI keys added — additive only.
#
# Cope numbers from design.fsf:
#   1  = Face > Object
#   2  = House > Object
#   3  = Object > Scramble
#   4  = Word > Object
#   9  = Word > mean(Face+House+Object+Scramble)
#   3  = Object > Scramble  (reused for LOC and pF)
#   3  = Object > Scramble  (reused for evc — most category-neutral differential)

CATEGORY_COPES = {
    # ── ORIGINAL — DO NOT CHANGE ──────────────────────────────────────────────
    'face':   1,   # Face > Object
    'house':  2,   # House > Object
    'object': 3,   # Object > Scramble
    'word':   4,   # Word > Object

    # ── NEW: house split ──────────────────────────────────────────────────────
    'house_PPA': 2,   # House > Object  (same as house)
    'house_TOS': 2,   # House > Object  (same as house)

    # ── NEW: face sub-ROIs ────────────────────────────────────────────────────
    'face_FFA': 1,    # Face > Object
    'face_STS': 1,    # Face > Object

    # ── NEW: object sub-ROIs ──────────────────────────────────────────────────
    'object_LOC': 3,  # Object > Scramble
    'object_pF':  3,  # Object > Scramble

    # ── NEW: word sub-ROIs ────────────────────────────────────────────────────
    'word_VWFA': 4,   # Word > Object
    'word_STG':  9,   # Word > mean(all) — more selective for language areas

    # ── NEW: early visual cortex ──────────────────────────────────────────────
    'evc': 3,         # Object > Scramble — least category-specific differential
}

# Default contrast map
DEFAULT_CONTRAST = 'differential'

# Output suffix (for different threshold/contrast versions)
OUTPUT_SUFFIX = ''

# Broad mask mode: category → pathway mapping
# When --broad is used, measures within ventral/dorsal parcels
# instead of category-specific searchmasks
BROAD_MASK_MAP = {
    'face': 'ventral_visual_cortex',
    'word': 'ventral_visual_cortex',
    'house': 'ventral_visual_cortex',
    'object': 'ventral_visual_cortex',   # LOC is ventral/lateral stream
}

# MNI ROI source directory (bilateral masks before splitting)
from sym_pt_params import roi_dir as mni_roi_dir, mni_brain

# ── Helper: determine which hemispheres to analyze ───────────────────────────

def get_hemis_for_subject(sub_clean, ses):
    """
    Return list of hemispheres to analyze for this subject.
    - Controls: both ['l', 'r']
    - Patients (post-surgical only): intact hemisphere only
    - Pre-surgical patients: empty list (skip)
    """
    info = get_sub_info(sub_clean, ses)
    group = info.get('group', '')
    intact_hemi = info.get('intact_hemi', '')
    pre_post = info.get('pre_post', '') if 'pre_post' in info else ''
    
    # Get pre_post directly from CSV since get_sub_info might not include it
    df = _load_csv()
    sub_rows = df[(df['sub_clean'] == sub_clean) & (df['ses_num'] == ses)]
    if not sub_rows.empty:
        pre_post = sub_rows.iloc[0].get('pre_post', 'na')
    
    if group == 'control':
        return ['l', 'r'], group, 'control'
    
    # Patient - skip pre-surgical
    if pre_post == 'pre':
        return [], group, intact_hemi
    
    # Patient post-surgical - intact hemisphere only
    if intact_hemi == 'left':
        return ['l'], group, intact_hemi
    elif intact_hemi == 'right':
        return ['r'], group, intact_hemi
    else:
        return [], group, intact_hemi


# ── Core Functions ───────────────────────────────────────────────────────────

def get_or_create_broad_hemi_mask(sub_clean, first_ses, hemi, pathway_name):
    """
    Create a hemisphere-specific broad mask on the fly.
    
    The pre-split masks (lVentral, rVentral etc.) from register_mirror.py
    have a splitting bug (empty left masks). Instead, we warp the full
    bilateral mask to native space and split by hemisphere here.
    
    Caches result so it's only done once per subject.
    """
    import subprocess
    
    first_ses_str = f'{first_ses:02d}'
    anat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/anat'
    roi_out_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/ROIs'
    os.makedirs(roi_out_dir, exist_ok=True)
    
    hemi_label = 'left' if hemi == 'l' else 'right'
    out_path = f'{roi_out_dir}/{hemi}_{pathway_name}_broad.nii.gz'
    
    if os.path.exists(out_path):
        return out_path
    
    # Source: bilateral MNI-space mask
    src_mask = f'{mni_roi_dir}/{pathway_name}.nii.gz'
    if not os.path.exists(src_mask):
        print(f'    WARNING: MNI mask not found: {src_mask}')
        return None
    
    # Transform
    mni2anat = f'{anat_dir}/mni2anat.mat'
    ref_brain = f'{anat_dir}/T1w_brain.nii.gz'
    
    if not os.path.exists(mni2anat) or not os.path.exists(ref_brain):
        print(f'    WARNING: transform or ref brain not found')
        return None
    
    # Warp bilateral mask to native space
    tmp_bilateral = f'{roi_out_dir}/{pathway_name}_bilateral_tmp.nii.gz'
    if not os.path.exists(tmp_bilateral):
        cmd = (f'flirt -in {src_mask} -ref {ref_brain} '
               f'-out {tmp_bilateral} -applyxfm -init {mni2anat} '
               f'-interp trilinear')
        subprocess.run(cmd.split(), check=True, capture_output=True)
    
    # Load and split by hemisphere
    bilateral_img = nib.load(tmp_bilateral)
    bilateral_data = bilateral_img.get_fdata()
    
    # Binarize first
    binary_data = (bilateral_data > 0.5).astype(np.float32)
    
    # Split at midline
    mid_x = binary_data.shape[0] // 2
    hemi_data = np.zeros_like(binary_data)
    
    if hemi == 'l':
        hemi_data[mid_x:, :, :] = binary_data[mid_x:, :, :]
    elif hemi == 'r':
        hemi_data[:mid_x, :, :] = binary_data[:mid_x, :, :]
    
    # Save
    out_img = nib.Nifti1Image(hemi_data, bilateral_img.affine)
    nib.save(out_img, out_path)
    
    n_vox = int(hemi_data.sum())
    print(f'    Created broad mask: {out_path} ({n_vox:,} voxels)')
    
    # Clean up temp
    if os.path.exists(tmp_bilateral):
        os.remove(tmp_bilateral)
    
    return out_path


def calc_summary_vals(zstat_path, searchmask_path, thresh=THRESH):
    """
    Calculate selectivity metrics within a searchmask.
    
    Following Ayzenberg et al. (2023):
      - mean_act: mean of suprathreshold voxel values
      - volume: count of suprathreshold voxels × voxel size (mm³)
      - sum_selec: sum of suprathreshold voxel values
      - sum_selec_norm: sum_selec / total_mask_voxels × 1000
    
    Returns: (mask_size, mean_act, volume, sum_selec, sum_selec_norm)
    """
    zstat_img = nib.load(zstat_path)
    zstat_data = zstat_img.get_fdata()
    
    mask_img = nib.load(searchmask_path)
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


def process_subject(sub_clean, ses, first_ses, category_copes, thresh=THRESH, dry_run=False, broad=False):
    """Process all categories × hemispheres for one subject-session."""
    ses_str = f'{ses:02d}'
    first_ses_str = f'{first_ses:02d}'
    
    base_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}'
    # Searchmasks are always in the first session's ROI dir (anatomical, session-independent)
    roi_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/ROIs'
    # Broad ventral/dorsal masks are in derivatives/rois (from register_mirror.py)
    broad_roi_dir = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses_str}/derivatives/rois'
    gfeat_dir = f'{base_dir}/derivatives/fsl/loc/HighLevel.gfeat'
    
    # Determine which hemispheres to analyze
    hemis, group, intact_hemi = get_hemis_for_subject(sub_clean, ses)
    
    if not hemis:
        print(f'  SKIP: pre-surgical or no intact_hemi info')
        return []
    
    hemi_label = intact_hemi if group != 'control' else 'both'
    print(f'  Group: {group} | Intact hemi: {hemi_label} | Analyzing: {hemis}')
    
    results = []
    
    for category, cope_num in category_copes.items():
        zstat_path = (f'{gfeat_dir}/cope{cope_num}.feat/stats/'
                      f'zstat1_ses{first_ses_str}.nii.gz')
        
        if not os.path.exists(zstat_path):
            zstat_alt = f'{gfeat_dir}/cope{cope_num}.feat/stats/zstat1.nii.gz'
            if os.path.exists(zstat_alt):
                zstat_path = zstat_alt
            else:
                print(f'  SKIP {category}: zstat not found')
                continue
        
        for hemi in hemis:
            # Determine mask path based on mode
            if broad:
                pathway = BROAD_MASK_MAP[category]
                searchmask_path = get_or_create_broad_hemi_mask(
                    sub_clean, first_ses, hemi, pathway)
                if searchmask_path is None:
                    print(f'  SKIP {hemi}_{category}: broad mask creation failed')
                    continue
                mask_label = f'{hemi}_{category}_broad'
            else:
                searchmask_path = f'{roi_dir}/{hemi}_{category}_searchmask.nii.gz'
                mask_label = f'{hemi}_{category}'
            
            if not os.path.exists(searchmask_path):
                print(f'  SKIP {mask_label}: mask not found')
                continue
            
            if dry_run:
                print(f'  WOULD COMPUTE: {mask_label} (cope{cope_num})')
                continue
            
            mask_size, mean_act, volume, sum_selec, sum_selec_norm = \
                calc_summary_vals(zstat_path, searchmask_path, thresh)
            
            hemi_full = 'left' if hemi == 'l' else 'right'
            
            results.append({
                'sub': f'sub-{sub_clean}',
                'ses': ses_str,
                'group': group,
                'intact_hemi': intact_hemi,
                'hemi': hemi_full,
                'category': category,
                'contrast_map': 'broad' if broad else next(
                    (k for k, v in CATEGORY_COPES.items() if v == category_copes), 'unknown'),
                'cope': cope_num,
                'mask_type': 'broad' if broad else 'searchmask',
                'mask_size': mask_size,
                'mean_act': mean_act,
                'volume': volume,
                'sum_selec': sum_selec,
                'sum_selec_norm': sum_selec_norm,
            })
            
            if not np.isnan(mean_act):
                n_active = int(volume / np.prod(nib.load(zstat_path).header.get_zooms()[:3]))
                print(f'  {mask_label}: {n_active} active voxels, '
                      f'mean_act={mean_act:.2f}, sum_selec_norm={sum_selec_norm:.2f}')
            else:
                print(f'  {mask_label}: no suprathreshold voxels')
    
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Calculate selectivity summary values')
    parser.add_argument('--sub', type=str, help='Single subject (e.g., 004)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without computing')
    parser.add_argument('--threshold', type=float, default=THRESH,
                        help=f'Z-score threshold (default: {THRESH})')
    parser.add_argument('--suffix', type=str, default=OUTPUT_SUFFIX,
                        help='Output file suffix')
    parser.add_argument('--broad', action='store_true',
                        help='Use broad ventral/dorsal masks instead of category searchmasks')
    parser.add_argument('--contrast', type=str, default=DEFAULT_CONTRAST,
                        choices=list(CATEGORY_COPES.keys()),
                        help=f'Contrast map to use (default: {DEFAULT_CONTRAST})')
    args = parser.parse_args()
    
    thresh = args.threshold
    
    # Select contrast map
    active_copes = CATEGORY_COPES
    
    # Auto-set suffix based on flags
    suffix = args.suffix
    if not suffix:
        parts = []
        if args.contrast != DEFAULT_CONTRAST:
            parts.append(f'_{args.contrast}')
        if args.broad:
            parts.append('_broad')
        suffix = ''.join(parts)
    
    print('=' * 60)
    print('CALCULATE SELECTIVITY SUMMARY VALUES')
    print('=' * 60)
    print(f'Threshold: z > {thresh}')
    print(f'Contrast map: {args.contrast}')
    print(f'Categories & COPEs: {active_copes}')
    print(f'Mask type: {"BROAD ventral/dorsal" if args.broad else "category-specific searchmasks"}')
    print(f'Output suffix: "{suffix}"')
    print(f'NOTE: Patients = intact hemi only, pre-surgical excluded')
    print()
    
    # Determine subjects
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
    
    all_results = []
    
    for sub_clean in subjects:
        sessions = get_sessions(sub_clean)
        if not sessions:
            print(f'  SKIP sub-{sub_clean}: no sessions')
            continue
        
        first_ses = sessions[0]
        
        for ses in sessions:
            print(f'=== sub-{sub_clean} ses-{ses:02d} ===')
            
            results = process_subject(sub_clean, ses, first_ses, active_copes, thresh, args.dry_run, args.broad)
            all_results.extend(results)
    
    if not args.dry_run and all_results:
        df = pd.DataFrame(all_results)
        
        out_dir = f'{processed_dir}/group_results/selectivity'
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = f'{out_dir}/selectivity_summary{suffix}.csv'
        df.to_csv(out_file, index=False)
        
        # Print summary
        n_patients = df[df['group'] != 'control']['sub'].nunique()
        n_controls = df[df['group'] == 'control']['sub'].nunique()
        
        print()
        print('=' * 60)
        print(f'Saved: {out_file}')
        print(f'Total rows: {len(df)}')
        print(f'Patients: {n_patients} | Controls: {n_controls}')
        print(f'Sessions: {df["ses"].nunique()}')
        print('=' * 60)
    elif args.dry_run:
        print()
        print('DRY RUN complete - no files written')


if __name__ == '__main__':
    main()