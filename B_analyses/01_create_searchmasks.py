#!/usr/bin/env python3
"""
01_create_searchmasks.py - Create category-specific anatomical searchmasks
Uses Harvard-Oxford cortical atlas probability maps warped to native space.
Generates uniform searchmasks for ALL subjects (patients and controls).

Category-parcel mapping (ORIGINAL — unchanged):
  face:   Temporal Fusiform (anterior + posterior) + Temporal Occipital Fusiform
  word:   Temporal Fusiform (anterior + posterior) + Temporal Occipital Fusiform
  object: Lateral Occipital Cortex (superior + inferior)
  house:  Parahippocampal (anterior + posterior) + Lingual + Posterior Cingulate

New sub-ROIs (additive — original pipeline unaffected):
  house_PPA:  Parahippocampal anterior + posterior
  house_TOS:  Lateral Occipital sup/inf + Lingual
  face_FFA:   Temporal Fusiform posterior + Temporal Occipital Fusiform
  face_STS:   Superior Temporal Gyrus anterior + posterior
  object_LOC: Lateral Occipital superior + inferior
  object_pF:  Temporal Occipital Fusiform
  word_VWFA:  Temporal Fusiform anterior + posterior
  word_STG:   Superior Temporal Gyrus anterior + posterior
  evc:        Intracalcarine Cortex + Cuneal Cortex

Usage:
  python 01_create_searchmasks.py              # All subjects
  python 01_create_searchmasks.py --sub 004    # Single subject
  python 01_create_searchmasks.py --dry-run    # Preview only
  python 01_create_searchmasks.py --new-only   # Only new sub-ROIs (skip originals)
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

# ── Configuration ─────────────────────────────────────────────────────────────

FSLDIR        = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
ATLAS_DIR     = f'{FSLDIR}/data/atlases/HarvardOxford'
PROB_ATLAS    = f'{ATLAS_DIR}/HarvardOxford-cort-prob-2mm.nii.gz'
PROB_THRESHOLD = 25
DILATION_ITERS = 1

# ── Harvard-Oxford index reference (0-based) ──────────────────────────────────
#  4  Intracalcarine Cortex
#  5  Cuneal Cortex
# 15  Superior Temporal Gyrus anterior
# 16  Superior Temporal Gyrus posterior
# 21  Lateral Occipital Cortex superior
# 22  Lateral Occipital Cortex inferior
# 29  Cingulate Gyrus posterior
# 33  Parahippocampal Gyrus anterior
# 34  Parahippocampal Gyrus posterior
# 35  Lingual Gyrus
# 36  Temporal Fusiform Cortex anterior
# 37  Temporal Fusiform Cortex posterior
# 38  Temporal Occipital Fusiform Cortex

# ── Category parcels ──────────────────────────────────────────────────────────

# ORIGINAL keys — DO NOT CHANGE. These produce identical outputs to before.
CATEGORY_PARCELS_ORIGINAL = {
    'face': {
        'indices': [36, 37, 38],
        'names':   ['Temporal Fusiform anterior', 'Temporal Fusiform posterior',
                    'Temporal Occipital Fusiform'],
    },
    'word': {
        'indices': [36, 37, 38],
        'names':   ['Temporal Fusiform anterior', 'Temporal Fusiform posterior',
                    'Temporal Occipital Fusiform'],
    },
    'object': {
        'indices': [21, 22],
        'names':   ['Lateral Occipital superior', 'Lateral Occipital inferior'],
    },
    'house': {
        'indices': [33, 34, 35, 29],
        'names':   ['Parahippocampal anterior', 'Parahippocampal posterior',
                    'Lingual Gyrus', 'Cingulate posterior'],
    },
}

# NEW sub-ROIs — additive only.
# house_PPA / house_TOS: required split — bimodal Y distribution confirmed
#   (GMM BIC k=3 >> k=1, clusters at Y~20mm and Y~40mm, sep ~20mm).
#   Epstein & Kanwisher (1998); Dilks et al. (2013).
# All other sub-ROIs: future use, commented literature in roi_contrast_table.md
CATEGORY_PARCELS_NEW = {
    # ── House split (required) ────────────────────────────────────────────────
    'house_PPA': {
        'indices': [33, 34],
        'names':   ['Parahippocampal anterior', 'Parahippocampal posterior'],
    },
    'house_TOS': {
        'indices': [21, 22, 35],
        'names':   ['Lateral Occipital superior', 'Lateral Occipital inferior',
                    'Lingual Gyrus'],
    },
    # ── Face sub-ROIs ─────────────────────────────────────────────────────────
    'face_FFA': {
        'indices': [37, 38],
        'names':   ['Temporal Fusiform posterior', 'Temporal Occipital Fusiform'],
    },
    'face_STS': {
        'indices': [15, 16],
        'names':   ['Superior Temporal Gyrus anterior',
                    'Superior Temporal Gyrus posterior'],
    },
    # ── Object sub-ROIs ───────────────────────────────────────────────────────
    'object_LOC': {
        'indices': [21, 22],
        'names':   ['Lateral Occipital superior', 'Lateral Occipital inferior'],
    },
    'object_pF': {
        'indices': [38],
        'names':   ['Temporal Occipital Fusiform'],
    },
    # ── Word sub-ROIs ─────────────────────────────────────────────────────────
    'word_VWFA': {
        'indices': [36, 37],
        'names':   ['Temporal Fusiform anterior', 'Temporal Fusiform posterior'],
    },
    'word_STG': {
        'indices': [15, 16],
        'names':   ['Superior Temporal Gyrus anterior',
                    'Superior Temporal Gyrus posterior'],
    },
    # ── Early visual cortex (Liu-style reference ROI) ─────────────────────────
    'evc': {
        'indices': [4, 5],
        'names':   ['Intracalcarine Cortex', 'Cuneal Cortex'],
    },
}

# Combined — used when running all masks
CATEGORY_PARCELS_ALL = {**CATEGORY_PARCELS_ORIGINAL, **CATEGORY_PARCELS_NEW}


# ── Core functions ────────────────────────────────────────────────────────────

def load_atlas():
    """Load the Harvard-Oxford probability atlas."""
    print(f"Loading atlas: {PROB_ATLAS}")
    atlas_img  = nib.load(PROB_ATLAS)
    atlas_data = atlas_img.get_fdata()
    print(f"  Shape: {atlas_data.shape} (x, y, z, regions)")
    print(f"  {atlas_data.shape[3]} regions available")
    return atlas_img, atlas_data


def extract_hemisphere_mask(atlas_data, region_indices, hemisphere,
                            threshold=PROB_THRESHOLD):
    """Extract and combine probability maps for given regions, split by hemisphere.
    MNI x-midpoint is at voxel 45 (for 2mm 91-voxel atlas)."""
    combined = np.zeros(atlas_data.shape[:3], dtype=float)
    for idx in region_indices:
        combined = np.maximum(combined, atlas_data[:, :, :, idx])

    binary_mask = combined > threshold
    midpoint    = atlas_data.shape[0] // 2   # 45 for 91-voxel atlas

    hemi_mask = np.zeros_like(binary_mask)
    if hemisphere == 'l':
        hemi_mask[midpoint:, :, :] = binary_mask[midpoint:, :, :]
    elif hemisphere == 'r':
        hemi_mask[:midpoint, :, :] = binary_mask[:midpoint, :, :]
    return hemi_mask


def warp_mask_to_native(mask_nii_path, ref_brain, mni2anat_mat, output_path):
    """Warp an MNI-space mask to native space using FLIRT."""
    cmd = (f'flirt -in {mask_nii_path} -ref {ref_brain} '
           f'-out {output_path} -applyxfm -init {mni2anat_mat} '
           f'-interp nearestneighbour')
    subprocess.run(cmd.split(), check=True, capture_output=True)
    cmd_bin = f'fslmaths {output_path} -bin {output_path}'
    subprocess.run(cmd_bin.split(), check=True, capture_output=True)


def create_searchmasks_for_subject(sub, ses, atlas_img, atlas_data,
                                   parcels, prob_threshold=PROB_THRESHOLD,
                                   dilation_iters=DILATION_ITERS, dry_run=False):
    """Create searchmasks for one subject-session using the given parcels dict."""
    import shutil
    sub_clean = sub.replace('sub-', '')
    ses_str   = f'{ses:02d}'

    anat_dir  = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    roi_dir   = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/ROIs'
    ref_brain = f'{anat_dir}/T1w_brain.nii.gz'
    mni2anat  = f'{anat_dir}/mni2anat.mat'

    if not os.path.exists(ref_brain):
        print(f"  SKIP: T1w_brain.nii.gz missing")
        return 0
    if not os.path.exists(mni2anat):
        print(f"  SKIP: mni2anat.mat missing")
        return 0

    if not dry_run:
        os.makedirs(roi_dir, exist_ok=True)

    tmp_dir = f'/tmp/searchmasks_sub-{sub_clean}'
    if not dry_run:
        os.makedirs(tmp_dir, exist_ok=True)

    n_created = 0

    for hemi in ['l', 'r']:
        for category, parcel_info in parcels.items():
            output_file = f'{roi_dir}/{hemi}_{category}_searchmask.nii.gz'

            if os.path.exists(output_file):
                continue

            if dry_run:
                print(f"  WOULD CREATE: {hemi}_{category}_searchmask.nii.gz "
                      f"({', '.join(parcel_info['names'])})")
                n_created += 1
                continue

            # 1. Extract hemisphere mask in MNI space
            hemi_mask = extract_hemisphere_mask(
                atlas_data, parcel_info['indices'], hemi, prob_threshold)

            if hemi_mask.sum() == 0:
                print(f"  WARNING: {hemi}_{category} empty in MNI space")
                continue

            # 2. Save MNI-space mask temporarily
            tmp_mni = f'{tmp_dir}/{hemi}_{category}_mni.nii.gz'
            nib.save(nib.Nifti1Image(hemi_mask.astype(np.float32),
                                      atlas_img.affine), tmp_mni)

            # 3. Warp to native space
            tmp_native = f'{tmp_dir}/{hemi}_{category}_native.nii.gz'
            warp_mask_to_native(tmp_mni, ref_brain, mni2anat, tmp_native)

            # 4. Load, optionally dilate
            native_img  = nib.load(tmp_native)
            native_data = native_img.get_fdata() > 0
            if dilation_iters > 0:
                native_data = binary_dilation(native_data,
                                              iterations=dilation_iters)

            # 5. Save
            nib.save(nib.Nifti1Image(native_data.astype(np.float32),
                                      native_img.affine), output_file)

            print(f"  {hemi}_{category}: {native_data.sum():,} voxels "
                  f"({', '.join(parcel_info['names'])})")
            n_created += 1

    if not dry_run and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return n_created


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Create Harvard-Oxford searchmasks')
    parser.add_argument('--sub',       type=str,
                        help='Single subject (e.g., 004)')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Preview without creating files')
    parser.add_argument('--new-only',  action='store_true',
                        help='Only generate new sub-ROI masks (skip originals)')
    parser.add_argument('--threshold', type=int, default=PROB_THRESHOLD,
                        help=f'Probability threshold %% (default: {PROB_THRESHOLD})')
    parser.add_argument('--no-dilate', action='store_true',
                        help='Skip dilation step')
    args = parser.parse_args()

    prob_threshold = args.threshold
    dilation_iters = 0 if args.no_dilate else DILATION_ITERS

    # Select parcel set
    if args.new_only:
        parcels    = CATEGORY_PARCELS_NEW
        parcel_tag = 'new sub-ROIs only'
    else:
        parcels    = CATEGORY_PARCELS_ALL
        parcel_tag = 'all (original + new sub-ROIs)'

    print("=" * 60)
    print("CREATE SEARCHMASKS (Harvard-Oxford Atlas)")
    print("=" * 60)
    print(f"Parcel set:            {parcel_tag}")
    print(f"Probability threshold: {prob_threshold}%")
    print(f"Dilation iterations:   {dilation_iters}")
    print(f"Masks to generate:     {list(parcels.keys())}")
    print()

    atlas_img, atlas_data = load_atlas()

    if args.sub:
        subjects = [args.sub.replace('sub-', '')]
    else:
        subjects = sorted(
            d.replace('sub-', '')
            for d in os.listdir(processed_dir)
            if d.startswith('sub-') and d.replace('sub-', '') not in skip_subs
        )

    print(f"\nProcessing {len(subjects)} subjects\n")
    total_created = 0

    for sub_clean in subjects:
        sessions = get_sessions(sub_clean)
        if not sessions:
            continue
        first_ses = sessions[0]
        print(f"=== sub-{sub_clean} ses-{first_ses:02d} ===")
        n = create_searchmasks_for_subject(
            f'sub-{sub_clean}', first_ses, atlas_img, atlas_data,
            parcels, prob_threshold, dilation_iters, args.dry_run)
        total_created += n

    print()
    print("=" * 60)
    print(f"{'Would create' if args.dry_run else 'Created'}: "
          f"{total_created} searchmasks")
    print("=" * 60)


if __name__ == '__main__':
    main()