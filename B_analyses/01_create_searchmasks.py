#!/usr/bin/env python3
"""
01_create_searchmasks.py - Create category-specific anatomical searchmasks
Uses Harvard-Oxford cortical atlas probability maps warped to native space.
Generates uniform searchmasks for ALL subjects (patients and controls).

Category-parcel mapping:
  face: Temporal Fusiform (anterior + posterior) + Temporal Occipital Fusiform
  word: Temporal Fusiform (anterior + posterior) + Temporal Occipital Fusiform
  object: Lateral Occipital Cortex (superior + inferior)
  house: Parahippocampal (anterior + posterior) + Lingual + Posterior Cingulate

Usage:
  python 01_create_searchmasks.py              # All subjects
  python 01_create_searchmasks.py --sub 004    # Single subject
  python 01_create_searchmasks.py --dry-run    # Preview only
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import nibabel as nib
from glob import glob
from scipy.ndimage import binary_dilation

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

# ── Configuration ────────────────────────────────────────────────────────────

FSLDIR = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
ATLAS_DIR = f'{FSLDIR}/data/atlases/HarvardOxford'

# Harvard-Oxford probability maps (one per hemisphere)
# These are 4D: each volume is a region's probability map (0-100)
PROB_ATLAS = f'{ATLAS_DIR}/HarvardOxford-cort-prob-2mm.nii.gz'

# Probability threshold (%) - voxels above this are included
PROB_THRESHOLD = 25

# Dilation iterations (to broaden masks, matching original FreeSurfer approach)
DILATION_ITERS = 1

# Harvard-Oxford index -> category mapping
# Indices are 0-based volumes in the probability atlas
CATEGORY_PARCELS = {
    'face': {
        'indices': [36, 37, 38],  # Temporal Fusiform ant/post + Temporal Occipital Fusiform
        'names': ['Temporal Fusiform anterior', 'Temporal Fusiform posterior',
                  'Temporal Occipital Fusiform']
    },
    'word': {
        'indices': [36, 37, 38],  # Same as face (fusiform covers both FFA and VWFA)
        'names': ['Temporal Fusiform anterior', 'Temporal Fusiform posterior',
                  'Temporal Occipital Fusiform']
    },
    'object': {
        'indices': [21, 22],  # Lateral Occipital superior + inferior
        'names': ['Lateral Occipital superior', 'Lateral Occipital inferior']
    },
    'house': {
        'indices': [33, 34, 35, 29],  # Parahippocampal ant/post + Lingual + Posterior Cingulate
        'names': ['Parahippocampal anterior', 'Parahippocampal posterior',
                  'Lingual Gyrus', 'Cingulate posterior']
    }
}


# ── Core Functions ───────────────────────────────────────────────────────────

def load_atlas():
    """Load the Harvard-Oxford probability atlas."""
    print(f"Loading atlas: {PROB_ATLAS}")
    atlas_img = nib.load(PROB_ATLAS)
    atlas_data = atlas_img.get_fdata()
    print(f"  Shape: {atlas_data.shape} (x, y, z, regions)")
    print(f"  {atlas_data.shape[3]} regions available")
    return atlas_img, atlas_data


def extract_hemisphere_mask(atlas_data, region_indices, hemisphere, threshold=PROB_THRESHOLD):
    """
    Extract and combine probability maps for given regions, split by hemisphere.
    MNI x-midpoint is at voxel 45 (for 2mm 91-voxel atlas).
    """
    combined = np.zeros(atlas_data.shape[:3], dtype=float)

    for idx in region_indices:
        combined = np.maximum(combined, atlas_data[:, :, :, idx])

    # Threshold
    binary_mask = combined > threshold

    # Split by hemisphere (MNI: x < midpoint = right hemisphere, x > midpoint = left)
    midpoint = atlas_data.shape[0] // 2  # 45 for 91-voxel

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

    # Binarize (FLIRT can introduce interpolation artifacts)
    cmd_bin = f'fslmaths {output_path} -bin {output_path}'
    subprocess.run(cmd_bin.split(), check=True, capture_output=True)


def create_searchmasks_for_subject(sub, ses, atlas_img, atlas_data, prob_threshold=PROB_THRESHOLD, dilation_iters=DILATION_ITERS, dry_run=False):
    """Create all category searchmasks for one subject-session."""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'

    anat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    roi_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/ROIs'
    ref_brain = f'{anat_dir}/T1w_brain.nii.gz'
    mni2anat = f'{anat_dir}/mni2anat.mat'

    if not os.path.exists(ref_brain):
        print(f"  SKIP: T1w_brain.nii.gz missing")
        return 0
    if not os.path.exists(mni2anat):
        print(f"  SKIP: mni2anat.mat missing")
        return 0

    if not dry_run:
        os.makedirs(roi_dir, exist_ok=True)

    # Temp directory for MNI-space masks
    tmp_dir = f'/tmp/searchmasks_sub-{sub_clean}'
    if not dry_run:
        os.makedirs(tmp_dir, exist_ok=True)

    n_created = 0

    for hemi in ['l', 'r']:
        for category, parcel_info in CATEGORY_PARCELS.items():
            output_file = f'{roi_dir}/{hemi}_{category}_searchmask.nii.gz'

            if os.path.exists(output_file):
                continue

            if dry_run:
                print(f"  WOULD CREATE: {hemi}_{category}_searchmask.nii.gz")
                n_created += 1
                continue

            # 1. Extract hemisphere-specific mask in MNI space
            hemi_mask = extract_hemisphere_mask(
                atlas_data, parcel_info['indices'], hemi, prob_threshold
            )

            if hemi_mask.sum() == 0:
                print(f"  WARNING: {hemi}_{category} empty in MNI space")
                continue

            # 2. Save MNI-space mask temporarily
            tmp_mni = f'{tmp_dir}/{hemi}_{category}_mni.nii.gz'
            mni_img = nib.Nifti1Image(hemi_mask.astype(np.float32), atlas_img.affine)
            nib.save(mni_img, tmp_mni)

            # 3. Warp to native space
            tmp_native = f'{tmp_dir}/{hemi}_{category}_native.nii.gz'
            warp_mask_to_native(tmp_mni, ref_brain, mni2anat, tmp_native)

            # 4. Load native-space mask and dilate
            native_img = nib.load(tmp_native)
            native_data = native_img.get_fdata() > 0

            if dilation_iters > 0:
                native_data = binary_dilation(native_data, iterations=dilation_iters)

            n_voxels = native_data.sum()

            # 5. Save final searchmask
            out_img = nib.Nifti1Image(native_data.astype(np.float32), native_img.affine)
            nib.save(out_img, output_file)

            print(f"  {hemi}_{category}: {n_voxels:,} voxels "
                  f"({', '.join(parcel_info['names'])})")
            n_created += 1

    # Cleanup temp files
    if not dry_run and os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    return n_created


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Create Harvard-Oxford searchmasks')
    parser.add_argument('--sub', type=str, help='Single subject (e.g., 004)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without creating')
    parser.add_argument('--threshold', type=int, default=PROB_THRESHOLD,
                        help=f'Probability threshold %% (default: {PROB_THRESHOLD})')
    parser.add_argument('--no-dilate', action='store_true', help='Skip dilation')
    args = parser.parse_args()

    prob_threshold = args.threshold
    dilation_iters = 0 if args.no_dilate else DILATION_ITERS

    print("=" * 60)
    print("CREATE SEARCHMASKS (Harvard-Oxford Atlas)")
    print("=" * 60)
    print(f"Probability threshold: {prob_threshold}%")
    print(f"Dilation iterations: {dilation_iters}")
    print(f"Categories: {list(CATEGORY_PARCELS.keys())}")
    print()

    # Load atlas once
    atlas_img, atlas_data = load_atlas()

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

    print(f"\nProcessing {len(subjects)} subjects")
    print()

    total_created = 0

    for sub_clean in subjects:
        sessions = get_sessions(f'sub-{sub_clean}')
        if not sessions:
            continue

        first_ses = sessions[0]
        print(f"=== sub-{sub_clean} ses-{first_ses:02d} ===")

        n = create_searchmasks_for_subject(
            f'sub-{sub_clean}', first_ses, atlas_img, atlas_data,
            prob_threshold, dilation_iters, args.dry_run
        )
        total_created += n

    print()
    print("=" * 60)
    print(f"{'Would create' if args.dry_run else 'Created'}: {total_created} searchmasks")
    print("=" * 60)


if __name__ == '__main__':
    main()