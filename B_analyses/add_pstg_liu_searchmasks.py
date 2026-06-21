#!/usr/bin/env python3
"""
add_pstg_liu_searchmasks.py — create word_pSTG_liu searchmasks only.

Uses ONLY Harvard-Oxford index 16 (Superior Temporal Gyrus posterior),
matching Liu (2025)'s posterior STG specification. Additive — does NOT
touch existing word_STG (indices 15+16).

Usage:
  python add_pstg_liu_searchmasks.py
  python add_pstg_liu_searchmasks.py --sub 004
  python add_pstg_liu_searchmasks.py --dry-run
"""
import os, sys, argparse, subprocess, shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

# ── Configuration ────────────────────────────────────────────────────────────
FSLDIR         = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
PROB_ATLAS     = f'{FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz'
PROB_THRESHOLD = 25
DILATION_ITERS = 1

PARCEL = {
    'word_pSTG_liu': {
        'indices': [16],
        'names':   ['Superior Temporal Gyrus posterior'],
    },
}

# ── Helpers (self-contained, copied from 01_create_searchmasks.py) ───────────
def load_atlas():
    print(f"Loading atlas: {PROB_ATLAS}")
    img = nib.load(PROB_ATLAS)
    data = img.get_fdata()
    print(f"  Shape: {data.shape}")
    return img, data


def extract_hemisphere_mask(atlas_data, indices, hemisphere, threshold):
    combined = np.zeros(atlas_data.shape[:3], dtype=float)
    for idx in indices:
        combined = np.maximum(combined, atlas_data[:, :, :, idx])
    binary = combined > threshold
    midpoint = atlas_data.shape[0] // 2
    out = np.zeros_like(binary)
    if hemisphere == 'l':
        out[midpoint:, :, :] = binary[midpoint:, :, :]
    elif hemisphere == 'r':
        out[:midpoint, :, :] = binary[:midpoint, :, :]
    return out


def warp_mask_to_native(mni_mask, ref_brain, mni2anat, out_path):
    subprocess.run(
        f'flirt -in {mni_mask} -ref {ref_brain} -out {out_path} '
        f'-applyxfm -init {mni2anat} -interp nearestneighbour'.split(),
        check=True, capture_output=True)
    subprocess.run(f'fslmaths {out_path} -bin {out_path}'.split(),
                   check=True, capture_output=True)


def create_pstg_for_subject(sub, ses, atlas_img, atlas_data, dry_run=False):
    ses_str = f'ses-{ses:02d}'
    ses_dir = os.path.join(processed_dir, sub, ses_str)
    roi_dir = os.path.join(ses_dir, 'ROIs')

    ref_brain = os.path.join(ses_dir, 'anat', 'T1w_brain.nii.gz')
    mni2anat  = os.path.join(ses_dir, 'anat', 'mni2anat.mat')

    if not os.path.exists(ref_brain) or not os.path.exists(mni2anat):
        print(f"  skip: missing ref_brain or mni2anat")
        return 0

    os.makedirs(roi_dir, exist_ok=True)
    tmp_dir = f'{roi_dir}/_tmp_pstg'
    if not dry_run:
        os.makedirs(tmp_dir, exist_ok=True)

    n_created = 0
    info = PARCEL['word_pSTG_liu']
    for hemi in ['l', 'r']:
        out_file = f'{roi_dir}/{hemi}_word_pSTG_liu_searchmask.nii.gz'

        if dry_run:
            print(f"  would create: {out_file}")
            n_created += 1
            continue

        # 1. Build MNI hemisphere mask
        hemi_mask = extract_hemisphere_mask(
            atlas_data, info['indices'], hemi, PROB_THRESHOLD)
        tmp_mni = f'{tmp_dir}/{hemi}_pstg_mni.nii.gz'
        nib.save(nib.Nifti1Image(hemi_mask.astype(np.float32), atlas_img.affine),
                 tmp_mni)

        # 2. Warp to native
        tmp_native = f'{tmp_dir}/{hemi}_pstg_native.nii.gz'
        warp_mask_to_native(tmp_mni, ref_brain, mni2anat, tmp_native)

        # 3. Dilate + save
        ni = nib.load(tmp_native)
        data = ni.get_fdata() > 0
        if DILATION_ITERS > 0:
            data = binary_dilation(data, iterations=DILATION_ITERS)
        nib.save(nib.Nifti1Image(data.astype(np.float32), ni.affine), out_file)

        print(f"  {hemi}_word_pSTG_liu: {data.sum():,} voxels")
        n_created += 1

    if not dry_run and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    return n_created


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub',     type=str)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("ADD word_pSTG_liu SEARCHMASK (Liu-matched posterior STG)")
    print("=" * 60)
    print(f"Parcel:    H-O index 16 only (STG posterior)")
    print(f"Threshold: {PROB_THRESHOLD}%, Dilation: {DILATION_ITERS}")
    print(f"Note:      does NOT modify existing word_STG")
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
    total = 0
    for sub_clean in subjects:
        sessions = get_sessions(sub_clean)
        if not sessions:
            continue
        first_ses = sessions[0]
        print(f"=== sub-{sub_clean} ses-{first_ses:02d} ===")
        total += create_pstg_for_subject(
            f'sub-{sub_clean}', first_ses, atlas_img, atlas_data, args.dry_run)

    print(f"\n{'Would create' if args.dry_run else 'Created'}: {total} masks")


if __name__ == '__main__':
    main()