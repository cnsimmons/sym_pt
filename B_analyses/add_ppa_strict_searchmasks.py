#!/usr/bin/env python3
"""
add_ppa_strict_searchmasks.py — create house_PPA_strict searchmasks only.

Uses ONLY Harvard-Oxford indices 33 + 34 (Parahippocampal Gyrus anterior +
posterior) — i.e. PPA WITHOUT the Lingual Gyrus (35). This is the tight,
literature-canonical PPA. The existing house_PPA = [33,34,35] mask is left
untouched, so both definitions coexist.

Matches 01_create_searchmasks.py pipeline conventions:
  - 25% probability threshold
  - NO dilation (Liu/Ayzenberg do not dilate)
  - FLIRT linear MNI->native, nearest-neighbour
  - Overwrites only the house_PPA_strict file (additive; never touches others)

Usage:
  python add_ppa_strict_searchmasks.py
  python add_ppa_strict_searchmasks.py --sub 004
  python add_ppa_strict_searchmasks.py --dry-run
"""
import os, sys, argparse, subprocess, shutil
import numpy as np
import nibabel as nib

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

# ── Configuration ────────────────────────────────────────────────────────────
FSLDIR         = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
PROB_ATLAS     = f'{FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz'
PROB_THRESHOLD = 25
DILATION_ITERS = 0  # match 01_create_searchmasks.py — NO dilation

PARCEL = {
    'house_PPA_strict': {
        'indices': [33, 34],
        'names':   ['Parahippocampal anterior', 'Parahippocampal posterior'],
    },
}

# ── Helpers (self-contained, mirror 01_create_searchmasks.py) ────────────────
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
    midpoint = atlas_data.shape[0] // 2  # 45
    out = np.zeros_like(binary)
    if hemisphere == 'l':
        out[midpoint:, :, :] = binary[midpoint:, :, :]
    elif hemisphere == 'r':
        out[:midpoint, :, :] = binary[:midpoint, :, :]
    else:
        raise ValueError(f"hemisphere must be 'l' or 'r', got {hemisphere!r}")
    return out


def warp_mask_to_native(mni_mask, ref_brain, mni2anat, out_path):
    subprocess.run(
        ['flirt', '-in', mni_mask, '-ref', ref_brain, '-out', out_path,
         '-applyxfm', '-init', mni2anat, '-interp', 'nearestneighbour'],
        check=True, capture_output=True)
    subprocess.run(['fslmaths', out_path, '-bin', out_path],
                   check=True, capture_output=True)


def create_ppa_strict_for_subject(sub, ses, atlas_img, atlas_data, dry_run=False):
    ses_str = f'ses-{ses:02d}'
    ses_dir = os.path.join(processed_dir, sub, ses_str)
    roi_dir = os.path.join(ses_dir, 'ROIs')

    ref_brain = os.path.join(ses_dir, 'anat', 'T1w_brain.nii.gz')
    mni2anat  = os.path.join(ses_dir, 'anat', 'mni2anat.mat')

    if not os.path.exists(ref_brain) or not os.path.exists(mni2anat):
        print(f"  skip: missing ref_brain or mni2anat")
        return 0

    if not dry_run:
        os.makedirs(roi_dir, exist_ok=True)
    tmp_dir = f'{roi_dir}/_tmp_ppa_strict'
    if not dry_run:
        os.makedirs(tmp_dir, exist_ok=True)

    n_created = 0
    info = PARCEL['house_PPA_strict']
    for hemi in ['l', 'r']:
        out_file = f'{roi_dir}/{hemi}_house_PPA_strict_searchmask.nii.gz'

        if dry_run:
            print(f"  would create: {out_file} [{', '.join(info['names'])}]")
            n_created += 1
            continue

        # 1. Build MNI hemisphere mask
        hemi_mask = extract_hemisphere_mask(
            atlas_data, info['indices'], hemi, PROB_THRESHOLD)
        if hemi_mask.sum() == 0:
            print(f"  WARN: {hemi}_house_PPA_strict empty at {PROB_THRESHOLD}% thresh")
            continue
        tmp_mni = f'{tmp_dir}/{hemi}_ppa_strict_mni.nii.gz'
        nib.save(nib.Nifti1Image(hemi_mask.astype(np.float32), atlas_img.affine),
                 tmp_mni)

        # 2. Warp to native (overwrites only this file)
        warp_mask_to_native(tmp_mni, ref_brain, mni2anat, out_file)

        # 3. Optional dilation (default 0 = none)
        if DILATION_ITERS > 0:
            from scipy.ndimage import binary_dilation
            ni = nib.load(out_file)
            data = binary_dilation(ni.get_fdata() > 0, iterations=DILATION_ITERS)
            nib.save(nib.Nifti1Image(data.astype(np.float32), ni.affine), out_file)

        final = nib.load(out_file).get_fdata() > 0
        print(f"  {hemi}_house_PPA_strict: {final.sum():,} voxels "
              f"[{', '.join(info['names'])}]")
        n_created += 1

    if not dry_run and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    return n_created


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub',     type=str, help='Single subject (e.g., 004)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("ADD house_PPA_strict SEARCHMASK (PHG ant+post, no Lingual)")
    print("=" * 60)
    print(f"Parcels:   H-O indices 33 + 34 (PHG anterior + posterior)")
    print(f"Threshold: {PROB_THRESHOLD}%, Dilation: {DILATION_ITERS}")
    print(f"Note:      additive — does NOT modify existing house_PPA [33,34,35]")
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
        total += create_ppa_strict_for_subject(
            f'sub-{sub_clean}', first_ses, atlas_img, atlas_data, args.dry_run)

    print(f"\n{'Would create' if args.dry_run else 'Created'}: {total} masks")


if __name__ == '__main__':
    main()