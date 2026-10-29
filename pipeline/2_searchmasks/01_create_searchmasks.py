#!/usr/bin/env python3
"""
01_create_searchmasks.py — Create category-specific anatomical searchmasks.

Harvard-Oxford cortical probability atlas (25% threshold) warped to each
subject's native anatomical space via FLIRT. Masks are used for:
  - Cross-sectional peak-finding and RSA sphere placement
  - Sum-selectivity (Ayzenberg-style) normalization
  - Longitudinal analyses (same masks across all time points)

ROI definitions (validated against localizer literature):
  face_FFA       [37, 38]     TF posterior + TO Fusiform
                              (Schütz et al. 2019, Nat Comms)
  face_STS       [9, 12]      STG posterior + MTG posterior
                              (covers pSTS at STG/MTG border;
                               HO has no sulcus parcel)
  house_PPA      [33, 34, 35] Parahippocampal ant + post + Lingual
                              (canonical PPA at CoS/lingual junction;
                               Epstein & Kanwisher 1998; Weiner et al. 2018)
  house_TOS      [21, 22]     LOC sup + LOC inf
                              (TOS/OPA on transverse occipital sulcus)
  object_LOC     [21, 22]     LOC sup + LOC inf
  object_pF      [38]         TO Fusiform
  word_VWFA      [37, 38]     TF posterior + TO Fusiform
                              (canonical VWFA at OTS/fusiform border;
                               MNI ~-44,-58,-15; overlaps FFA anatomy —
                               contrast differentiates)
  word_STG       [8, 9]       STG anterior + posterior (Wernicke's)
  word_pSTG_liu  [9]          STG posterior only (tight Liu-spec reference)
  evc            [23, 31]     Intracalcarine + Cuneal (V1/V2 territory)

Overlap by design:
  - face_FFA and word_VWFA share anatomy [37, 38]; contrast differentiates.
    Required for face/word competition analyses in longitudinal data.
  - face_STS [9, 12] and word_STG [8, 9] share parcel 9 only.
  - word_pSTG_liu ⊂ word_STG (tighter subset for Liu RDM comparison).
  - object_LOC and house_TOS share [21, 22]; contrast differentiates.

Harvard-Oxford cortical volume index reference (0-indexed, 4D prob atlas):
   4  IFG pars triangularis           22  Lateral Occipital Cortex inferior
   5  IFG pars opercularis            23  Intracalcarine Cortex
   8  Superior Temporal Gyrus ant     29  Cingulate Gyrus posterior
   9  Superior Temporal Gyrus post    31  Cuneal Cortex
  12  Middle Temporal Gyrus post      33  Parahippocampal Gyrus anterior
  15  ITG temporooccipital            34  Parahippocampal Gyrus posterior
  16  Postcentral Gyrus               35  Lingual Gyrus
  21  Lateral Occipital Cortex sup    36  Temporal Fusiform Cortex anterior
                                      37  Temporal Fusiform Cortex posterior
                                      38  Temporal Occipital Fusiform Cortex

Usage:
  python 01_create_searchmasks.py              # All subjects, all masks
  python 01_create_searchmasks.py --sub 004    # Single subject
  python 01_create_searchmasks.py --dry-run    # Preview only
  python 01_create_searchmasks.py --threshold 50  # More conservative

Design choices:
  - 25% probability threshold (liberal, accommodates patient displacement)
  - No dilation (Liu/Ayzenberg do not dilate; threshold already liberal)
  - FLIRT linear MNI→native (stable near resections; FNIRT can warp tissue
    into lesion cavity)
  - Overwrites existing masks (no skip-if-exists; prevents stale files from
    prior index bugs)
"""
import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import nibabel as nib

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

# ── Configuration ─────────────────────────────────────────────────────────────

FSLDIR         = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
ATLAS_DIR      = f'{FSLDIR}/data/atlases/HarvardOxford'
PROB_ATLAS     = f'{ATLAS_DIR}/HarvardOxford-cort-prob-2mm.nii.gz'
PROB_THRESHOLD = 25  # % — liberal by design

# ── Category parcels ──────────────────────────────────────────────────────────

CATEGORY_PARCELS = {
    'face_FFA': {
        'indices': [37, 38],
        'names':   ['Temporal Fusiform posterior', 'Temporal Occipital Fusiform'],
    },
    'face_STS': {
        'indices': [9, 12],
        'names':   ['STG posterior', 'MTG posterior'],
    },
    'house_PPA': {
        'indices': [33, 34, 35],
        'names':   ['Parahippocampal anterior', 'Parahippocampal posterior',
                    'Lingual Gyrus'],
    },
    'house_TOS': {
        'indices': [21, 22],
        'names':   ['Lateral Occipital superior', 'Lateral Occipital inferior'],
    },
    'object_LOC': {
        'indices': [21, 22],
        'names':   ['Lateral Occipital superior', 'Lateral Occipital inferior'],
    },
    'object_pF': {
        'indices': [38],
        'names':   ['Temporal Occipital Fusiform'],
    },
    'word_VWFA': {
        'indices': [37, 38],
        'names':   ['Temporal Fusiform posterior', 'Temporal Occipital Fusiform'],
    },
    'word_STG': {
        'indices': [8, 9],
        'names':   ['STG anterior', 'STG posterior'],
    },
    'word_pSTG_liu': {
        'indices': [9],
        'names':   ['STG posterior'],
    },
    'evc': {
        'indices': [23, 31],
        'names':   ['Intracalcarine Cortex', 'Cuneal Cortex'],
    },
}

HEMIS = ['l', 'r']


# ── Core functions ────────────────────────────────────────────────────────────

def load_atlas():
    """Load the Harvard-Oxford probability atlas."""
    print(f"Loading atlas: {PROB_ATLAS}")
    img  = nib.load(PROB_ATLAS)
    data = img.get_fdata()
    print(f"  Shape: {data.shape} (x, y, z, n_regions={data.shape[3]})")
    return img, data


def extract_hemisphere_mask(atlas_data, region_indices, hemisphere,
                            threshold=PROB_THRESHOLD):
    """Combine probability maps for given parcels, threshold, split by hemi.

    FSL 2mm MNI152 (91x109x91, radiological convention):
      - x-midpoint at voxel 45
      - voxels [0:45]  → right hemisphere (MNI x > 0)
      - voxels [45:]   → left hemisphere  (MNI x < 0)
    """
    combined = np.zeros(atlas_data.shape[:3], dtype=float)
    for idx in region_indices:
        combined = np.maximum(combined, atlas_data[:, :, :, idx])

    binary_mask = combined > threshold
    midpoint    = atlas_data.shape[0] // 2  # 45

    hemi_mask = np.zeros_like(binary_mask)
    if hemisphere == 'l':
        hemi_mask[midpoint:, :, :] = binary_mask[midpoint:, :, :]
    elif hemisphere == 'r':
        hemi_mask[:midpoint, :, :] = binary_mask[:midpoint, :, :]
    else:
        raise ValueError(f"hemisphere must be 'l' or 'r', got {hemisphere!r}")
    return hemi_mask


def warp_mask_to_native(mni_mask_path, ref_brain, mni2anat_mat, output_path):
    """Warp MNI-space mask to native space via FLIRT (nearest-neighbour)."""
    subprocess.run(
        ['flirt', '-in', mni_mask_path, '-ref', ref_brain,
         '-out', output_path, '-applyxfm', '-init', mni2anat_mat,
         '-interp', 'nearestneighbour'],
        check=True, capture_output=True
    )
    subprocess.run(
        ['fslmaths', output_path, '-bin', output_path],
        check=True, capture_output=True
    )


def create_searchmasks_for_subject(sub_clean, ses, atlas_img, atlas_data,
                                   threshold=PROB_THRESHOLD, dry_run=False):
    """Generate all searchmasks for one subject-session. Overwrites existing."""
    ses_str   = f'{ses:02d}'
    anat_dir  = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    roi_dir   = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/ROIs'
    ref_brain = f'{anat_dir}/T1w_brain.nii.gz'
    mni2anat  = f'{anat_dir}/mni2anat.mat'

    if not os.path.exists(ref_brain):
        print(f"  SKIP: {ref_brain} missing")
        return 0
    if not os.path.exists(mni2anat):
        print(f"  SKIP: {mni2anat} missing")
        return 0

    if not dry_run:
        os.makedirs(roi_dir, exist_ok=True)

    tmp_dir = f'/tmp/searchmasks_sub-{sub_clean}'
    if not dry_run:
        os.makedirs(tmp_dir, exist_ok=True)

    n_created = 0

    for hemi in HEMIS:
        for category, parcel_info in CATEGORY_PARCELS.items():
            output_file = f'{roi_dir}/{hemi}_{category}_searchmask.nii.gz'

            if dry_run:
                print(f"  WOULD WRITE: {hemi}_{category}_searchmask.nii.gz "
                      f"[{', '.join(parcel_info['names'])}]")
                n_created += 1
                continue

            # 1. Build MNI-space hemisphere mask
            hemi_mask = extract_hemisphere_mask(
                atlas_data, parcel_info['indices'], hemi, threshold)

            if hemi_mask.sum() == 0:
                print(f"  WARN: {hemi}_{category} empty at {threshold}% thresh")
                continue

            # 2. Save MNI-space mask temporarily
            tmp_mni = f'{tmp_dir}/{hemi}_{category}_mni.nii.gz'
            nib.save(nib.Nifti1Image(hemi_mask.astype(np.float32),
                                     atlas_img.affine), tmp_mni)

            # 3. Warp to native space (overwrites if exists — prevents stale files)
            warp_mask_to_native(tmp_mni, ref_brain, mni2anat, output_file)

            # 4. Report final voxel count
            final = nib.load(output_file).get_fdata() > 0
            print(f"  {hemi}_{category:15s}: {final.sum():>6,} voxels "
                  f"[{', '.join(parcel_info['names'])}]")
            n_created += 1

    if not dry_run and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return n_created


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Create Harvard-Oxford searchmasks (overwrites existing).')
    parser.add_argument('--sub', type=str,
                        help='Single subject (e.g., 004)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without creating files')
    parser.add_argument('--threshold', type=int, default=PROB_THRESHOLD,
                        help=f'Probability threshold %% (default: {PROB_THRESHOLD})')
    args = parser.parse_args()

    print("=" * 64)
    print("CREATE SEARCHMASKS — Harvard-Oxford cortical atlas")
    print("=" * 64)
    print(f"Probability threshold: {args.threshold}%")
    print(f"Masks per hemisphere:  {len(CATEGORY_PARCELS)}")
    print(f"Categories: {list(CATEGORY_PARCELS.keys())}")
    print(f"Mode: {'DRY RUN (no files written)' if args.dry_run else 'WRITE (overwrites existing)'}")
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
            sub_clean, first_ses, atlas_img, atlas_data,
            args.threshold, args.dry_run)
        total_created += n

    print()
    print("=" * 64)
    print(f"{'Would write' if args.dry_run else 'Wrote'}: "
          f"{total_created} searchmasks")
    print("=" * 64)


if __name__ == '__main__':
    main()