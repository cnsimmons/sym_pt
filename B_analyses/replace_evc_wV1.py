#!/usr/bin/env python3
"""
Replace EVC searchmasks with Mruczek V1 (warped from MNI to native via FLIRT).
One-off fix. Run after main searchmask generation.

Usage:
  python replace_evc_with_v1.py              # all subjects
  python replace_evc_with_v1.py --sub 004    # single subject
  python replace_evc_with_v1.py --dry-run
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import nibabel as nib

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

MRUCZEK_DIR = Path('/user_data/csimmon2/git_repos/ptoc/roiParcels/mruczek_parcels/binary')


def warp_to_native(src, ref_brain, mni2anat, out):
    subprocess.run(
        ['flirt', '-in', str(src), '-ref', ref_brain,
         '-out', out, '-applyxfm', '-init', mni2anat,
         '-interp', 'nearestneighbour'],
        check=True, capture_output=True)
    subprocess.run(['fslmaths', out, '-bin', out],
                   check=True, capture_output=True)


def replace_evc(sub_clean, ses, dry_run=False):
    ses_str   = f'{ses:02d}'
    anat_dir  = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    roi_dir   = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/ROIs'
    ref_brain = f'{anat_dir}/T1w_brain.nii.gz'
    mni2anat  = f'{anat_dir}/mni2anat.mat'

    if not os.path.exists(ref_brain) or not os.path.exists(mni2anat):
        print(f'  SKIP: missing T1 or transform')
        return 0

    n = 0
    for hemi in ['l', 'r']:
        src = MRUCZEK_DIR / f'{hemi}V1.nii.gz'
        out = f'{roi_dir}/{hemi}_evc_searchmask.nii.gz'
        if not src.exists():
            print(f'  WARN: {src} missing')
            continue
        if dry_run:
            print(f'  WOULD WRITE: {hemi}_evc_searchmask.nii.gz [Mruczek {hemi}V1]')
            n += 1
            continue
        warp_to_native(src, ref_brain, mni2anat, out)
        final = nib.load(out).get_fdata() > 0
        print(f'  {hemi}_evc: {final.sum():>6,} voxels [Mruczek {hemi}V1]')
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.sub:
        subjects = [args.sub.replace('sub-', '')]
    else:
        subjects = sorted(
            d.replace('sub-', '')
            for d in os.listdir(processed_dir)
            if d.startswith('sub-') and d.replace('sub-', '') not in skip_subs
        )

    print(f'Replacing EVC with Mruczek V1 for {len(subjects)} subjects\n')
    total = 0
    for sc in subjects:
        sessions = get_sessions(sc)
        if not sessions:
            continue
        first_ses = sessions[0]
        print(f'=== sub-{sc} ses-{first_ses:02d} ===')
        total += replace_evc(sc, first_ses, args.dry_run)
    print(f'\n{"Would write" if args.dry_run else "Wrote"}: {total} EVC masks')


if __name__ == '__main__':
    main()