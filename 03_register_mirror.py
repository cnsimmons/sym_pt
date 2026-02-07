#!/usr/bin/env python3
"""
03_register_mirror.py - Stage ROIs, Mirror Brain (patients), Register to MNI, Warp ROIs
  - Controls: standard FLIRT on actual brain
  - Patients: hemisphere-aware mirroring, FLIRT on mirror
  - FLIRT-only (dof 12) both directions, matching hemispace pipeline
Usage:
  python 03_register_mirror.py          # Process ALL subjects
  python 03_register_mirror.py --sub 022 # Process ONLY subject 022
"""
import os
import glob
import subprocess
import shutil
import argparse
import numpy as np
import nibabel as nib
from sym_pt_params import (processed_dir, raw_dir, roi_dir, roi_source_lib,
                           mni_brain, skip_subs,
                           is_patient, get_sessions, get_sub_info)


def run_cmd(cmd):
    print(f"  CMD: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# ── ROI Staging ──────────────────────────────────────────────────────────────

def stage_rois():
    """Copy ROIs from source library and split bilateral masks."""
    print(f"\n--- Staging ROIs from {roi_source_lib} ---")
    os.makedirs(roi_dir, exist_ok=True)

    kw = ['FFA', 'PPA', 'LOC', 'OFA', 'RSC', 'OPA', 'TOS', 'PFS', 'LO',
          'ventral_visual', 'dorsal_visual']

    n = 0
    for root, _, files in os.walk(roi_source_lib):
        for f in files:
            if f.endswith('.nii.gz') and any(k in f for k in kw):
                dst = os.path.join(roi_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(root, f), dst)
                    n += 1
    print(f"  Copied {n} new ROI files")

    splits = {
        'ventral_visual_cortex.nii.gz': ('lVentral.nii.gz', 'rVentral.nii.gz'),
        'dorsal_visual_cortex.nii.gz':  ('lDorsal.nii.gz',  'rDorsal.nii.gz'),
    }
    for src_name, (l_name, r_name) in splits.items():
        src = f"{roi_dir}/{src_name}"
        if not os.path.exists(src):
            continue
        for name, roi_args in [(r_name, "0 91 0 -1 0 -1 0 1"),
                                (l_name, "91 -1 0 -1 0 -1 0 1")]:
            out = f"{roi_dir}/{name}"
            if not os.path.exists(out):
                print(f"  Splitting {src_name} -> {name}")
                run_cmd(f"fslmaths {src} -roi {roi_args} {out}")


# ── Mirror Brain (patients only) ────────────────────────────────────────────

def create_mirror_brain(brain_file, mask_file, mirror_file, intact_hemi):
    """
    Hemisphere-aware mirroring for hemispherectomy patients.
    Copies intact hemisphere to fill the resected side.
    """
    if os.path.exists(mirror_file):
        print("  Mirror brain already exists")
        return True

    print(f"  Creating mirror brain (intact hemi: {intact_hemi})")

    img = nib.load(brain_file)
    mask = nib.load(mask_file)
    data = img.get_fdata().copy()
    affine = img.affine

    mid_x = data.shape[0] // 2
    flipped = np.flip(data, axis=0)

    if intact_hemi.lower() == 'left':
        data[mid_x:, :, :] = flipped[mid_x:, :, :]
    else:
        data[:mid_x, :, :] = flipped[:mid_x, :, :]

    nib.save(nib.Nifti1Image(data, affine), mirror_file)
    print("  Mirror brain created")
    return True


# ── Registration (FLIRT only, both directions) ───────────────────────────────

def register_to_mni(sub, anat_dir, t1_brain, pt, intact_hemi):
    """
    FLIRT registration (dof 12).
    - Patients: compute anat2stand.mat from mirror brain
    - Controls: compute anat2stand.mat from actual brain
    - Separate FLIRT call for mni2anat.mat (inverse direction)
    """
    mask_file = f"{anat_dir}/T1w_brain_mask.nii.gz"
    mirror_file = f"{anat_dir}/T1w_brain_mirrored.nii.gz"
    anat2stand = f"{anat_dir}/anat2stand.mat"
    mni2anat = f"{anat_dir}/mni2anat.mat"
    stand_brain = f"{anat_dir}/T1w_brain_stand.nii.gz"

    # Patient mirroring
    if pt:
        if not os.path.exists(mask_file):
            print("  ERROR: No brain mask — BET needs -m flag")
            return False
        if not create_mirror_brain(t1_brain, mask_file, mirror_file, intact_hemi):
            return False

    flirt_input = mirror_file if pt else t1_brain

    # Forward: native -> MNI
    if not os.path.exists(anat2stand):
        print(f"  FLIRT: {'mirror' if pt else 'brain'} -> MNI")
        run_cmd(f"flirt -in {flirt_input} -ref {mni_brain} "
                f"-omat {anat2stand} -bins 256 -cost corratio "
                f"-searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12")

    # Apply to original brain
    if not os.path.exists(stand_brain):
        print("  Applying transform to original brain...")
        run_cmd(f"flirt -in {t1_brain} -ref {mni_brain} "
                f"-out {stand_brain} -applyxfm -init {anat2stand} -interp trilinear")

    # Inverse: invert anat2stand.mat
    if not os.path.exists(mni2anat):
        print("  Inverting anat2stand.mat...")
        run_cmd(f"convert_xfm -omat {mni2anat} -inverse {anat2stand}")

    return True


# ── Warp ROIs to Native ─────────────────────────────────────────────────────

def warp_rois(t1_brain, anat_dir, roi_out_dir):
    """Apply mni2anat.mat to bring MNI ROIs into native space, then binarize."""
    os.makedirs(roi_out_dir, exist_ok=True)
    mni2anat = f"{anat_dir}/mni2anat.mat"
    skip_names = {'ventral_visual_cortex', 'dorsal_visual_cortex'}

    for roi in glob.glob(f"{roi_dir}/*.nii.gz"):
        name = os.path.basename(roi).replace('.nii.gz', '')
        if name in skip_names:
            continue
        out = f"{roi_out_dir}/{name}.nii.gz"
        if not os.path.exists(out):
            print(f"  Warping {name}...")
            run_cmd(f"flirt -in {roi} -ref {t1_brain} "
                    f"-out {out} -applyxfm -init {mni2anat} -interp trilinear")
            run_cmd(f"fslmaths {out} -bin {out}")


# ── Subject Processing ───────────────────────────────────────────────────────

def process_subject(sub, ses):
    pt = is_patient(sub)
    info = get_sub_info(sub, ses)
    intact_hemi = info.get('intact_hemi', '')

    print(f"\n{'='*60}")
    print(f"  sub-{sub}  ses-{ses:02d}  ({'PATIENT' if pt else 'CONTROL'})")
    if pt:
        print(f"  intact_hemi: {intact_hemi}")
    print(f"{'='*60}")

    base = f"{processed_dir}/sub-{sub}/ses-{ses:02d}"
    anat_dir = f"{base}/anat"
    roi_out = f"{base}/derivatives/rois"
    os.makedirs(anat_dir, exist_ok=True)

    t1_head = f"{anat_dir}/T1w.nii.gz"
    t1_brain = f"{anat_dir}/T1w_brain.nii.gz"

    # Locate T1
    if not os.path.exists(t1_head):
        patterns = [
            f"{raw_dir}/sub-{sub}/ses-{ses:02d}/anat/*T1w.nii.gz",
            f"{raw_dir}/sub-{sub}/anat/*T1w.nii.gz",
        ]
        found = []
        for p in patterns:
            found = glob.glob(p)
            if found:
                break
        if found:
            print(f"  Copying T1 from {found[0]}")
            shutil.copyfile(found[0], t1_head)
        else:
            print(f"  WARNING: No T1 found — skipping")
            return

    # Skull strip
    if not os.path.exists(t1_brain):
        print("  Skull stripping (BET -R -B -m)...")
        run_cmd(f"bet {t1_head} {t1_brain} -R -B -m")

    # Register + warp ROIs
    if register_to_mni(sub, anat_dir, t1_brain, pt, intact_hemi):
        warp_rois(t1_brain, anat_dir, roi_out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anatomy & ROI Setup")
    parser.add_argument('--sub', type=str, help="Run only this subject (e.g., 022)")
    args = parser.parse_args()

    stage_rois()

    if args.sub:
        sub_clean = args.sub.replace('sub-', '')
        subs = [f'sub-{sub_clean}']
        print(f"--- SINGLE SUBJECT: {sub_clean} ---")
    else:
        subs = sorted(d for d in os.listdir(processed_dir) if d.startswith('sub-'))
        print(f"--- BATCH: {len(subs)} subjects ---")

    for sub_dir in subs:
        sub = sub_dir.replace('sub-', '')
        if sub in skip_subs:
            print(f"Skipping {sub} (skip list)")
            continue

        sessions = get_sessions(f"sub-{sub}")
        if not sessions:
            print(f"No sessions for {sub}")
            continue

        for ses in sessions:
            try:
                process_subject(sub, ses)
            except Exception as e:
                print(f"FAILED: sub-{sub} ses-{ses:02d}: {e}")
                continue


if __name__ == "__main__":
    main()