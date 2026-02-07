#!/usr/bin/env python3
"""
03_setup_anatomy.py - Preprocess anatomy and warp MNI ROIs to Native Space
"""
import os
import glob
import subprocess
import shutil
# Import from your config file
from sym_pt_params import (processed_dir, roi_dir, mni_brain, mni_2mm, 
                           skip_subs, get_sessions)

def run_command(cmd):
    """Run shell command and print output/errors"""
    print(f"RUNNING: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing: {cmd}")
        raise e

def create_mirror_brain(anat_brain, anat_head):
    """
    Create a mirrored brain to help registration for lesion patients.
    Flipping the healthy hemisphere to cover the lesion.
    """
    output_base = anat_brain.replace('_brain.nii.gz', '_mirror')
    
    # 1. Flip the brain (x-axis)
    # Note: fslswapdim output doesn't need extension if provided in filename
    if not os.path.exists(f"{output_base}_flipped.nii.gz"):
        run_command(f"fslswapdim {anat_brain} -x y z {output_base}_flipped")
    
    # 2. Register flipped to original (rigid body, 6 DOF) 
    # This handles slight head tilt so the mirror matches perfectly
    if not os.path.exists(f"{output_base}_registered.nii.gz"):
        run_command(f"flirt -in {output_base}_flipped -ref {anat_brain} "
                    f"-out {output_base}_registered -omat {output_base}.mat -dof 6")
    
    # 3. Average them (chimeric brain)
    # (Original + Mirrored) / 2
    if not os.path.exists(f"{output_base}.nii.gz"):
        run_command(f"fslmaths {anat_brain} -add {output_base}_registered "
                    f"-div 2 {output_base}")
    
    return f"{output_base}.nii.gz"

def process_subject(sub, ses):
    """Run anatomical preprocessing for one session"""
    print(f"\nProcessing {sub} Session {ses}...")
    
    # Define directories
    base_dir = f"{processed_dir}/sub-{sub}/ses-{ses:02d}"
    anat_dir = f"{base_dir}/anat"
    roi_out_dir = f"{base_dir}/derivatives/rois"
    
    # Ensure directories exist
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(roi_out_dir, exist_ok=True)
    
    # PATHS
    t1_head = f"{anat_dir}/T1w.nii.gz"
    t1_brain = f"{anat_dir}/T1w_brain.nii.gz"
    
    # 1. Locate Raw Anatomy
    # If T1 doesn't exist in our processed folder, find it in RAW and copy it.
    if not os.path.exists(t1_head):
        # Search pattern for BIDS T1w
        # This looks in: /lab_data/.../sub-XX/ses-XX/anat/*T1w.nii.gz
        # We need to import raw_dir from params to be safe, or hardcode the search
        from sym_pt_params import raw_dir
        raw_t1_search = f"{raw_dir}/sub-{sub}/ses-{ses:02d}/anat/*T1w.nii.gz"
        found_t1s = glob.glob(raw_t1_search)
        
        if found_t1s:
            print(f"  Copying T1 from {found_t1s[0]}")
            shutil.copyfile(found_t1s[0], t1_head)
        else:
            print(f"  WARNING: No T1 found for {sub} ses-{ses} in {raw_t1_search}")
            return # Skip this session if no anatomy

    # 2. Skull Strip (BET)
    if not os.path.exists(t1_brain):
        print("  Skull stripping...")
        # -R = robust, -f 0.5 = standard fractional intensity
        run_command(f"bet {t1_head} {t1_brain} -R -f 0.5 -g 0")

    # 3. Create Mirror Brain
    print("  Creating Mirror Brain...")
    mirror_brain = create_mirror_brain(t1_brain, t1_head)
    
    # 4. Register Mirror -> MNI
    # This creates the transform we need to bring ROIs *back* to the patient
    mni_tfm = f"{anat_dir}/native_to_mni"
    
    # Step 4a: Linear Registration (FLIRT)
    if not os.path.exists(f"{mni_tfm}.mat"):
        print("  Running FLIRT (Linear Registration)...")
        run_command(f"flirt -in {mirror_brain} -ref {mni_brain} "
                    f"-out {mni_tfm}_linear -omat {mni_tfm}.mat")
    
    # Step 4b: Non-Linear Registration (FNIRT) - THE SLOW PART
    if not os.path.exists(f"{mni_tfm}_warp.nii.gz"):
        print("  Running FNIRT (Non-Linear Registration)...")
        # Note: We use the *head* (T1w) for FNIRT, guided by the linear transform of the *mirror brain*
        run_command(f"fnirt --in={t1_head} --aff={mni_tfm}.mat --cout={mni_tfm}_warp "
                    f"--config=T1_2_MNI152_2mm --ref={mni_2mm}")
        
    # 5. Create Inverse Warp (MNI -> Native)
    mni_to_native_warp = f"{anat_dir}/mni_to_native_warp.nii.gz"
    if not os.path.exists(mni_to_native_warp):
        print("  Creating Inverse Warp...")
        run_command(f"invwarp --ref={t1_brain} --warp={mni_tfm}_warp "
                    f"--out={mni_to_native_warp}")

    # 6. Warp ROIs (MNI -> Native)
    # This takes every .nii.gz file in your central ROI folder and warps it to this patient
    source_rois = glob.glob(f"{roi_dir}/*.nii.gz")
    
    if not source_rois:
        print(f"  WARNING: No source ROIs found in {roi_dir}. Skipping ROI warping.")
        return

    for roi in source_rois:
        roi_name = os.path.basename(roi).replace('.nii.gz', '')
        out_roi = f"{roi_out_dir}/{roi_name}.nii.gz"
        
        if not os.path.exists(out_roi):
            print(f"  Warping ROI: {roi_name}...")
            # inter=nn (Nearest Neighbor) keeps the ROI binary (0 or 1), no fuzzy edges
            run_command(f"applywarp --ref={t1_brain} --in={roi} --warp={mni_to_native_warp} "
                        f"--out={out_roi} --interp=nn")

def main():
    print("Starting Anatomy Setup...")
    
    # Get list of subjects from processed directory
    # Only process subjects that have folders there
    subs = sorted([d for d in os.listdir(processed_dir) if d.startswith('sub-')])
    
    if not subs:
        print(f"No subject directories found in {processed_dir}. Did you run 01_organize.py?")
        return

    for sub_dir in subs:
        sub = sub_dir.replace('sub-', '')
        
        if sub in skip_subs:
            print(f"Skipping {sub} (in skip list)")
            continue
            
        sessions = get_sessions(f"sub-{sub}")
        if not sessions:
            print(f"No sessions found for {sub} in CSV.")
            continue

        for ses in sessions:
            try:
                process_subject(sub, ses)
            except Exception as e:
                print(f"FAILED on {sub} ses-{ses}: {e}")
                # We continue to the next subject instead of crashing entirely
                continue

if __name__ == "__main__":
    main()
