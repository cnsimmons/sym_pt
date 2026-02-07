#!/usr/bin/env python3
"""
03_setup_anatomy.py - Preprocess anatomy, Split Bilateral ROIs, and Warp to Native
"""
import os
import glob
import subprocess
import shutil
from sym_pt_params import (processed_dir, roi_dir, mni_brain, mni_2mm, 
                           skip_subs, get_sessions)

def run_command(cmd):
    """Run shell command and print output"""
    print(f"RUNNING: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing: {cmd}")
        # Don't crash the whole script, just raise to catch in main
        raise e

def split_bilateral_masks():
    """
    Check for massive bilateral masks (Ventral/Dorsal) and split them 
    into Left/Right MNI files before processing.
    Assumes 1mm MNI (182x218x182). Midline is 91.
    """
    print("\n--- Checking for Bilateral Masks to Split ---")
    
    # Map: 'SourceFile' -> ['LeftName', 'RightName']
    to_split = {
        'ventral_visual_cortex.nii.gz': ['lVentral.nii.gz', 'rVentral.nii.gz'],
        'dorsal_visual_cortex.nii.gz': ['lDorsal.nii.gz', 'rDorsal.nii.gz']
    }

    for source, (l_name, r_name) in to_split.items():
        source_path = f"{roi_dir}/{source}"
        l_path = f"{roi_dir}/{l_name}"
        r_path = f"{roi_dir}/{r_name}"

        if os.path.exists(source_path):
            # Split Right (0 to 91 in X)
            if not os.path.exists(r_path):
                print(f"  Splitting {source} -> {r_name}...")
                # fslmaths -roi <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> <tmin> <tsize>
                # MNI X-dim is 182. Right is 0-91.
                run_command(f"fslmaths {source_path} -roi 0 91 0 -1 0 -1 0 1 {r_path}")

            # Split Left (91 to end in X)
            if not os.path.exists(l_path):
                print(f"  Splitting {source} -> {l_name}...")
                run_command(f"fslmaths {source_path} -roi 91 -1 0 -1 0 -1 0 1 {l_path}")

def create_mirror_brain(anat_brain):
    """Create a mirrored brain to help registration for lesion patients."""
    output_base = anat_brain.replace('_brain.nii.gz', '_mirror')
    
    if not os.path.exists(f"{output_base}.nii.gz"):
        # 1. Flip (x-axis)
        run_command(f"fslswapdim {anat_brain} -x y z {output_base}_flipped")
        
        # 2. Register flipped to original (rigid body) 
        run_command(f"flirt -in {output_base}_flipped -ref {anat_brain} "
                    f"-out {output_base}_registered -omat {output_base}.mat -dof 6")
        
        # 3. Average (Original + Mirrored) / 2
        run_command(f"fslmaths {anat_brain} -add {output_base}_registered "
                    f"-div 2 {output_base}")
    
    return f"{output_base}.nii.gz"

def process_subject(sub, ses):
    print(f"\nProcessing {sub} Session {ses}...")
    
    base_dir = f"{processed_dir}/sub-{sub}/ses-{ses:02d}"
    anat_dir = f"{base_dir}/anat"
    roi_out_dir = f"{base_dir}/derivatives/rois"
    
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(roi_out_dir, exist_ok=True)
    
    # PATHS
    t1_head = f"{anat_dir}/T1w.nii.gz"
    t1_brain = f"{anat_dir}/T1w_brain.nii.gz"
    
    # 1. Find/Copy T1
    if not os.path.exists(t1_head):
        from sym_pt_params import raw_dir
        # Try multiple BIDS patterns
        patterns = [
            f"{raw_dir}/sub-{sub}/ses-{ses:02d}/anat/*T1w.nii.gz",
            f"{raw_dir}/sub-{sub}/anat/*T1w.nii.gz" # If single anat for multiple sessions
        ]
        found = []
        for p in patterns:
            found = glob.glob(p)
            if found: break
            
        if found:
            print(f"  Copying T1 from {found[0]}")
            shutil.copyfile(found[0], t1_head)
        else:
            print(f"  WARNING: No T1 found for {sub} ses-{ses}")
            return

    # 2. Skull Strip
    if not os.path.exists(t1_brain):
        print("  Skull stripping...")
        run_command(f"bet {t1_head} {t1_brain} -R -f 0.5 -g 0")

    # 3. Create Mirror Brain
    mirror_brain = create_mirror_brain(t1_brain)
    
    # 4. Register Mirror -> MNI
    mni_tfm = f"{anat_dir}/native_to_mni"
    
    if not os.path.exists(f"{mni_tfm}_warp.nii.gz"):
        print("  Registration (FLIRT + FNIRT)...")
        # FLIRT (Linear)
        run_command(f"flirt -in {mirror_brain} -ref {mni_brain} "
                    f"-out {mni_tfm}_linear -omat {mni_tfm}.mat")
        # FNIRT (Non-linear) - Heavy lifting
        run_command(f"fnirt --in={t1_head} --aff={mni_tfm}.mat --cout={mni_tfm}_warp "
                    f"--config=T1_2_MNI152_2mm --ref={mni_2mm}")
        
    # 5. Inverse Warp (MNI -> Native)
    mni_to_native_warp = f"{anat_dir}/mni_to_native_warp.nii.gz"
    if not os.path.exists(mni_to_native_warp):
        print("  Creating Inverse Warp...")
        run_command(f"invwarp --ref={t1_brain} --warp={mni_tfm}_warp "
                    f"--out={mni_to_native_warp}")

    # 6. Warp ROIs (MNI -> Native)
    # We warp everything in roi_dir (including our newly split ventral/dorsal masks)
    source_rois = glob.glob(f"{roi_dir}/*.nii.gz")
    
    for roi in source_rois:
        roi_name = os.path.basename(roi).replace('.nii.gz', '')
        
        # SKIP the original massive bilateral files, only warp the split ones
        if roi_name in ['ventral_visual_cortex', 'dorsal_visual_cortex']:
            continue
            
        out_roi = f"{roi_out_dir}/{roi_name}.nii.gz"
        
        if not os.path.exists(out_roi):
            print(f"  Warping {roi_name}...")
            run_command(f"applywarp --ref={t1_brain} --in={roi} --warp={mni_to_native_warp} "
                        f"--out={out_roi} --interp=nn")

def main():
    # Step 0: Ensure masks are split before we start looping subjects
    split_bilateral_masks()
    
    # Iterate subjects
    subs = sorted([d for d in os.listdir(processed_dir) if d.startswith('sub-')])
    for sub_dir in subs:
        sub = sub_dir.replace('sub-', '')
        if sub in skip_subs: continue
            
        sessions = get_sessions(f"sub-{sub}")
        for ses in sessions:
            try:
                process_subject(sub, ses)
            except Exception as e:
                print(f"FAILED on {sub} ses-{ses}: {e}")

if __name__ == "__main__":
    main()
