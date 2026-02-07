#!/usr/bin/env python3
"""
04_setup_anatomy.py - Skull strip, mirror, register to MNI, and pull back ROIs
Adapted from Ayzenberg 'register_mirror.py'
"""
import os
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import image
from long_pt_params import raw_dir, processed_dir, csv_file, mni_brain, skip_subs, get_sessions

# =============================================================================
# CONFIG: ROI PATHS
# Update these paths to match where your standard MNI parcels are stored
# =============================================================================
MNI_ROIS = {
    'ventral_visual': '/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/ventral_visual_cortex.nii.gz',
    'dorsal_visual': '/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/dorsal_visual_cortex.nii.gz'
}

def skull_strip(sub, ses):
    """Skull strip anatomical using BET"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    
    anat_file = f'{raw_dir}/sub-{sub_clean}/ses-{ses_str}/anat/sub-{sub_clean}_ses-{ses_str}_T1w.nii.gz'
    out_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    brain_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain.nii.gz'
    
    if not os.path.exists(anat_file):
        print(f'    ERROR: {anat_file} not found')
        return False
    
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(brain_file):
        print(f'    Skull strip: exists')
        return True
    
    print(f'    Skull stripping...')
    cmd = ['bet', anat_file, brain_file, '-R', '-B', '-m']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print(f'    Skull strip: FAILED')
        return False

def create_mirror(sub, ses, intact_hemi):
    """Create mirrored brain for hemispherectomy patients"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    
    anat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    brain_file = f'{anat_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain.nii.gz'
    mask_file = f'{anat_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_mask.nii.gz'
    mirror_file = f'{anat_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_mirrored.nii.gz'
    
    if not os.path.exists(brain_file): return False
    if os.path.exists(mirror_file):
        print(f'    Mirror: exists')
        return True
    
    try:
        print(f'    Creating mirror brain...')
        anat_img = image.load_img(brain_file)
        anat_data = anat_img.get_fdata()
        mask_data = image.load_img(mask_file).get_fdata()
        
        mid_x = anat_data.shape[0] // 2
        anat_flip = anat_data[::-1, :, :]
        anat_mirror = anat_data.copy()
        
        # Ayzenberg logic: Replace missing hemi with flipped intact hemi
        if intact_hemi.lower() == 'left':
            anat_mirror[mid_x:, :, :] = anat_flip[mid_x:, :, :] # Fill Right with Left
        else:
            anat_mirror[:mid_x, :, :] = anat_flip[:mid_x, :, :] # Fill Left with Right
            
        nib.save(nib.Nifti1Image(anat_mirror, anat_img.affine), mirror_file)
        return True
    except Exception as e:
        print(f'    Mirror FAILED: {e}')
        return False

def register_and_pullback_rois(sub, ses, is_patient):
    """Register to MNI and warp standard ROIs back to Native Space"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    
    anat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    roi_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/rois'
    
    brain_file = f'{anat_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain.nii.gz'
    mirror_file = f'{anat_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_mirrored.nii.gz'
    
    # Files for registration
    reg_input = mirror_file if is_patient and os.path.exists(mirror_file) else brain_file
    mat_anat2std = f'{anat_dir}/anat2stand.mat'
    mat_std2anat = f'{anat_dir}/mni2anat.mat'
    
    if not os.path.exists(reg_input): return False
    
    # 1. Register Anat -> MNI
    if not os.path.exists(mat_anat2std):
        print("    Registering to MNI...")
        cmd = ['flirt', '-in', reg_input, '-ref', mni_brain, '-omat', mat_anat2std, 
               '-bins', '256', '-cost', 'corratio', '-dof', '12']
        subprocess.run(cmd, check=True)
        
        # Create Inverse (MNI -> Anat)
        subprocess.run(['convert_xfm', '-omat', mat_std2anat, '-inverse', mat_anat2std], check=True)

    # 2. Warp ROIs MNI -> Anat
    os.makedirs(roi_dir, exist_ok=True)
    for roi_name, roi_path in MNI_ROIS.items():
        out_roi = f'{roi_dir}/{roi_name}.nii.gz'
        
        if not os.path.exists(roi_path):
            print(f"    SKIP ROI: {roi_name} (Source file missing)")
            continue
            
        if not os.path.exists(out_roi):
            print(f"    Warping ROI: {roi_name}")
            # Apply Inverse Transform (MNI -> Native)
            cmd = ['flirt', '-in', roi_path, '-ref', brain_file, '-out', out_roi,
                   '-applyxfm', '-init', mat_std2anat, '-interp', 'nearestneighbour']
            subprocess.run(cmd, check=True)
            
            # Binarize
            subprocess.run(['fslmaths', out_roi, '-bin', out_roi], check=True)

    return True

def main():
    print('Processing anatomy and ROIs...')
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        sub = row['sub'].replace('sub-', '')
        if sub in skip_subs: continue
        
        is_patient = row['patient'] == 1
        intact_hemi = row['intact_hemi']
        sessions = get_sessions(sub, df)
        
        print(f'\nsub-{sub}')
        for ses in sessions:
            skull_strip(sub, ses)
            if is_patient:
                create_mirror(sub, ses, intact_hemi)
            register_and_pullback_rois(sub, ses, is_patient)

if __name__ == '__main__':
    main()
