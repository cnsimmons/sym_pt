#!/usr/bin/env python3
"""
04_register_mirror.py - Skull strip, mirror (for patients), register to MNI
"""
import os
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import image
from long_pt_params import raw_dir, processed_dir, csv_file, mni_brain, skip_subs, get_sessions


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
    
    if os.path.exists(brain_file):
        print(f'    Skull strip: exists')
        return True
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = ['bet', anat_file, brain_file, '-R', '-B', '-m']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f'    Skull strip: done')
        return True
    except subprocess.CalledProcessError as e:
        print(f'    Skull strip: FAILED')
        return False


def create_mirror(sub, ses, intact_hemi):
    """Create mirrored brain for hemispherectomy patients"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    
    out_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    brain_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain.nii.gz'
    mask_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_mask.nii.gz'
    mirror_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_mirrored.nii.gz'
    
    if not os.path.exists(brain_file) or not os.path.exists(mask_file):
        print(f'    Mirror: missing inputs')
        return False
    
    if os.path.exists(mirror_file):
        print(f'    Mirror: exists')
        return True
    
    try:
        anat_img = image.load_img(brain_file)
        anat_data = anat_img.get_fdata()
        affine = anat_img.affine
        
        mid_x = anat_data.shape[0] // 2
        mirrored = anat_data.copy()
        flipped = np.flip(anat_data, axis=0)
        
        if intact_hemi.lower() == 'left':
            mirrored[mid_x:, :, :] = flipped[mid_x:, :, :]
        else:
            mirrored[:mid_x, :, :] = flipped[:mid_x, :, :]
        
        nib.save(nib.Nifti1Image(mirrored, affine), mirror_file)
        print(f'    Mirror: done')
        return True
    except Exception as e:
        print(f'    Mirror: FAILED - {e}')
        return False


def register_to_mni(sub, ses, is_patient):
    """Register to MNI space"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    
    out_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat'
    brain_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain.nii.gz'
    mirror_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_mirrored.nii.gz'
    mat_file = f'{out_dir}/anat2stand.mat'
    inv_mat = f'{out_dir}/mni2anat.mat'
    out_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_T1w_brain_stand.nii.gz'
    
    # Use mirrored brain for registration if patient
    reg_input = mirror_file if is_patient and os.path.exists(mirror_file) else brain_file
    
    if not os.path.exists(reg_input):
        print(f'    Register: missing input')
        return False
    
    if os.path.exists(mat_file):
        print(f'    Register: exists')
        return True
    
    try:
        # Forward transform
        subprocess.run([
            'flirt', '-in', reg_input, '-ref', mni_brain,
            '-omat', mat_file, '-bins', '256', '-cost', 'corratio',
            '-searchrx', '-90', '90', '-searchry', '-90', '90',
            '-searchrz', '-90', '90', '-dof', '12'
        ], check=True, capture_output=True)
        
        # Apply to original brain
        subprocess.run([
            'flirt', '-in', brain_file, '-ref', mni_brain,
            '-out', out_file, '-applyxfm', '-init', mat_file
        ], check=True, capture_output=True)
        
        # Inverse transform
        subprocess.run([
            'convert_xfm', '-omat', inv_mat, '-inverse', mat_file
        ], check=True, capture_output=True)
        
        print(f'    Register: done')
        return True
    except subprocess.CalledProcessError as e:
        print(f'    Register: FAILED')
        return False


def process_subject(sub, ses, is_patient, intact_hemi):
    """Full anatomical processing for one session"""
    print(f'  Session {ses}:')
    
    if not skull_strip(sub, ses):
        return False
    
    if is_patient:
        if not create_mirror(sub, ses, intact_hemi):
            return False
    
    if not register_to_mni(sub, ses, is_patient):
        return False
    
    return True


def main():
    print('Processing anatomical data...')
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        sub = row['sub'].replace('sub-', '')
        
        if sub in skip_subs:
            print(f'SKIP: {sub}')
            continue
        
        is_patient = row['patient'] == 1
        intact_hemi = row['intact_hemi']
        sessions = get_sessions(sub, df)
        
        print(f'\nsub-{sub} (patient={is_patient}, intact={intact_hemi})')
        
        for ses in sessions:
            process_subject(sub, ses, is_patient, intact_hemi)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
