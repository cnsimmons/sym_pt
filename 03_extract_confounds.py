#!/usr/bin/env python3
"""
03_extract_confounds.py - Extract motion outliers using FSL
"""
import os
import subprocess
import pandas as pd
from long_pt_params import raw_dir, processed_dir, csv_file, task, fd_threshold, skip_subs, get_sessions, get_runs


def extract_confounds(sub, ses, run):
    """Extract motion outliers for one run"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    run_str = f'{run:02d}'
    
    func_file = f'{raw_dir}/sub-{sub_clean}/ses-{ses_str}/func/sub-{sub_clean}_ses-{ses_str}_task-{task}_run-{run_str}_bold.nii.gz'
    out_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/{task}/run-{run_str}'
    out_file = f'{out_dir}/sub-{sub_clean}_ses-{ses_str}_task-{task}_run-{run_str}_bold_spikes.txt'
    
    if not os.path.exists(func_file):
        print(f'    SKIP: {func_file} not found')
        return False
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        'fsl_motion_outliers',
        '-i', func_file,
        '-o', out_file,
        '--fd',
        f'--thresh={fd_threshold}',
        '--dummy=0'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f'    Run {run}: done')
        return True
    except subprocess.CalledProcessError as e:
        print(f'    Run {run}: FAILED - {e}')
        return False


def main():
    print('Extracting motion confounds...')
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        sub = row['sub'].replace('sub-', '')
        
        if sub in skip_subs:
            print(f'SKIP: {sub}')
            continue
        
        sessions = get_sessions(sub, df)
        print(f'\nsub-{sub}')
        
        for ses in sessions:
            runs = get_runs(sub, ses)
            print(f'  Session {ses}:')
            
            for run in runs:
                extract_confounds(sub, ses, run)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
