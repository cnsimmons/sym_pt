#!/usr/bin/env python3
"""
01_organize.py - Setup directory structure for long_pt
"""
import os
import pandas as pd
from long_pt_params import raw_dir, processed_dir, csv_file, task, skip_subs, get_sessions, get_runs


def setup_subject(sub, sessions):
    """Create directory structure for one subject"""
    sub_clean = sub.replace('sub-', '')
    
    for ses in sessions:
        ses_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses:02d}'
        runs = get_runs(sub_clean, ses)
        
        if not runs:
            print(f'  WARNING: No runs found for ses-{ses:02d}')
            continue
        
        print(f'  Session {ses}: {len(runs)} runs')
        
        dirs = [
            f'{ses_dir}/timing',
            f'{ses_dir}/anat',
            f'{ses_dir}/derivatives/fsl/{task}',
            f'{ses_dir}/derivatives/qc'
        ]
        
        for run in runs:
            dirs.append(f'{ses_dir}/derivatives/fsl/{task}/run-{run:02d}')
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)


def main():
    print(f'Setting up directories in {processed_dir}')
    os.makedirs(processed_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        sub = row['sub'].replace('sub-', '')
        
        if sub in skip_subs:
            print(f'SKIP: {sub}')
            continue
        
        sessions = get_sessions(sub, df)
        print(f'\nsub-{sub} ({len(sessions)} sessions)')
        setup_subject(sub, sessions)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
