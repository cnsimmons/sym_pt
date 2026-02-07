#!/usr/bin/env python3
"""
06_run_feat.py - Execute FEAT for all subjects
"""
import os
import subprocess
import pandas as pd
from long_pt_params import processed_dir, csv_file, task, skip_subs, get_sessions, get_runs


def run_feat(sub, ses, run):
    """Run FEAT for one run"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    run_str = f'{run:02d}'
    
    fsf_file = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/{task}/run-{run_str}/1stLevel.fsf'
    feat_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/{task}/run-{run_str}/1stLevel.feat'
    
    if not os.path.exists(fsf_file):
        print(f'    Run {run}: FSF not found')
        return False
    
    if os.path.exists(feat_dir):
        print(f'    Run {run}: FEAT exists, skipping')
        return True
    
    print(f'    Run {run}: running FEAT...')
    try:
        subprocess.run(['feat', fsf_file], check=True)
        print(f'    Run {run}: done')
        return True
    except subprocess.CalledProcessError as e:
        print(f'    Run {run}: FAILED')
        return False


def main():
    print('Running FEAT analyses...')
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
                run_feat(sub, ses, run)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
