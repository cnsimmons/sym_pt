#!/usr/bin/env python3
"""
02_convert_timing.py - Convert BIDS events to FSL 3-column timing files
"""
import os
import pandas as pd
from sym_pt_params import raw_dir, processed_dir, csv_file, task, skip_subs, get_sessions, get_runs, conditions


def convert_run(sub, ses, run):
    """Convert events.tsv to FSL timing files for one run"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    run_str = f'{run:02d}'
    
    # Input events file
    events_file = f'{raw_dir}/sub-{sub_clean}/ses-{ses_str}/func/sub-{sub_clean}_ses-{ses_str}_task-{task}_run-{run_str}_events.tsv'
    
    # Special case: sub-007 ses-03
    if sub_clean == '007' and ses == 3:
        events_file = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/func/sub-{sub_clean}_ses-{ses_str}_task-{task}_run-{run_str}_events.tsv'
    
    if not os.path.exists(events_file):
        print(f'    SKIP: {events_file} not found')
        return 0
    
    # Output directory
    timing_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/timing'
    os.makedirs(timing_dir, exist_ok=True)
    
    # Read events
    events = pd.read_csv(events_file, sep='\t')
    
    # Create timing file for each condition
    count = 0
    for cond in conditions:
        cond_events = events[events['block_type'] == cond]
        
        if len(cond_events) == 0:
            continue
        
        out_file = f'{timing_dir}/catloc_{sub_clean}_run-{run_str}_{cond}.txt'
        
        with open(out_file, 'w') as f:
            for _, row in cond_events.iterrows():
                f.write(f"{row['onset']:.3f} {row['duration']:.3f} 1\n")
        
        count += 1
    
    print(f'    Run {run}: {count}/{len(conditions)} conditions')
    return count


def main():
    print('Converting timing files...')
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
                convert_run(sub, ses, run)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
