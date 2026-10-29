#!/usr/bin/env python3
"""
02_convert_timing.py - Convert BIDS events to FSL 3-column timing files
"""
import os
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
import pandas as pd
from sym_pt_params import raw_dir, processed_dir, task, skip_subs, conditions, get_sessions, get_runs, _load_csv


def convert_run(sub, ses, run):
    """Convert events.tsv to FSL timing files for one run"""
    ses_str = f'{ses:02d}'
    run_str = f'{run:02d}'

    events_file = f'{raw_dir}/sub-{sub}/ses-{ses_str}/func/sub-{sub}_ses-{ses_str}_task-{task}_run-{run_str}_events.tsv'

    if not os.path.exists(events_file):
        print(f'    SKIP: {events_file} not found')
        return 0

    timing_dir = f'{processed_dir}/sub-{sub}/ses-{ses_str}/timing'
    os.makedirs(timing_dir, exist_ok=True)

    events = pd.read_csv(events_file, sep='\t')

    # Handle different column names (most use block_type, some use trial_type)
    if 'block_type' in events.columns:
        type_col = 'block_type'
    elif 'trial_type' in events.columns:
        type_col = 'trial_type'
    else:
        print(f'    SKIP: no block_type or trial_type column found')
        return 0

    count = 0
    for cond in conditions:
        cond_events = events[events[type_col] == cond]

        if len(cond_events) == 0:
            continue

        out_file = f'{timing_dir}/catloc_{sub}_run-{run_str}_{cond}.txt'

        with open(out_file, 'w') as f:
            for _, row in cond_events.iterrows():
                f.write(f"{row['onset']:.3f} {row['duration']:.3f} 1\n")

        count += 1

    print(f'    Run {run}: {count}/{len(conditions)} conditions')
    return count


def main():
    print('Converting timing files...')

    df = _load_csv()
    subs = df['sub_clean'].unique()

    for sub in subs:
        if sub in skip_subs:
            print(f'SKIP: {sub}')
            continue

        sessions = get_sessions(sub)
        print(f'\nsub-{sub}')

        for ses in sessions:
            runs = get_runs(sub, ses)
            print(f'  Session {ses}:')

            for run in runs:
                convert_run(sub, ses, run)

    print('\nDone!')


if __name__ == '__main__':
    main()