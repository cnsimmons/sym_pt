#!/usr/bin/env python3
"""
01_organize.py - Setup directory structure for sym_pt
"""
import os
from sym_pt_params import processed_dir, task, skip_subs, get_sessions, get_runs, _load_csv


def setup_subject(sub, sessions):
    """Create directory structure for one subject"""
    for ses in sessions:
        ses_dir = f'{processed_dir}/sub-{sub}/ses-{ses:02d}'
        runs = get_runs(sub, ses)

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

    df = _load_csv()
    # Get unique subjects
    subs = df['sub_clean'].unique()

    for sub in subs:
        if sub in skip_subs:
            print(f'SKIP: {sub}')
            continue

        sessions = get_sessions(sub)
        print(f'\nsub-{sub} ({len(sessions)} sessions)')
        setup_subject(sub, sessions)

    print('\nDone!')


if __name__ == '__main__':
    main()