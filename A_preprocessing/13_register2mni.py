#!/usr/bin/env python3
"""
13_register_zstats_mni.py — Register HighLevel zstats/copes to MNI space.

Companion to register_zstats.py (which keeps things in ses-01 anat space).
This script applies anat2stand.mat (from register_mirror.py) to put HighLevel
zstats into MNI 2mm space, enabling fsaverage surface plotting.

Inputs (per subject/session):
  - {processed_dir}/sub-XXX/ses-YY/derivatives/fsl/loc/HighLevel.gfeat/
        cope{N}.feat/stats/zstat1.nii.gz       (and cope1.nii.gz)
  - {processed_dir}/sub-XXX/ses-{first_ses}/anat/anat2stand.mat

Outputs (alongside the originals):
  - cope{N}.feat/stats/zstat1_mni.nii.gz
  - cope{N}.feat/stats/cope1_mni.nii.gz

Notes:
  - HighLevel outputs are ALREADY in ses-01 anat space (by FEAT design).
    Apply ses-01 anat2stand.mat for ALL sessions of a subject.
  - Liu pipeline copes: 1 (Face>Object), 2 (House>Object), 3 (Object>Scramble),
    13 (Face>Word; negate for Word>Face), 19 (Scramble_raw, EVC).
  - Registers all copes 1-19 by default; downstream picks what it needs.
  - FLIRT trilinear interpolation (continuous data).
  - Honors skip_subs + EXTRA_SKIP and PRE_SURGERY_SESSIONS from
    liu_recreation_csv_v2.py.

Setup:
  module load fsl/6.0.3
  conda activate fmri

Usage:
  python 13_register_zstats_mni.py sub-005 01           # single sub/ses
  python 13_register_zstats_mni.py --sub 005,008        # all sessions of listed subs
  python 13_register_zstats_mni.py --all                # everyone
  python 13_register_zstats_mni.py --all --dry-run      # preview
"""
import os
import sys
import subprocess
import argparse
import time

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions, _load_csv

MNI_BRAIN = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
COPES = list(range(1, 20))

# Match liu_recreation_csv_v2.py exclusions
EXTRA_SKIP = {'sub-017', 'control083', 'control085'}
PRE_SURGERY_SESSIONS = {
    'sub-021': {'01'}, 'sub-045': {'01'}, 'sub-047': {'01'}, 'sub-049': {'01'},
    'sub-070': {'01'}, 'sub-073': {'01'}, 'sub-081': {'01'}, 'sub-086': {'01'},
    'sub-108': {'02'},
}


def register_one(sub, ses, dry_run=False):
    """Register all copes for a single sub/ses. Returns (n_done, n_skipped, n_failed)."""
    task = 'loc'
    sub_clean = sub.replace('sub-', '')

    sessions = get_sessions(sub_clean)
    if not sessions:
        print(f"  ERROR: no sessions found for {sub}")
        return 0, 0, 1
    first_ses = f'{sessions[0]:02d}'

    anat_dir = f'{processed_dir}/{sub}/ses-{first_ses}/anat'
    anat2stand = f'{anat_dir}/anat2stand.mat'
    if not os.path.exists(anat2stand):
        print(f"  ERROR: anat2stand.mat missing — run register_mirror.py first")
        print(f"    expected: {anat2stand}")
        return 0, 0, 1
    if not os.path.exists(MNI_BRAIN):
        print(f"  ERROR: MNI reference not found: {MNI_BRAIN}")
        return 0, 0, 1

    highlevel_dir = (f'{processed_dir}/{sub}/ses-{ses}/'
                     f'derivatives/fsl/{task}/HighLevel.gfeat')
    if not os.path.exists(highlevel_dir):
        print(f"  ERROR: HighLevel.gfeat not found: {highlevel_dir}")
        return 0, 0, 1

    n_done, n_skipped, n_failed = 0, 0, 0

    for cope_num in COPES:
        cope_stats_dir = f'{highlevel_dir}/cope{cope_num}.feat/stats'
        if not os.path.isdir(cope_stats_dir):
            continue

        for prefix in ['zstat', 'cope']:
            src = f'{cope_stats_dir}/{prefix}1.nii.gz'
            dst = f'{cope_stats_dir}/{prefix}1_mni.nii.gz'

            if not os.path.exists(src):
                continue
            if os.path.exists(dst):
                n_skipped += 1
                continue

            cmd = (f'flirt -in {src} -ref {MNI_BRAIN} -out {dst} '
                   f'-applyxfm -init {anat2stand} -interp trilinear')
            if dry_run:
                print(f"    DRY: cope{cope_num} {prefix}")
                continue
            try:
                subprocess.run(cmd.split(), check=True,
                               capture_output=True, text=True)
                n_done += 1
            except subprocess.CalledProcessError as e:
                print(f"    cope{cope_num} {prefix} ERROR: {e.stderr.strip()}")
                n_failed += 1

    return n_done, n_skipped, n_failed


def expand_targets(sub_filter=None):
    """Yield (sub, ses_str) tuples honoring skip lists."""
    df = _load_csv()
    for sc in sorted(df['sub_clean'].unique()):
        sid = f'sub-{sc}'
        if sub_filter is not None and sc not in sub_filter:
            continue
        if sc in skip_subs or sid in EXTRA_SKIP:
            print(f'  SKIP {sid}: in skip list')
            continue

        row = df[df['sub_clean'] == sc].iloc[0]
        code = f"{row.get('group', '')}{sc}"
        if code in EXTRA_SKIP:
            print(f'  SKIP {sid}: {code} in EXTRA_SKIP')
            continue

        sessions = get_sessions(sc)
        if not sessions:
            print(f'  SKIP {sid}: no sessions')
            continue

        for ses in sessions:
            ses_str = f'{ses:02d}'
            if ses_str in PRE_SURGERY_SESSIONS.get(sid, set()):
                print(f'  SKIP {sid} ses-{ses_str}: pre-surgery')
                continue
            yield sid, ses_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('positional', nargs='*',
                        help='sub-XXX YY for single subject/session mode')
    parser.add_argument('--sub', type=str,
                        help='Comma-separated subject list (e.g., 005,008,021)')
    parser.add_argument('--all', action='store_true',
                        help='Run for all subjects (honoring skip lists)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run without executing')
    args = parser.parse_args()

    # Mode 1: single sub/ses positional args
    if len(args.positional) == 2:
        sub, ses = args.positional
        print(f'Processing {sub} ses-{ses}')
        d, s, f = register_one(sub, ses, dry_run=args.dry_run)
        print(f'\nDone: {d} registered, {s} already existed, {f} failed')
        return

    # Mode 2: --sub or --all
    if not (args.sub or args.all):
        parser.print_help()
        sys.exit(1)

    sub_filter = None
    if args.sub:
        sub_filter = {s.replace('sub-', '').strip() for s in args.sub.split(',')}

    targets = list(expand_targets(sub_filter))
    print(f'\n{len(targets)} sub/ses pairs in queue')
    print(f'Mode: {"DRY RUN" if args.dry_run else "EXECUTE"}\n')

    t0 = time.time()
    tot_done, tot_skipped, tot_failed = 0, 0, 0
    for sid, ses_str in targets:
        elapsed = time.time() - t0
        print(f'[{elapsed:5.0f}s] {sid} ses-{ses_str}')
        d, s, f = register_one(sid, ses_str, dry_run=args.dry_run)
        tot_done += d; tot_skipped += s; tot_failed += f

    elapsed = time.time() - t0
    print(f'\nFinished in {elapsed/60:.1f} min: '
          f'{tot_done} registered, {tot_skipped} already existed, '
          f'{tot_failed} failed')


if __name__ == '__main__':
    main()