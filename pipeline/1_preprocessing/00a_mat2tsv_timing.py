#!/usr/bin/env python3
"""
00a_mat2tsv_timing.py - Convert localizer .mat files to BIDS events files

Changed 2026-11-01 (see deviation log):
  - --sub and --ses are REQUIRED (the old version silently defaulted to
    sub-108 ses-02 when they were omitted)
  - run number is read from any name ending in <N>loc.mat (ebloc1loc.mat,
    EC1loc.mat, FOloc1loc.mat, ...), or given with --run
  - no silent fallback filename: if the run cannot be determined, it stops
Conversion itself is unchanged: onsets = block_starts - block_starts[0] + 8,
duration 16 s, codes 0 Face, 1 Word, 2 Scramble, 3 House, 4 Object.

Usage:
  python 00a_mat2tsv_timing.py <file.mat> --sub 092 --ses 02 [--run 1] [--out path.tsv]
Default output: <session>/func/sub-XXX_ses-NN_task-loc_run-RR_events.tsv
(the func/ folder next to the .mat file's covs/ folder)
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import scipy.io

CONDITION_MAP = {0: 'Face', 1: 'Word', 2: 'Scramble', 3: 'House', 4: 'Object'}


def run_from_filename(name):
    """'ebloc1loc.mat' / 'EC1loc.mat' / 'FOloc1loc.mat' -> 1; None if no match."""
    m = re.search(r'(\d+)loc\.mat$', name)
    return int(m.group(1)) if m else None


def convert(mat_file, sub, ses, run=None, out=None):
    mat_path = Path(mat_file)
    if run is None:
        run = run_from_filename(mat_path.name)
    if run is None:
        sys.exit(f'ERROR: cannot read a run number from {mat_path.name}; pass --run')

    print(f'Loading {mat_path} ...')
    mat = scipy.io.loadmat(mat_path)
    block_starts = mat['block_starts'][0]
    stim_ord = mat['stimOrd'].flatten()
    if len(block_starts) != len(stim_ord):
        sys.exit(f'ERROR: {len(block_starts)} block starts but {len(stim_ord)} codes')

    onsets = block_starts - block_starts[0] + 8      # 8 s initial fixation
    df = pd.DataFrame({'onset': onsets,
                       'duration': [16.0] * len(onsets),
                       'block_type': [CONDITION_MAP[int(c)] for c in stim_ord]})

    if out is None:
        fname = f'sub-{sub}_ses-{ses}_task-loc_run-{run:02d}_events.tsv'
        func_dir = mat_path.parent.parent / 'func'
        if not func_dir.exists():
            sys.exit(f'ERROR: {func_dir} does not exist; pass --out')
        out = func_dir / fname
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False, float_format='%.3f')

    print(f'Saved {out}')
    print(f'  sub-{sub} ses-{ses} run-{run:02d}: {len(df)} blocks, '
          f'last block ends {onsets[-1] + 16:.1f} s')
    print(f'  order: {" ".join(df.block_type)}')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mat_file')
    ap.add_argument('--sub', required=True, help='subject number, e.g. 092')
    ap.add_argument('--ses', required=True, help='session number, e.g. 02')
    ap.add_argument('--run', type=int, default=None, help='run number (else read from filename)')
    ap.add_argument('--out', default=None, help='output .tsv path (else BIDS name in func/)')
    a = ap.parse_args()

    if not Path(a.mat_file).exists():
        sys.exit(f'ERROR: {a.mat_file} does not exist')
    sub = a.sub.removeprefix('sub-')
    ses = a.ses.removeprefix('ses-').zfill(2)
    convert(a.mat_file, sub, ses, a.run, a.out)


if __name__ == '__main__':
    main()
