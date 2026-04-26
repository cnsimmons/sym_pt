#!/usr/bin/env python3
"""
00_prt2tsv_timing.py - Convert BrainVoyager .prt files to BIDS-compatible TSV events files

.prt format (ResolutionOfTime: Volumes):
  Onsets/offsets are 1-indexed TR units, inclusive on both ends.
  e.g. "17  24" with TR=2 -> onset=32.0s, duration=16.0s (8 volumes).

Output matches 00_mat2tsv_timing.py convention:
  Columns: onset, duration, block_type
  BIDS naming: sub-{ID}_ses-{SES}_task-loc_run-{RUN}_events.tsv

Usage:
  python 00_prt2tsv_timing.py <prt_file> --tr <TR> --sub <ID> --ses <SES> --run <RUN> [--out <path>]

Example:
  python 00_prt2tsv_timing.py Localizer_run1_-_vol.prt --tr 2.0 --sub 099 --ses 01 --run 1
"""
import sys
import re
import argparse
import pandas as pd
from pathlib import Path

# Conditions modeled as EVs (Fixation = baseline, skipped)
SKIP_CONDITIONS = {'Fixation'}

# Expected output conditions (matches sym_pt_params.conditions)
EXPECTED_CONDITIONS = {'Face', 'House', 'Object', 'Word', 'Scramble'}


def parse_prt(prt_path, tr):
    """Parse a BrainVoyager .prt file (Volumes resolution) into event rows.

    Returns list of (onset_sec, duration_sec, condition_name) tuples.
    """
    text = Path(prt_path).read_text()

    # Verify volumes resolution
    res_match = re.search(r'ResolutionOfTime:\s*(\w+)', text)
    if not res_match or res_match.group(1).lower() != 'volumes':
        raise ValueError(
            f'{prt_path}: ResolutionOfTime is not "Volumes" '
            f'(got {res_match.group(1) if res_match else "missing"}). '
            'Script assumes volume-indexed onsets.'
        )

    # Find body after NrOfConditions line
    m = re.search(r'NrOfConditions:\s*(\d+)', text)
    if not m:
        raise ValueError(f'{prt_path}: no NrOfConditions header found')
    body = text[m.end():]

    rows = []
    # Split into blocks on blank lines
    blocks = re.split(r'\n\s*\n', body.strip())

    for blk in blocks:
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if not lines or lines[0].startswith('Color:'):
            continue

        cond = lines[0]
        if cond in SKIP_CONDITIONS:
            continue

        # Second line should be number of blocks
        try:
            n_blocks = int(lines[1])
        except (ValueError, IndexError):
            continue

        # Next n_blocks lines are "start_TR  end_TR"
        for i in range(n_blocks):
            try:
                parts = lines[2 + i].split()
                start_tr = int(parts[0])
                end_tr = int(parts[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f'{prt_path}: malformed block for condition "{cond}" '
                    f'(expected {n_blocks} blocks, line {i} bad)'
                )

            onset = (start_tr - 1) * tr
            duration = (end_tr - start_tr + 1) * tr
            rows.append((onset, duration, cond))

    rows.sort(key=lambda r: r[0])
    return rows


def write_tsv(rows, out_path):
    """Write events to TSV with columns matching 00_mat2tsv_timing.py."""
    df = pd.DataFrame(rows, columns=['onset', 'duration', 'block_type'])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False, float_format='%.3f')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prt', help='Path to input .prt file')
    ap.add_argument('--tr', type=float, required=True,
                    help='Repetition time in seconds (verify against BOLD JSON)')
    ap.add_argument('--sub', required=True, help='Subject ID without sub- prefix (e.g. 099)')
    ap.add_argument('--ses', required=True, help='Session ID without ses- prefix (e.g. 01)')
    ap.add_argument('--run', type=int, required=True, help='Run number (1-indexed)')
    ap.add_argument('--out', default=None,
                    help='Output TSV path. Default: '
                         '/lab_data/behrmannlab/hemi/Raw/sub-{ID}/ses-{SES}/func/'
                         'sub-{ID}_ses-{SES}_task-loc_run-{RUN}_events.tsv '
                         '(lab-shared raw dir, used by 02_convert_timing.py)')
    args = ap.parse_args()

    if not Path(args.prt).exists():
        sys.exit(f'ERROR: {args.prt} does not exist')

    sub = args.sub.lstrip('sub-')
    ses = args.ses.lstrip('ses-').zfill(2)
    run_str = f'{args.run:02d}'

    if args.out is None:
        out_path = (
            f'/lab_data/behrmannlab/hemi/Raw/sub-{sub}/ses-{ses}/func/'
            f'sub-{sub}_ses-{ses}_task-loc_run-{run_str}_events.tsv'
        )
    else:
        out_path = args.out

    print(f'Parsing {args.prt} (TR={args.tr}s)...')
    rows = parse_prt(args.prt, args.tr)

    df = write_tsv(rows, out_path)
    print(f'Wrote {len(df)} events -> {out_path}')

    # Sanity report
    found_conds = set(df['block_type'].unique())
    missing = EXPECTED_CONDITIONS - found_conds
    extra = found_conds - EXPECTED_CONDITIONS
    if missing:
        print(f'  WARNING: expected conditions missing from .prt: {sorted(missing)}')
    if extra:
        print(f'  NOTE: unexpected conditions in .prt: {sorted(extra)}')

    print(f'\nPer-condition counts:')
    print(df.groupby('block_type').size().to_string())
    print(f'\nFirst rows:')
    print(df.head().to_string(index=False))


if __name__ == '__main__':
    main()