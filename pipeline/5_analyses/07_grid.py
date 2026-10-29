#!/usr/bin/env python3
"""
07_grid.py — runs the three grid scripts and merges their output.

WHY A DRIVER RATHER THAN A REWRITE
  marlene_grid.py, marlene_lmm.py and marlene_roi.py are ~1270 lines of
  validated code that produce the numbers in marlene_table_final.pdf. Folding
  their loop logic into one file would mean reproducing it, which risks
  changing results while refactoring. This runs them unmodified and
  concatenates what they write.

WHAT EACH CONTRIBUTES
  marlene_grid.py   the |LI| codings — continuous, rank, symmetric/lateralised
                    — as category x group interactions. One row per
                    measure x comparison x coding.
  marlene_lmm.py    the omnibus, as a joint Wald chi2 from a mixed model. One
                    row per measure x comparison, plus the pooled-over-ROI
                    version for between-category similarity.
  marlene_roi.py    per-ROI omnibus and per-pair follow-ups with FDR. One row
                    per measure x comparison x ROI, and per pair.

THE COHORT
  EXCLUDE below is the single definition for all three. They import it from
  here, so there is one list rather than three. It must match EXCLUDE in
  05_stats_harmony.py or the grid and the stats file describe different
  cohorts — which they did until Sept 2026, when the grid reported 36 controls
  and the manuscript 38.

  The age cap removes sub-091, sub-095 and sub-096, so they are not named
  here. sub-017 is excluded for polymicrogyria; sub-027 and sub-084 are
  control exclusions.

REQUIRED EDIT TO THE THREE SCRIPTS
  Each needs two lines. At the top:

      from importlib import import_module
      EXCLUDE = import_module('07_grid').EXCLUDE

  or, if the driver sits alongside them, simply:

      from grid_driver import EXCLUDE

  and inside load_measure, immediately after the if/elif chain that sets `d`
  and before the line beginning `ctl = `:

      d = d[~d['subject_id'].isin(EXCLUDE)]

  Without that filter the three scripts include sub-027 and sub-084, because
  those were never in params.py's skip lists and are therefore present in the
  harmonized CSVs.

OUTPUT
  One long-format CSV. The three scripts have different column sets, so the
  merge keeps a common core and leaves the rest as NaN where a stage does not
  produce them:

      stage measure comparison comparison_name role paired
      spec roi pair beta chi2 df p q_fdr cohen_d diff n_a n_b

  This is written separately rather than appended to
  stats_results_harmonized_corrected.csv, because that file is one row per
  per-ROI scalar comparison and these rows are a different shape. Merging them
  would make the measure column mean two things.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')

# ── the cohort ──────────────────────────────────────────────────────────────
# Defined in grid_cohort.py so the three grid scripts can import the same set.
# A driver cannot filter for them: each loads the harmonized CSVs itself, so
# the exclusion must be applied before their statistics are computed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_cohort import EXCLUDE, AGE_CAP   # noqa: E402

# ── where the three scripts live ────────────────────────────────────────────
# (path, accepts --n-perm). marlene_lmm.py is a mixed model and has no
# permutation argument; passing one to it is an error.
SCRIPTS = {
    'grid': (GIT / 'marlene_grid.py', True),
    'lmm':  (GIT / 'marlene_lmm.py', False),
    'roi':  (GIT / 'one_last_8.27' / 'marlene_roi.py', True),
}

OUT = GIT / 'D_liu' / 'grid_results.csv'

CORE = ['stage', 'measure', 'comparison', 'comparison_name', 'role', 'paired',
        'spec', 'roi', 'pair', 'model', 'factor',
        'beta', 'chi2', 'df', 'p', 'q_fdr', 'cohen_d', 'diff',
        'n_a', 'n_b', 'converged', 'age_cap']

# marlene_grid.py names its n columns differently
RENAME = {'n_group_a': 'n_a', 'n_group_b': 'n_b'}


def run(stage, script, tmp, n_perm, extra=()):
    if not script.exists():
        print(f'  {stage}: MISSING {script} — skipped')
        return None
    cmd = [sys.executable, str(script), '--csv', str(tmp)]
    if n_perm:
        cmd += ['--n-perm', str(n_perm)]
    cmd += list(extra)
    print(f'  {stage}: {" ".join(cmd[1:])}')
    r = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f'{stage} failed (exit {r.returncode})')
    if not tmp.exists():
        print(f'  {stage}: no output written — skipped')
        return None
    d = pd.read_csv(tmp).rename(columns=RENAME)
    d.insert(0, 'stage', stage)
    print(f'  {stage}: {len(d)} rows')
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--clean', action='store_true',
                    help='delete the three intermediate part CSVs after merging')
    ap.add_argument('--force', action='store_true',
                    help='rerun every stage even if its part file exists')
    args = ap.parse_args()

    print(f'cohort: EXCLUDE={sorted(EXCLUDE)}  age cap <= {AGE_CAP:g}')
    print('running the three grid scripts')

    parts, tmps = [], []
    for stage, (script, takes_nperm) in SCRIPTS.items():
        tmp = GIT / 'D_liu' / f'_grid_part_{stage}.csv'
        tmps.append(tmp)
        if tmp.exists() and not args.force:
            # resume: a previous run got this far. The grid stage is 10,000
            # permutations per cell, so redoing it after a downstream failure
            # is expensive and pointless.
            d = pd.read_csv(tmp).rename(columns=RENAME)
            d.insert(0, 'stage', stage)
            print(f'  {stage}: {len(d)} rows (reused {tmp.name}; --force to rerun)')
        else:
            d = run(stage, script, tmp, args.n_perm if takes_nperm else None)
        if d is not None:
            parts.append(d)

    if not parts:
        raise SystemExit('nothing produced')

    df = pd.concat(parts, ignore_index=True)
    for c in CORE:
        if c not in df.columns:
            df[c] = pd.NA
    rest = [c for c in df.columns if c not in CORE]
    df = df[CORE + rest]

    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f'\nwrote {out}  ({len(df)} rows)')
    print(df.groupby(['stage', 'measure']).size().to_string())

    # Part files are kept by default. The grid stage is 10,000 permutations
    # per cell, so a downstream failure must not discard it.
    if args.clean:
        for t in tmps:
            if t.exists():
                t.unlink()
    else:
        print('parts kept:', ', '.join(t.name for t in tmps if t.exists()))


if __name__ == '__main__':
    main()