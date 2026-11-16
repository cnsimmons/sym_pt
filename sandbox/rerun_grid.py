#!/usr/bin/env python3
"""
rerun_grid.py - V3 Tables 2 and 4 (does the group x category difference scale
with how lateralized each category is?) for each processing version.

Runs pipeline/4_analyses/04_marlene_grid.py unchanged, with its default
settings (age cap 23, roi-set primary_strict, 5000 permutations, seed 0), its
own cohort (grid_cohort.EXCLUDE: sub-017, sub-027, sub-084), and only its two
input files swapped per version; sub-010 added to the exclusions for A1.
Same four runs as rerun_er.py, one SLURM job each.
Table 2 = measure peak_z; Table 4 = measure distinctiveness (the mean of the
three pairs containing each ROI's own category).

  python rerun_grid.py plan          inputs and which files V3's runs read
  python rerun_grid.py submit        4 jobs
  python rerun_grid.py run INDEX     one run (what the jobs call)
  python rerun_grid.py collect       finds the specs that reproduce V3 in the
                                     record run, then prints Tables 2 and 4 for
                                     every version
Writes only into sandbox/rerun_grid/.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
SB = REPO / 'sandbox'
OUT = SB / 'rerun_grid'
GRID = REPO / 'pipeline' / '4_analyses' / '04_marlene_grid.py'
HZ = SB / 'rerun_harmonize'
VERSIONS = {
    'P0_record':       (REPO / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv', REPO / 'D_liu' / 'rsa_v1_harmonized.csv'),
    'P1_nosmooth':     (HZ / 'new13' / 'univariate_v1_harmonized_sqrt.csv', HZ / 'new13' / 'rsa_v1_harmonized.csv'),
    'P2_nosmooth_c20': (HZ / 'new20' / 'univariate_v1_harmonized_sqrt.csv', HZ / 'new20' / 'rsa_v1_harmonized.csv'),
}
EXTRA = {'A0_record': set(), 'A1_no010': {'sub-010'}}
RUNS = [('P0_record', 'A0_record'), ('P0_record', 'A1_no010'),
        ('P1_nosmooth', 'A1_no010'), ('P2_nosmooth_c20', 'A1_no010')]
# V3 Tables 2 and 4: (measure, comparison, column) -> (beta, p)
V3 = {('peak_z', 6, '|LI|'): (-0.38, .578), ('peak_z', 6, 'binary'): (-0.36, .241),
      ('peak_z', 1, '|LI|'): (6.91, .012), ('peak_z', 1, 'binary'): (1.30, .037),
      ('peak_z', 2, '|LI|'): (4.95, .131), ('peak_z', 2, 'binary'): (0.98, .232),
      ('peak_z', 3, '|LI|'): (3.01, .463), ('peak_z', 3, 'binary'): (-0.08, .933),
      ('distinctiveness', 6, '|LI|'): (0.51, .003), ('distinctiveness', 6, 'binary'): (0.22, .0005),
      ('distinctiveness', 1, '|LI|'): (-0.50, .349), ('distinctiveness', 1, 'binary'): (-0.18, .108),
      ('distinctiveness', 2, '|LI|'): (1.02, .066), ('distinctiveness', 2, 'binary'): (0.18, .138),
      ('distinctiveness', 3, '|LI|'): (-0.10, .882), ('distinctiveness', 3, 'binary'): (-0.06, .671)}
ROWS = {6: 'LH ctrl vs RH ctrl', 1: 'LH-intact pt vs LH ctrl', 2: 'RH-intact pt vs RH ctrl',
        3: 'LH-intact pt vs RH-intact pt'}


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest() if p.exists() else None


def plan():
    ok = GRID.exists() and (GRID.parent / 'grid_cohort.py').exists()
    print(f'grid script: {GRID} {"found" if ok else "MISSING (or grid_cohort.py missing)"}')
    miss = [str(p) for v in VERSIONS.values() for p in v if not p.exists()]
    print('inputs:', 'all present' if not miss else 'MISSING ' + ', '.join(miss))
    pipe = REPO / 'pipeline' / 'outputs' / 'D_liu'
    for f in ['univariate_v1_harmonized_sqrt.csv', 'rsa_v1_harmonized.csv']:
        a, b = _md5(REPO / 'D_liu' / f), _md5(pipe / f)
        print(f'  record {f}: pipeline/outputs copy (the grid default) '
              + ('identical' if a == b and a else 'DIFFERS' if b else 'not found'))
    print(f'{len(RUNS)} runs: ' + ', '.join(f'{v} x {c}' for v, c in RUNS))
    return ok and not miss


def submit():
    if not plan():
        sys.exit('STOP: something is missing')
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    sh = OUT / 'grid_array.sh'
    sh.write_text(f"""#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --array=0-{len(RUNS) - 1}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output={OUT}/logs/grid_%a.out
#SBATCH --error={OUT}/logs/grid_%a.err
source /home/csimmon2/anaconda3/etc/profile.d/conda.sh && conda activate fmri
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python {Path(__file__).resolve()} run $SLURM_ARRAY_TASK_ID
""")
    r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


def run(i):
    v, c = RUNS[i]
    sys.path.insert(0, str(GRID.parent))
    spec = importlib.util.spec_from_file_location('grid_mod', GRID)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.UNI, mod.RSA = VERSIONS[v]
    mod.EXCLUDE = set(mod.EXCLUDE) | EXTRA[c]
    out = OUT / f'grid_{v}__{c}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(GRID), '--csv', str(out)]
    if os.environ.get('GRID_NPERM'):                  # plumbing tests only
        argv += ['--n-perm', os.environ['GRID_NPERM']]
    print(f'{v} x {c}: exclusions {sorted(mod.EXCLUDE)}')
    sys.argv = argv
    mod.main()
    print(f'wrote {out}')


def _spec_match(rec):
    """In the record run, the |LI| and binary specs whose values reproduce V3."""
    best = {}
    for col in ['|LI|', 'binary']:
        cands = [s for s in rec['spec'].unique() if (('LI|' in s) if col == '|LI|' else s.startswith('bin'))]
        score = {}
        for s in cands:
            hits = 0
            for (m, comp, cc), (b, p) in V3.items():
                if cc != col:
                    continue
                r = rec[(rec.measure == m) & (rec.comparison == comp) & (rec.spec == s)]
                if len(r) and abs(round(r.beta.iloc[0], 2) - b) < .006 and (
                        abs(round(r.p.iloc[0], 3) - p) < .0006 or (p < .001 and r.p.iloc[0] < .001)):
                    hits += 1
            score[s] = hits
        best[col] = max(score, key=score.get) if score else None
        print(f'  {col:7s} spec reproducing V3: {best[col]!r}  ({score.get(best[col], 0)} of 8 cells)'
              + ('   others: ' + ', '.join(f'{k!r} {n}' for k, n in score.items() if k != best[col]) if len(score) > 1 else ''))
    return best


def collect():
    got = {(v, c): OUT / f'grid_{v}__{c}.csv' for v, c in RUNS}
    have = {k: pd.read_csv(p) for k, p in got.items() if p.exists()}
    print(f'{len(have)} of {len(RUNS)} runs present')
    if ('P0_record', 'A0_record') not in have:
        print('the record run is needed first, to identify the specs')
        return
    print('record run against V3:')
    best = _spec_match(have[('P0_record', 'A0_record')])
    rows = []
    for (v, c), g in have.items():
        for (m, comp, col) in V3:
            r = g[(g.measure == m) & (g.comparison == comp) & (g.spec == best[col])]
            if len(r):
                rows.append({'table': 2 if m == 'peak_z' else 4, 'comparison': ROWS[comp], 'column': col,
                             'version': f'{v} | {c}', 'beta': r.beta.iloc[0], 'p': r.p.iloc[0]})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'tables_2_4_by_version.csv', index=False)
    T['cell'] = T.apply(lambda r: f"b = {r.beta:+.2f}, p = {r.p:.3f}", axis=1)
    for tab in (2, 4):
        print(f'\nTable {tab}  ({"peak selectivity" if tab == 2 else "between-category similarity, own-category pairs"})')
        x = T[T.table == tab].pivot_table(index=['comparison', 'column'], columns='version', values='cell', aggfunc='first')
        x = x.reindex(index=[(ROWS[k], col) for k in (6, 1, 2, 3) for col in ('|LI|', 'binary')])
        with pd.option_context('display.width', 250, 'display.max_colwidth', 30):
            print(x.to_string())
    print(f'\nsaved: {OUT / "tables_2_4_by_version.csv"}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'plan'
    {'plan': plan, 'submit': submit, 'collect': collect}.get(cmd, lambda: run(int(sys.argv[2])))()
