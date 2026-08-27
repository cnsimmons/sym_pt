#!/usr/bin/env python
"""
merge_marlene.py — one table from grid.csv (permutation OLS) + lmm.csv (mixed model).

Row unit stays interpretable: `test` says which model produced the line.
  perm_ols   45 rows   5 specs per measure x comparison, 1 df each, beta + perm p
  lmm         9 rows   1 per measure x comparison, category x group omnibus, chi2 + df

Blocks are ordered measure -> comparison -> (5 perm_ols rows, then the lmm row),
so each LMM row sits directly under the five specs it corresponds to.

Usage:
  python merge_marlene.py                      # writes marlene_table.csv
  python merge_marlene.py --out combined.csv
"""
import argparse
from pathlib import Path

import pandas as pd

COLS = ['test', 'measure', 'comparison', 'comparison_name', 'spec',
        'estimate', 'stat', 'df', 'p', 'n_group_a', 'n_group_b',
        'age_beta', 'age_p', 'surg_beta', 'surg_p', 'converged',
        'age_cap', 'roi_set']

MEASURE_ORDER = {'peak_z': 0, 'distinctiveness': 1, 'geometry': 2}
SPEC_ORDER = {'binA': 0, 'binB': 1, 'binC': 2, 'cont': 3}


def spec_key(s):
    """binA/binB/binC/cont z>1.96/cont z>2.33, then the lmm row last."""
    if str(s).startswith('omnibus'):
        return (9, '')
    head = str(s).split()[0]
    return (SPEC_ORDER.get(head, 8), str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='grid.csv')
    ap.add_argument('--lmm', default='lmm.csv')
    ap.add_argument('--out', default='marlene_table.csv')
    args = ap.parse_args()

    for f in (args.grid, args.lmm):
        if not Path(f).exists():
            raise SystemExit(f'missing: {f}')

    g = pd.read_csv(args.grid)
    g = g.assign(test='perm_ols',
                 estimate=g['beta'], stat=pd.NA, df=pd.NA,
                 age_beta=pd.NA, age_p=pd.NA,
                 surg_beta=pd.NA, surg_p=pd.NA, converged=pd.NA)

    l = pd.read_csv(args.lmm)
    l = l[l['model'] == 'primary'].copy()          # drop any --surg-look rows
    l = l.assign(test='lmm',
                 spec='omnibus  category x group',
                 estimate=pd.NA, stat=l['chi2'],
                 n_group_a=l['n_a'], n_group_b=l['n_b'],
                 roi_set=pd.NA)

    out = pd.concat([g.reindex(columns=COLS), l.reindex(columns=COLS)],
                    ignore_index=True)

    out['_m'] = out['measure'].map(MEASURE_ORDER).fillna(9)
    out['_s'] = out['spec'].map(spec_key)
    out = (out.sort_values(['_m', 'comparison', '_s'])
              .drop(columns=['_m', '_s'])
              .reset_index(drop=True))

    out.to_csv(args.out, index=False)

    n_perm = int((out['test'] == 'perm_ols').sum())
    n_lmm = int((out['test'] == 'lmm').sum())
    print(f'wrote {args.out}   {len(out)} rows  '
          f'({n_perm} perm_ols + {n_lmm} lmm)')

    chk = out.groupby(['measure', 'comparison'])['test'].value_counts().unstack(fill_value=0)
    print('\nrows per measure x comparison:')
    print(chk.to_string())
    bad = chk[(chk.get('perm_ols', 0) != 5) | (chk.get('lmm', 0) != 1)]
    if len(bad):
        print('\n** unexpected block sizes — check the inputs:')
        print(bad.to_string())

    if out['converged'].notna().any():
        nc = out[(out['test'] == 'lmm') & (out['converged'] == False)]
        if len(nc):
            print('\n** LMM rows that did not converge:')
            print(nc[['measure', 'comparison', 'stat', 'df', 'p']].to_string(index=False))


if __name__ == '__main__':
    main()
