#!/usr/bin/env python3
"""
otc_rsm_loo.py — which pair carries the consistency gap, and does any single
patient drive it.

Reads otc_rsm_persubject.csv, written by:
    python otc_rsm_rosenke.py --csv otc_rsm.csv

Two questions:
  1  drop each category pair in turn and recompute the ctrl-vs-pt consistency
     difference. A difference that shrinks when a pair is dropped means that
     pair carries it.
  2  drop each patient in turn. If the gap closes, the finding is that patient.

Usage:
  python otc_rsm_loo.py [otc_rsm_persubject.csv]
"""
import sys
import numpy as np
import pandas as pd

F = sys.argv[1] if len(sys.argv) > 1 else 'otc_rsm_persubject.csv'
RNG = np.random.default_rng(42)
NPERM = 10000
INTACT = {'l': 'left', 'r': 'right'}

df = pd.read_csv(F)
PAIRS = [c for c in df.columns if '-' in c]
print(f'{F}: {len(df)} rows | pairs: {PAIRS}\n')


def within(mat):
    n = len(mat)
    out = np.full(n, np.nan)
    for i in range(n):
        o = [j for j in range(n) if j != i]
        out[i] = np.mean([np.corrcoef(mat[i], mat[j])[0, 1] for j in o])
    return out


def perm(a, b, n=NPERM):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b]); na = len(a); k = 0
    for _ in range(n):
        p = RNG.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


for hemi in ('l', 'r'):
    d = df[df.hemi == hemi]
    C = d[d.group == 'control'][PAIRS].to_numpy(float)
    ptd = d[(d.group == 'OTC') & (d.intact_hemi == INTACT[hemi])]
    P = ptd[PAIRS].to_numpy(float)
    pids = ptd['subject_id'].astype(str).to_numpy()
    if len(P) < 4:
        continue

    print('=' * 72)
    print(f'[{hemi.upper()}H]  {len(C)} controls  vs  {len(P)} '
          f'{"LH" if hemi == "l" else "RH"}-intact patients')
    print('=' * 72)

    base_d, base_p = perm(within(C), within(P))
    print(f'\nbaseline: ctrl {np.nanmean(within(C)):+.3f}  '
          f'pt {np.nanmean(within(P)):+.3f}  diff {base_d:+.3f}  p={base_p:.4f}')

    print('\n1. DROP ONE CATEGORY PAIR (consistency on the remaining 5)')
    for k, c in enumerate(PAIRS):
        keep = [i for i in range(len(PAIRS)) if i != k]
        wc, wp = within(C[:, keep]), within(P[:, keep])
        dd, pp = perm(wc, wp)
        shrink = abs(dd) / abs(base_d) if base_d else np.nan
        print(f'   drop {c:14s} ctrl {np.nanmean(wc):+.3f}  pt {np.nanmean(wp):+.3f}'
              f'  diff {dd:+.3f} ({shrink:.0%} of baseline)  p={pp:.4f}'
              + ('  *' if pp < .05 else '   n.s.'))

    print('\n2. DROP ONE PATIENT')
    wc = within(C)
    rows = []
    for i, sid in enumerate(pids):
        keep = [j for j in range(len(P)) if j != i]
        dd, pp = perm(wc, within(P[keep]))
        rows.append((sid, dd, pp))
    for sid, dd, pp in sorted(rows, key=lambda r: r[2], reverse=True):
        print(f'   drop {sid:10s} diff {dd:+.3f}  p={pp:.4f}'
              + ('   <- GAP CLOSES' if pp >= .05 else ''))

    print('\n3. DROP THE LOW-CONSISTENCY TAIL (cumulative, worst first)')
    order = np.argsort(within(P))          # lowest consistency first
    for nd in range(1, len(P) - 3):
        keep = order[nd:]
        dd, pp = perm(wc, within(P[keep]))
        dropped = ', '.join(pids[order[:nd]])
        print(f'   n={len(keep):2d}  dropped [{dropped}]  diff {dd:+.3f}  p={pp:.4f}'
              + ('   <- GAP CLOSES' if pp >= .05 else ''))
    print()
