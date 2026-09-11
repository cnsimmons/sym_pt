#!/usr/bin/env python3
"""
otc_rsm_clr.py — closure check on the whole-OTC odRSM.

The four cat-vs-all-others z-maps are near-closed: their per-voxel sum has
sd 1.85 against a mean per-map sd of 2.05 (ratio 0.905), and the six pairwise
correlations average -0.255 against the -1/3 expected under exact closure.
Under that constraint the six values carry far fewer than six degrees of
freedom, so a rise in one pair mechanically forces the others down. The RH
result shows exactly that signature: all three word pairs up, all three
non-word pairs down.

This script re-tests the same numbers three ways:

  RAW   the six Fisher-z values as-is                       (current analysis)
  CLR   centered log-ratio on the six values, after shifting to positive and
        renormalising to a constant sum. Removes the constant-sum constraint,
        the same fix used for the WTA composition.
  RANK  within-subject rank of the six pairs (1-6). Scale-free and
        constraint-free; tests whether the ORDERING of the six differs by
        group, which is what the closure argument cannot explain away.

Consistency (section B) is reported for all three. It is the one measure that
was never threatened by closure, since both groups sit under the same
constraint, so agreement across the three is a sanity check rather than a test.

Reads otc_rsm_persubject.csv from `otc_rsm_rosenke.py --csv`.

Usage:
  python otc_rsm_clr.py [otc_rsm_persubject.csv]
"""
import sys
import numpy as np
import pandas as pd

F = sys.argv[1] if len(sys.argv) > 1 else 'otc_rsm_persubject.csv'
RNG = np.random.default_rng(42)
NPERM = 10000
INTACT = {'l': 'left', 'r': 'right'}
CATS = ['face', 'house', 'object', 'word']

df = pd.read_csv(F)
PAIRS = [c for c in df.columns if '-' in c]
assert len(PAIRS) == 6, PAIRS
print(f'{F}: {len(df)} rows\npairs: {PAIRS}\n')


def clr(M):
    """Centered log-ratio. Shift the whole matrix to strictly positive, make
    each row sum to 1, then log and center within row."""
    M = np.asarray(M, float)
    X = M - M.min() + 1e-3
    X = X / X.sum(axis=1, keepdims=True)
    L = np.log(X)
    return L - L.mean(axis=1, keepdims=True)


def ranks(M):
    M = np.asarray(M, float)
    return np.apply_along_axis(lambda r: r.argsort().argsort() + 1.0, 1, M)


def perm(a, b, n=NPERM):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b]); na = len(a); k = 0
    for _ in range(n):
        p = RNG.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


def bh(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    q = np.minimum.accumulate(
        (ps[o] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.clip(q, 0, 1)
    return out


def within(mat):
    n = len(mat)
    out = np.full(n, np.nan)
    for i in range(n):
        o = [j for j in range(n) if j != i]
        out[i] = np.mean([np.corrcoef(mat[i], mat[j])[0, 1] for j in o])
    return out


def per_cat(M):
    """Mean of the 3 pairs containing each category."""
    out = np.full((len(M), 4), np.nan)
    for k, c in enumerate(CATS):
        cols = [i for i, p in enumerate(PAIRS) if c in p.split('-')]
        assert len(cols) == 3
        out[:, k] = np.nanmean(M[:, cols], axis=1)
    return out


for hemi in ('l', 'r'):
    d = df[df.hemi == hemi]
    C0 = d[d.group == 'control'][PAIRS].to_numpy(float)
    P0 = d[(d.group == 'OTC') & (d.intact_hemi == INTACT[hemi])][PAIRS].to_numpy(float)
    if len(P0) < 4:
        continue
    lab = 'LH' if hemi == 'l' else 'RH'
    print('=' * 76)
    print(f'[{lab}]  {len(C0)} controls  vs  {len(P0)} {lab}-intact patients')
    print('=' * 76)

    # closure diagnostic on the six values themselves
    for nm, M in (('controls', C0), ('patients', P0)):
        rs = M.sum(axis=1)
        print(f'  {nm}: row-sum of the six pairs  mean {rs.mean():+.3f}  '
              f'sd {rs.std(ddof=1):.3f}')
    print('  (a small sd here = the six values are near-constant-sum '
          'within subject = closure)')

    both = np.vstack([C0, P0])
    for tag, T in (('RAW', lambda X: X), ('CLR', clr), ('RANK', ranks)):
        Z = T(both)
        C, P = Z[:len(C0)], Z[len(C0):]
        print(f'\n--- {tag} ---')
        print(f'   {"pair":14s} {"ctrl":>8s} {"pt":>8s} {"diff":>8s} '
              f'{"p":>8s} {"q":>7s}')
        ps, ds = [], []
        for j, c in enumerate(PAIRS):
            dd, pp = perm(C[:, j], P[:, j])
            ps.append(pp); ds.append(dd)
        qs = bh(ps)
        for j, c in enumerate(PAIRS):
            print(f'   {c:14s} {C[:, j].mean():8.3f} {P[:, j].mean():8.3f} '
                  f'{ds[j]:+8.3f} {ps[j]:8.4f} {qs[j]:7.3f}'
                  + ('  *' if qs[j] < .05 else ''))
        pc_c, pc_p = per_cat(C), per_cat(P)
        ps2, ds2 = [], []
        for k in range(4):
            dd, pp = perm(pc_c[:, k], pc_p[:, k])
            ps2.append(pp); ds2.append(dd)
        qs2 = bh(ps2)
        print(f'   {"category":14s} {"ctrl":>8s} {"pt":>8s} {"diff":>8s} '
              f'{"p":>8s} {"q":>7s}')
        for k, c in enumerate(CATS):
            print(f'   {c:14s} {pc_c[:, k].mean():8.3f} {pc_p[:, k].mean():8.3f} '
                  f'{ds2[k]:+8.3f} {ps2[k]:8.4f} {qs2[k]:7.3f}'
                  + ('  *' if qs2[k] < .05 else ''))
        wc, wp = within(C), within(P)
        dd, pp = perm(wc, wp)
        print(f'   consistency: ctrl {np.nanmean(wc):+.3f}  pt {np.nanmean(wp):+.3f}'
              f'  diff {dd:+.3f}  p={pp:.4f}' + ('  *' if pp < .05 else ''))
    print()

print('READING IT')
print('  A pair effect present in RAW but gone in CLR was closure.')
print('  A pair effect that survives CLR and RANK is not a constraint artifact.')
print('  Consistency should be similar across all three; if it is not, the')
print('  transform is doing something unintended.')
