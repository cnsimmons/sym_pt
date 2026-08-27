#!/usr/bin/env python3
"""
marlene_roi.py — the per-ROI table, third piece of the series.

WHY THIS EXISTS
  marlene_grid.py pools all four ROIs into a single category x group
  interaction, so an effect carried by ONE ROI is invisible there. The two
  headline results in the manuscript are exactly of that kind:
    rVWFA distinctiveness reduced in RH-intact patients (d = 0.91, q = .031)
    rFFA face-word less differentiable
  Neither appears anywhere in the grid or the LMM. This script reports every
  ROI separately, and for geometry every category PAIR separately, so the
  series is complete.

WHAT IT REPORTS
  distinctiveness, peak_z   one row per ROI: group difference, Cohen's d,
                            permutation p, BH q within the 4-ROI family
  geometry                  one row per ROI x pair (24 rows): same statistics,
                            BH q within the 6-pair family for that ROI,
                            plus a per-ROI omnibus row over the 6 pairs

  Six comparisons, matching marlene_grid.py:
    1  LH ctrl   vs LH-intact pt      PRIMARY
    2  RH ctrl   vs RH-intact pt      PRIMARY
    3  LH-int pt vs RH-intact pt      PRIMARY
    4  RH ctrl   vs LH-intact pt      supplemental, crossed
    5  LH ctrl   vs RH-intact pt      supplemental, crossed
    6  LH ctrl   vs RH ctrl           supplemental, paired within subject

SIGN CONVENTION — differs from the manuscript, read this
  All measures are oriented so HIGHER = MORE selective / MORE distinct /
  MORE separated, matching marlene_grid.py. diff = group B - group A.

  The manuscript reports distinctiveness and geometry on the RAW SIMILARITY
  scale, where higher = LESS distinct. So the manuscript's rVWFA value of
  +0.358 appears here as -0.358. Same effect, flipped sign. The higher_group
  column states which group is higher on the oriented scale, so direction can
  be read without tracking the flip.

Usage
  python marlene_roi.py
  python marlene_roi.py --csv roi.csv           # WRITE requires --csv
  python marlene_roi.py --comparisons 1 2 3
  python marlene_roi.py --measures distinctiveness geometry
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
RSA = GIT / 'D_liu' / 'rsa_v1_harmonized.csv'
UNI = GIT / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv'
INFO = GIT / 'sub_info.csv'

ROIS = ['object_LOC', 'house_PPA_strict', 'face_FFA', 'word_VWFA']
PAIRS = ['face-house', 'face-object', 'face-word',
         'house-object', 'house-word', 'object-word']
AGE_CAP = 23.0

COMPARISONS = {
    1: 'A = LH ctrl        B = LH-intact pt',
    2: 'A = RH ctrl        B = RH-intact pt',
    3: 'A = LH-intact pt   B = RH-intact pt',
    4: 'A = RH ctrl        B = LH-intact pt  [crossed]',
    5: 'A = LH ctrl        B = RH-intact pt  [crossed]',
    6: 'A = LH ctrl        B = RH ctrl       [paired]',
}
PRIMARY = (1, 2, 3)
PAIRED = (6,)


# ---------------------------------------------------------------- data loading

def _sessions(df, group, rule):
    x = df[df['group'] == group].copy()
    s = x.groupby('subject_id')['session'].agg(rule).rename('sx')
    x = x.join(s, on='subject_id')
    return x[x['session'] == x['sx']].drop(columns=['sx'], errors='ignore')


def load_measure(measure, cap=AGE_CAP, quiet=False):
    """(ctl, pat) long frames, val oriented HIGHER = MORE selective / distinct."""
    info = pd.read_csv(INFO)
    info['session'] = info['ses'].str.replace('ses-', '', regex=False).astype(int)

    if measure == 'peak_z':
        d = pd.read_csv(UNI)
        d['session'] = d['session'].astype(int)
        d = d[d['category'].isin(ROIS)].drop_duplicates(
            ['subject_id', 'session', 'hemi', 'category'])
        d = d.rename(columns={'category': 'roi', 'peak_z': 'val'})
        flip = False
    elif measure == 'distinctiveness':
        d = pd.read_csv(RSA)
        d['session'] = d['session'].astype(int)
        d = d[d['category'].isin(ROIS)].drop_duplicates(
            ['subject_id', 'session', 'hemi', 'category'])
        d = d.rename(columns={'category': 'roi', 'liu_distinctiveness': 'val'})
        flip = True
    elif measure == 'geometry':
        d = pd.read_csv(RSA)
        d['session'] = d['session'].astype(int)
        d = d[d['category'].isin(ROIS)].drop_duplicates(
            ['subject_id', 'session', 'hemi', 'category', 'pair'])
        d = d.rename(columns={'category': 'roi', 'fisher_r': 'val'})
        flip = True
    else:
        raise ValueError(measure)

    def add_age(x):
        return x.merge(info[['sub', 'session', 'age']],
                       left_on=['subject_id', 'session'],
                       right_on=['sub', 'session'], how='left')

    ctl = add_age(_sessions(d, 'control', 'min'))
    pat = _sessions(d, 'OTC', 'max')
    pat['intact'] = pat['intact_hemi'].map({'left': 'l', 'right': 'r'})
    pat = add_age(pat[pat['hemi'] == pat['intact']])

    if flip:
        ctl['val'] = -ctl['val']
        pat['val'] = -pat['val']

    if cap is not None:
        dc = sorted(ctl.loc[ctl['age'] > cap, 'subject_id'].unique())
        dp = sorted(pat.loc[pat['age'] > cap, 'subject_id'].unique())
        if (dc or dp) and not quiet:
            print(f'  [{measure}] age > {cap:g} excluded — '
                  f'controls {dc}, patients {dp}')
        ctl = ctl[ctl['age'] <= cap]
        pat = pat[pat['age'] <= cap]

    cols = ['subject_id', 'hemi', 'roi', 'val']
    if measure == 'geometry':
        cols = cols + ['pair']
    return ctl[cols], pat[cols + ['intact']]


def build_frame(ctl, pat, comparison, pair_level=False):
    """subject_id, roi, val, grp (+pair). grp=1 is GROUP B."""
    keep = ['subject_id', 'roi', 'val'] + (['pair'] if pair_level else [])
    if comparison == 1:
        a = ctl[ctl['hemi'] == 'l'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'l'][keep].assign(grp=1)
    elif comparison == 2:
        a = ctl[ctl['hemi'] == 'r'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'r'][keep].assign(grp=1)
    elif comparison == 3:
        a = pat[pat['intact'] == 'l'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'r'][keep].assign(grp=1)
    elif comparison == 4:
        a = ctl[ctl['hemi'] == 'r'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'l'][keep].assign(grp=1)
    elif comparison == 5:
        a = ctl[ctl['hemi'] == 'l'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'r'][keep].assign(grp=1)
    elif comparison == 6:
        a = ctl[ctl['hemi'] == 'l'][keep].assign(grp=0)
        b = ctl[ctl['hemi'] == 'r'][keep].assign(grp=1)
    else:
        raise ValueError(comparison)
    return pd.concat([a, b], ignore_index=True).dropna(subset=['val'])


# ------------------------------------------------------------------- the tests

def cohens_d(x, y):
    """Group B minus group A, pooled SD."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                  (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return np.nan if sp == 0 else (np.mean(y) - np.mean(x)) / sp


def cell_test(d, n_perm, seed=0, paired=False):
    """One ROI (or ROI x pair) cell: diff, d, permutation p, n per group.

    Between-subject: shuffle the group label across subjects.
    Paired: flip each subject's own two labels together with p = 0.5, which is
    the correct exchangeability when every subject supplies both groups.
    """
    a = d.loc[d['grp'] == 0, 'val'].to_numpy(float)
    b = d.loc[d['grp'] == 1, 'val'].to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, len(a), len(b)

    rng = np.random.default_rng(seed)
    y = d['val'].to_numpy(float)
    sid = d['subject_id']

    if paired:
        # subject-wise difference (B - A); sign-flip test
        w = (d.pivot_table(index='subject_id', columns='grp', values='val')
               .dropna())
        diffs = (w[1] - w[0]).to_numpy(float)
        obs = float(diffs.mean())
        k = 0
        for _ in range(n_perm):
            s = rng.choice([-1.0, 1.0], size=len(diffs))
            if abs(float((diffs * s).mean())) >= abs(obs) - 1e-12:
                k += 1
        dd = obs / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else np.nan
        return obs, dd, (k + 1) / (n_perm + 1), len(diffs), len(diffs)

    obs = float(np.mean(b) - np.mean(a))
    g = d.drop_duplicates('subject_id').set_index('subject_id')['grp']
    ids, labels = g.index.to_numpy(), g.to_numpy(float)
    k = 0
    for _ in range(n_perm):
        m = dict(zip(ids, rng.permutation(labels)))
        gv = sid.map(m).to_numpy(float)
        if abs(float(y[gv == 1].mean() - y[gv == 0].mean())) >= abs(obs) - 1e-12:
            k += 1
    return obs, cohens_d(a, b), (k + 1) / (n_perm + 1), len(a), len(b)


def bh(pvals):
    """Benjamini-Hochberg q, NaN-safe."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def omnibus_pairs(d, n_perm, seed=0, paired=False):
    """Per-ROI omnibus over the 6 pairs: sum of squared pair differences,
    tested by the same label permutation. Direction-free, so it fires whenever
    the pair profile differs in any pattern."""
    def stat(gv):
        s = 0.0
        for pr in d['pair'].unique():
            m = d['pair'].to_numpy() == pr
            aa, bb = d['val'].to_numpy(float)[m & (gv == 0)], \
                     d['val'].to_numpy(float)[m & (gv == 1)]
            if len(aa) and len(bb):
                s += (bb.mean() - aa.mean()) ** 2
        return s

    gv0 = d['grp'].to_numpy(float)
    obs = stat(gv0)
    rng = np.random.default_rng(seed)
    sid = d['subject_id']
    k = 0
    if paired:
        ids = sid.unique()
        for _ in range(n_perm):
            flip = dict(zip(ids, rng.integers(0, 2, len(ids)).astype(float)))
            if stat(np.abs(gv0 - sid.map(flip).to_numpy(float))) >= obs - 1e-12:
                k += 1
    else:
        g = d.drop_duplicates('subject_id').set_index('subject_id')['grp']
        ids, labels = g.index.to_numpy(), g.to_numpy(float)
        for _ in range(n_perm):
            m = dict(zip(ids, rng.permutation(labels)))
            if stat(sid.map(m).to_numpy(float)) >= obs - 1e-12:
                k += 1
    return obs, (k + 1) / (n_perm + 1)


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=float, default=AGE_CAP)
    ap.add_argument('--n-perm', type=int, default=5000)
    ap.add_argument('--measures', nargs='+',
                    default=['peak_z', 'distinctiveness', 'geometry'])
    ap.add_argument('--comparisons', nargs='+', type=int,
                    default=sorted(COMPARISONS))
    ap.add_argument('--csv', default=None, help='write the table to this path')
    args = ap.parse_args()

    cap = None if args.cap < 0 else args.cap
    comps = [c for c in args.comparisons if c in COMPARISONS]

    print(f'ROIs: {ROIS}')
    print(f'age cap: {"none" if cap is None else cap}   '
          f'permutations: {args.n_perm}')
    print(f'comparisons: {comps}  (primary {list(PRIMARY)}, rest supplemental)')
    print('diff = group B - group A, on the ORIENTED scale')
    print('  (higher = MORE selective / MORE distinct / MORE separated)')
    print('  NOTE the manuscript reports distinctiveness and geometry on the')
    print('  raw SIMILARITY scale, so its signs are the reverse of these.')
    print('FDR: BH within the 4-ROI family; for geometry, within the 6-pair '
          'family per ROI')

    rows = []
    for measure in args.measures:
        pair_level = (measure == 'geometry')
        ctl, pat = load_measure(measure, cap=cap)
        print(f'\n{"=" * 78}\nMEASURE: {measure}')

        for comp in comps:
            is_paired = comp in PAIRED
            df = build_frame(ctl, pat, comp, pair_level=pair_level)
            tag = '' if comp in PRIMARY else '   [SUPPLEMENTAL]'
            print(f'\n  {comp}. {COMPARISONS[comp]}{tag}')

            cells = []
            for roi in ROIS:
                sub = df[df['roi'] == roi]
                if pair_level:
                    ob, op = omnibus_pairs(sub, args.n_perm, paired=is_paired)
                    prs = []
                    for pr in PAIRS:
                        s2 = sub[sub['pair'] == pr]
                        diff, dd, p, na, nb = cell_test(
                            s2, args.n_perm, paired=is_paired)
                        prs.append(dict(roi=roi, pair=pr, diff=diff, d=dd,
                                        p=p, n_a=na, n_b=nb))
                    qs = bh([x['p'] for x in prs])
                    for x, q in zip(prs, qs):
                        x['q_fdr'] = q
                    cells.append(dict(roi=roi, pair='OMNIBUS (6 pairs)',
                                      diff=ob, d=np.nan, p=op, q_fdr=np.nan,
                                      n_a=prs[0]['n_a'], n_b=prs[0]['n_b']))
                    cells.extend(prs)
                else:
                    diff, dd, p, na, nb = cell_test(
                        sub, args.n_perm, paired=is_paired)
                    cells.append(dict(roi=roi, pair='', diff=diff, d=dd,
                                      p=p, n_a=na, n_b=nb))
            if not pair_level:
                qs = bh([c['p'] for c in cells])
                for c, q in zip(cells, qs):
                    c['q_fdr'] = q

            hdr = f'     {"ROI":18s} {"pair":18s}' if pair_level \
                  else f'     {"ROI":18s}'
            print(hdr + f' {"diff":>8s} {"d":>7s} {"p":>8s} {"q":>8s}')
            for c in cells:
                nm = f'     {c["roi"]:18s}'
                if pair_level:
                    nm += f' {c["pair"]:18s}'
                ds = '     —' if c['d'] != c['d'] else f'{c["d"]:+7.2f}'
                qs_ = '     —' if c['q_fdr'] != c['q_fdr'] else f'{c["q_fdr"]:8.4f}'
                star = ' *' if (c['q_fdr'] == c['q_fdr']
                                and c['q_fdr'] < .05) else ''
                print(f'{nm} {c["diff"]:+8.3f} {ds} {c["p"]:8.4f} {qs_}{star}')

                hg = ('—' if c['diff'] != c['diff'] else
                      ('B' if c['diff'] > 0 else 'A'))
                rows.append(dict(
                    measure=measure, comparison=comp,
                    comparison_name=COMPARISONS[comp],
                    role='primary' if comp in PRIMARY else 'supplemental',
                    paired=is_paired, roi=c['roi'], pair=c['pair'],
                    diff=c['diff'], cohen_d=c['d'], p=c['p'],
                    q_fdr=c['q_fdr'], higher_group=hg,
                    n_a=c['n_a'], n_b=c['n_b'],
                    age_cap=cap, roi_set='primary_strict'))

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'\nWrote {args.csv}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
