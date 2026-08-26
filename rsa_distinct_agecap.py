#!/usr/bin/env python3
"""
rsa_distinct_agecap.py — distinctiveness AND geometry, patients vs controls,
with and without the agreed age<=23 exclusion.

Reproduces the comparison that showed the age cap was not applied in the
2026-08-25 session's RSA analyses. sub-091/SI (scan age 37.3) was the only
patient over 23; controls sub-095 (38.5) and sub-096 (27.9) were also included.

Sign convention: liu_distinctiveness stores SIMILARITY, so HIGHER = LESS
distinct. d is computed pt - ctrl, so d > 0 = patients less distinct (deficit).

Session rule: controls first session, patients last session. Patients contribute
only their intact hemisphere.

Usage:
  python rsa_distinct_agecap.py
  python rsa_distinct_agecap.py --cap 23
  python rsa_distinct_agecap.py --roi-set primary_strict
  python rsa_distinct_agecap.py --geometry          # add the fisher_r pair tests
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
RSA = GIT / 'D_liu' / 'rsa_v1_harmonized.csv'
INFO = GIT / 'sub_info.csv'

ROI_SETS = {
    'primary_strict': ['object_LOC', 'house_PPA_strict', 'face_FFA', 'word_VWFA'],
    'primary':        ['object_LOC', 'house_PPA', 'face_FFA', 'word_VWFA'],
}
SHORT = {'object_LOC': 'object', 'house_PPA_strict': 'house',
         'house_PPA': 'house', 'face_FFA': 'face', 'word_VWFA': 'word'}
PAIRS = ['face-house', 'face-object', 'face-word',
         'house-object', 'house-word', 'object-word']

N_PERM = 20000
SEED = 1


def cohens_d(a, b):
    """pt - ctrl.  a = control values, b = patient values."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) +
                  (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(b) - np.mean(a)) / sp


def perm_p(a, b, n_perm=N_PERM, seed=SEED):
    """Two-sided label-shuffle permutation test on the difference in means."""
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(b) - np.mean(a))
    pool = np.concatenate([a, b])
    nb = len(b)
    k = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(np.mean(pool[:nb]) - np.mean(pool[nb:])) >= obs - 1e-12:
            k += 1
    return (k + 1) / (n_perm + 1)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = prev
    return q


def load(rois, cap):
    info = pd.read_csv(INFO)
    info['session'] = info['ses'].str.replace('ses-', '', regex=False).astype(int)

    d = pd.read_csv(RSA)
    d['session'] = d['session'].astype(int)
    d = d[d['category'].isin(rois)].drop_duplicates(
        ['subject_id', 'session', 'hemi', 'category', 'pair'])

    # session rule: controls first, patients last
    ctl = d[d['group'] == 'control'].copy()
    s = ctl.groupby('subject_id')['session'].min().rename('sx')
    ctl = ctl.join(s, on='subject_id')
    ctl = ctl[ctl['session'] == ctl['sx']]

    pat = d[d['group'] == 'OTC'].copy()
    s = pat.groupby('subject_id')['session'].max().rename('sx')
    pat = pat.join(s, on='subject_id')
    pat = pat[pat['session'] == pat['sx']]

    # patients: intact hemisphere only
    pat['intact'] = pat['intact_hemi'].map({'left': 'l', 'right': 'r'})
    pat = pat[pat['hemi'] == pat['intact']]

    for f in (ctl, pat):
        f.drop(columns=['sx'], inplace=True, errors='ignore')
    ctl = ctl.merge(info[['sub', 'session', 'age']],
                    left_on=['subject_id', 'session'],
                    right_on=['sub', 'session'], how='left')
    pat = pat.merge(info[['sub', 'session', 'age']],
                    left_on=['subject_id', 'session'],
                    right_on=['sub', 'session'], how='left')

    if cap is not None:
        dropped_c = sorted(ctl.loc[ctl['age'] > cap, 'subject_id'].unique())
        dropped_p = sorted(pat.loc[pat['age'] > cap, 'subject_id'].unique())
        if dropped_c or dropped_p:
            print(f'age > {cap} excluded — controls: {dropped_c}   '
                  f'patients: {dropped_p}')
        ctl = ctl[ctl['age'] <= cap]
        pat = pat[pat['age'] <= cap]
    return ctl, pat


def report(ctl, pat, rois, label):
    print(f'\n{"=" * 72}\n{label}')
    for hemi, hl in [('l', 'LH-intact'), ('r', 'RH-intact')]:
        n_c = ctl.loc[ctl['hemi'] == hemi, 'subject_id'].nunique()
        n_p = pat.loc[pat['intact'] == hemi, 'subject_id'].nunique()
        ds, ps, means = [], [], []
        for roi in rois:
            a = (ctl[(ctl['hemi'] == hemi) & (ctl['category'] == roi)]
                 .drop_duplicates('subject_id')['liu_distinctiveness']
                 .dropna().values)
            b = (pat[(pat['intact'] == hemi) & (pat['hemi'] == hemi) &
                     (pat['category'] == roi)]
                 .drop_duplicates('subject_id')['liu_distinctiveness']
                 .dropna().values)
            # NOTE drop_duplicates('subject_id') keeps one row per subject, so
            # the pair rows collapse correctly (liu_distinctiveness is constant
            # within subject x hemi x ROI)
            ds.append(cohens_d(a, b))
            ps.append(perm_p(a, b))
            means.append((a.mean(), b.mean(), len(a), len(b)))
        qs = bh_fdr(ps)
        print(f'\n  {hl}   n ctrl={n_c}  n pt={n_p}')
        print(f'  {"ROI":8s} {"ctrl":>7s} {"pt":>7s} {"nC":>3s} {"nP":>3s} '
              f'{"d":>7s} {"p":>7s} {"q":>7s}')
        for roi, (am, bm, na, nb), dd, pp, qq in zip(rois, means, ds, ps, qs):
            star = ' **' if qq < .05 else (' *' if pp < .05 else '')
            print(f'  {SHORT[roi]:8s} {am:7.3f} {bm:7.3f} {na:3d} {nb:3d} '
                  f'{dd:+7.3f} {pp:7.4f} {qq:7.4f}{star}')



def report_geometry(ctl, pat, rois, label):
    """Per ROI: Fisher-combined omnibus over the 6 pairs, then BH across pairs.

    Sign convention here is pt - ctrl on fisher_r, so d > 0 = patients MORE
    similar for that pair (blending). Note this is the OPPOSITE sign convention
    to the geometry rows in stats_results_harmonized_corrected.csv, which use
    ctrl - pt.
    """
    from scipy import stats as _st
    print(f'\n{"=" * 72}\nGEOMETRY — {label}')
    print('d = pt - ctrl on fisher_r;  d > 0 = patients MORE similar (blending)')
    print('omnibus = Fisher-combined over the 6 pair permutations')
    for hemi, hl in [('l', 'LH-intact'), ('r', 'RH-intact')]:
        print(f'\n  {hl}')
        for roi in rois:
            ps, ds, ns = [], [], []
            for pr in PAIRS:
                a = ctl[(ctl['hemi'] == hemi) & (ctl['category'] == roi) &
                        (ctl['pair'] == pr)]['fisher_r'].dropna().values
                b = pat[(pat['intact'] == hemi) & (pat['hemi'] == hemi) &
                        (pat['category'] == roi) &
                        (pat['pair'] == pr)]['fisher_r'].dropna().values
                if len(a) < 3 or len(b) < 3:
                    ps.append(np.nan); ds.append(np.nan); ns.append((len(a), len(b)))
                    continue
                ps.append(perm_p(a, b)); ds.append(cohens_d(a, b))
                ns.append((len(a), len(b)))
            ok = ~np.isnan(ps)
            if ok.sum() < 2:
                print(f'    {SHORT[roi]:8s} insufficient data')
                continue
            chi = -2 * np.sum(np.log(np.clip(np.array(ps)[ok], 1e-300, 1)))
            p_om = 1 - _st.chi2.cdf(chi, 2 * ok.sum())
            qs = bh_fdr(np.where(ok, ps, 1.0))
            sig = '; '.join(f'{pr} {dd:+.2f} (q={qq:.3f})'
                            for pr, dd, qq, o in zip(PAIRS, ds, qs, ok)
                            if o and qq < .05)
            flag = ' **' if p_om < .05 else ''
            print(f'    {SHORT[roi]:8s} omnibus p={p_om:.4f}{flag}   '
                  f'{sig if sig else "no pair survives FDR"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=float, default=23.0,
                    help='max scan age; use -1 for no cap')
    ap.add_argument('--roi-set', choices=list(ROI_SETS),
                    default='primary_strict')
    ap.add_argument('--both', action='store_true',
                    help='report capped and uncapped side by side')
    ap.add_argument('--geometry', action='store_true',
                    help='also run the fisher_r pair / omnibus tests')
    args = ap.parse_args()

    rois = ROI_SETS[args.roi_set]
    print(f'ROIs: {rois}')
    print('liu_distinctiveness stores SIMILARITY -> HIGHER = LESS distinct')
    print('d = pt - ctrl, so d > 0 = patients less distinct (deficit)')
    print(f'permutation: {N_PERM} shuffles, seed {SEED}; BH-FDR across '
          f'{len(rois)} ROIs within model')

    caps = [None, args.cap] if args.both else \
           [None if args.cap < 0 else args.cap]
    for cap in caps:
        ctl, pat = load(rois, cap)
        lab = 'NO AGE CAP' if cap is None else f'AGE CAP <= {cap:g}'
        report(ctl, pat, rois, lab)
        if args.geometry:
            report_geometry(ctl, pat, rois, lab)


if __name__ == '__main__':
    main()
