#!/usr/bin/env python
"""
marlene_znorm_v2.py

Control-normalized patient-versus-patient comparison.

Each patient value is expressed as a z-score against the controls of that
patient's OWN intact side (LH-intact -> controls' left hemisphere, RH-intact ->
controls' right hemisphere), per cell. The two patient groups are then compared
on the z-scale, so the contrast no longer carries the normative difference
between the two control hemispheres.

Three quantities come out of the same normalization:
  (1) one-sample, each patient group's z against 0   -> does this group depart
                                                        from its own side?
  (2) two-sample, LH-intact z vs RH-intact z         -> do they depart by
                                                        different amounts?
  (3) profile test on z                              -> do they depart in a
                                                        different pattern?

Controls are z-scored leave-one-out against the other controls of their own
side, which puts them on the same scale and gives the validity check:
control-left and control-right z must land on top of each other. If
ctrl_check_p is significant the normalization has not removed the normative
hemispheric difference and nothing else in the output is interpretable.

Writes output only with --csv.

Usage:
    cd /user_data/csimmon2/git_repos/sym_pt
    python marlene_znorm_v2.py
    python marlene_znorm_v2.py --csv
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

REPO = '/user_data/csimmon2/git_repos/sym_pt'
UNIVAR_CSV = os.path.join(REPO, 'D_liu', 'univariate_v1_harmonized_sqrt.csv')
RSA_CSV = os.path.join(REPO, 'D_liu', 'rsa_v1_harmonized.csv')
OUT_CSV = os.path.join(REPO, 'D_liu', 'znorm_ptvspt.csv')

COL = {
    'sub': 'subject_id',
    'ses': 'ses_num',
    'status': 'status',        # 'control' | 'patient'
    'hemi': 'hemi',            # 'l' | 'r'
    'intact': 'intact_hemi',   # 'control' | 'left' | 'right'
    'cell': 'category',        # 'word_VWFA', 'face_FFA', ...
    'pair': 'pair',            # rsa only
}

CONTROL = 'control'
PATIENT = 'patient'
LEFT, RIGHT = 'left', 'right'
HEMI_MAP = {'l': LEFT, 'r': RIGHT}

# The four cells of record, in manuscript order. house_PPA_strict is the
# primary PPA parcel; house_PPA is the non-strict variant and is not used here.
CELLS = ['word_VWFA', 'face_FFA', 'house_PPA_strict', 'object_LOC']
PAIRS = ['face-word', 'house-word', 'object-word',
         'face-house', 'face-object', 'house-object']

# Overall separability is the mean of the six between-category pairs, derived
# below rather than read from a column. liu_distinctiveness (the mean of the
# three own-category pairs) is not used: that measure is no longer reported.
SEP6 = '_separability_mean6'

# (source, value column, label, kind)
# kind 'cell' -> one value per cell; kind 'pair' -> one value per cell x pair
MEASURES = [
    ('univar', 'peak_z', 'peak selectivity', 'cell'),
    ('univar', 'volume', 'number of selective voxels', 'cell'),
    ('rsa', SEP6, 'overall separability', 'cell'),
    ('rsa', 'fisher_r', 'between-category similarity', 'pair'),
]

# Cohort of record. Mirrors the EXCLUDE block in
# D_liu/verified/05_stats_harmony.py. Matched on digits, so 'sub-017', '017'
# and 'sub017' all resolve.
EXCLUDE = ['017',                # polymicrogyria
           '091', '095', '096',  # age cap <= 23
           '027', '084']         # control exclusions
EXCLUDE_SES = [('108', 2)]

N_PERM = 10000
RNG_SEED = 0
MIN_SD = 1e-8


# ----------------------------------------------------------------------------
# loading and cohort
# ----------------------------------------------------------------------------

def digits(s):
    m = re.findall(r'\d+', str(s))
    return str(int(m[-1])) if m else str(s)


def load(path, needed):
    if not os.path.exists(path):
        sys.exit('missing: %s' % path)
    df = pd.read_csv(path)
    gap = [c for c in needed if c not in df.columns]
    if gap:
        sys.exit('%s: missing columns %s\navailable: %s'
                 % (os.path.basename(path), gap, sorted(df.columns)))
    return df


def apply_cohort(df, name):
    sub, ses = COL['sub'], COL['ses']
    df = df.copy()
    df['_id'] = df[sub].map(digits)
    df[ses] = pd.to_numeric(df[ses], errors='coerce')

    n0 = df[sub].nunique()
    df = df[~df['_id'].isin([digits(e) for e in EXCLUDE])]
    for s, v in EXCLUDE_SES:
        df = df[~((df['_id'] == digits(s)) & (df[ses] == v))]

    df['is_control'] = df[COL['status']] == CONTROL

    # patients: last session. controls: first session.
    keep = []
    for _, block in df.groupby(sub):
        target = (block[ses].min() if bool(block['is_control'].iloc[0])
                  else block[ses].max())
        keep.append(block[block[ses] == target])
    df = pd.concat(keep, ignore_index=True)

    # side: controls from hemi, patients from intact_hemi (authoritative;
    # carries the sub-076 correction).
    df['side'] = np.where(df['is_control'],
                          df[COL['hemi']].map(HEMI_MAP),
                          df[COL['intact']])
    df = df[df['side'].isin([LEFT, RIGHT])]

    # consistency check: for patients, hemi should agree with intact_hemi
    pt = df[~df.is_control]
    bad = pt[pt[COL['hemi']].map(HEMI_MAP) != pt[COL['intact']]]
    n_bad = bad[sub].nunique()

    print('%s: %d subjects -> %d after exclusions | controls %d | '
          'LH-intact %d | RH-intact %d'
          % (name, n0, df[sub].nunique(),
             df[df.is_control][sub].nunique(),
             pt[pt.side == LEFT][sub].nunique(),
             pt[pt.side == RIGHT][sub].nunique()))
    if n_bad:
        print('   WARNING: %d patient(s) where hemi disagrees with '
              'intact_hemi: %s' % (n_bad, sorted(bad[sub].unique())))
    return df


# ----------------------------------------------------------------------------
# normalization
# ----------------------------------------------------------------------------

def znorm(block, value_col):
    """z-score one comparable quantity.

    Patients: against all controls of their own intact side.
    Controls: leave-one-out against the other controls of their own side.
    """
    block = block.copy()
    block['z'] = np.nan
    ref = {}

    for side in (LEFT, RIGHT):
        ctrl = block[block.is_control & (block.side == side)]
        vals = ctrl[value_col].dropna()
        if len(vals) < 3:
            continue
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        ref[side] = dict(n=len(vals), mean=mu, sd=sd)
        if sd < MIN_SD:
            continue

        pt = (~block.is_control) & (block.side == side)
        block.loc[pt, 'z'] = (block.loc[pt, value_col] - mu) / sd

        for i in vals.index:
            others = vals.drop(i)
            if len(others) < 3:
                continue
            o_sd = float(others.std(ddof=1))
            if o_sd < MIN_SD:
                continue
            block.at[i, 'z'] = (vals.at[i] - float(others.mean())) / o_sd

    return block, ref


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

def one_sample(z, rng):
    z = np.asarray(z, dtype=float)
    z = z[~np.isnan(z)]
    if len(z) < 5:
        return dict(n=len(z), mean=np.nan, p=np.nan)
    obs = abs(float(z.mean()))
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(z)))
    null = np.abs((flips * z).mean(axis=1))
    return dict(n=len(z), mean=float(z.mean()),
                p=(1 + int(np.sum(null >= obs))) / (1 + N_PERM))


def two_sample(a, b, rng):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 4 or len(b) < 4:
        return dict(n_lh=len(a), n_rh=len(b), diff=np.nan, d=np.nan,
                    p_mwu=np.nan, p=np.nan)
    try:
        _, p_mwu = stats.mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        p_mwu = np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                / (na + nb - 2))
    d = float((a.mean() - b.mean()) / s) if s > MIN_SD else np.nan
    pooled = np.concatenate([a, b])
    obs = abs(float(a.mean() - b.mean()))
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(pooled)
        null[i] = abs(p[:na].mean() - p[na:].mean())
    return dict(n_lh=na, n_rh=nb, diff=float(a.mean() - b.mean()), d=d,
                p_mwu=float(p_mwu) if p_mwu == p_mwu else np.nan,
                p=(1 + int(np.sum(null >= obs))) / (1 + N_PERM))


def profile(wide, group_level, a_lab, b_lab, rng):
    """Does the pattern of departure differ between two sets?

    Statistic: the set difference per level, centred on its own mean, summed as
    squares, so an overall level difference does not register as a pattern
    difference. Same profile statistic as the whole-OTC test. Label-shuffle
    permutation. Each subject's z is a fixed quantity, so the shuffle is exact.
    """
    wide = wide.dropna(axis=0, how='any')
    if wide.shape[0] < 8 or wide.shape[1] < 2:
        return dict(n=int(wide.shape[0]), k=int(wide.shape[1]),
                    stat=np.nan, p=np.nan)
    lab = wide.index.get_level_values(group_level).values
    X = wide.values
    if (lab == a_lab).sum() < 4 or (lab == b_lab).sum() < 4:
        return dict(n=int(wide.shape[0]), k=int(wide.shape[1]),
                    stat=np.nan, p=np.nan)

    def stat(l):
        dd = X[l == a_lab].mean(axis=0) - X[l == b_lab].mean(axis=0)
        return float(np.sum((dd - dd.mean()) ** 2))

    obs = stat(lab)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = stat(rng.permutation(lab))
    return dict(n=int(wide.shape[0]), k=int(wide.shape[1]), stat=obs,
                p=(1 + int(np.sum(null >= obs))) / (1 + N_PERM))


def fdr(p):
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = np.clip(np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = adj
    q[ok] = out
    return q


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def collapse_pairs(block, value_col, label, cell):
    """One row per subject per side.

    In the RSA file every row is a category x pair row, so a per-category
    quantity such as liu_distinctiveness is carried identically on all six pair
    rows of that category. Left uncollapsed it enters the tests as six
    pseudo-replicates per subject and the p-values are meaningless. Collapse to
    the per-subject mean, and warn if the six values were not in fact identical,
    since that would mean the column is not a per-category quantity.
    """
    sub = COL['sub']
    spread = (block.groupby([sub, 'side'])[value_col]
                   .agg(lambda s: float(s.max() - s.min())))
    if len(spread) and spread.max() > 1e-6:
        print('   WARNING: %s / %s varies within subject x side '
              '(max spread %.4g); collapsing by mean'
              % (label, cell, spread.max()))
    keep = ['side', 'is_control', COL['status'], COL['hemi'], COL['intact']]
    out = (block.sort_values(list(block.columns))
                .groupby([sub, 'side'], as_index=False)
                .agg({value_col: 'mean',
                      **{c: 'first' for c in keep if c != 'side'}}))
    return out


def run(df, value_col, label, kind, rng):
    sub = COL['sub']
    cellcol, paircol = COL['cell'], COL['pair']

    rows, zlong = [], []

    for cell in CELLS:
        cdf = df[df[cellcol] == cell]
        if cdf.empty:
            continue
        levels = PAIRS if kind == 'pair' else [cell]
        lvlcol = paircol if kind == 'pair' else cellcol
        for lev in levels:
            block = cdf[cdf[lvlcol] == lev] if kind == 'pair' else cdf
            block = block[block[value_col].notna()]
            if kind == 'cell' and paircol in block.columns:
                block = collapse_pairs(block, value_col, label, cell)
            if block.empty:
                continue
            block, ref = znorm(block, value_col)

            pts = block[~block.is_control]
            lh = pts[pts.side == LEFT]['z']
            rh = pts[pts.side == RIGHT]['z']
            o_lh, o_rh = one_sample(lh, rng), one_sample(rh, rng)
            ts = two_sample(lh, rh, rng)

            rows.append(dict(
                measure=label, cell=cell,
                level=(lev if kind == 'pair' else '-'),
                ctrl_L_n=ref.get(LEFT, {}).get('n', np.nan),
                ctrl_L_mean=ref.get(LEFT, {}).get('mean', np.nan),
                ctrl_L_sd=ref.get(LEFT, {}).get('sd', np.nan),
                ctrl_R_n=ref.get(RIGHT, {}).get('n', np.nan),
                ctrl_R_mean=ref.get(RIGHT, {}).get('mean', np.nan),
                ctrl_R_sd=ref.get(RIGHT, {}).get('sd', np.nan),
                lh_n=o_lh['n'], lh_mean_z=o_lh['mean'], lh_p_vs0=o_lh['p'],
                rh_n=o_rh['n'], rh_mean_z=o_rh['mean'], rh_p_vs0=o_rh['p'],
                ptvspt_diff_z=ts['diff'], ptvspt_d=ts['d'],
                ptvspt_p_mwu=ts['p_mwu'], ptvspt_p=ts['p']))

            for _, r in block.iterrows():
                if pd.isna(r['z']):
                    continue
                tag = ('ctrl' if r.is_control
                       else ('LH-intact' if r.side == LEFT else 'RH-intact'))
                zlong.append((r[sub], tag, r.side, cell,
                              lev if kind == 'pair' else cell, float(r['z'])))

    res = pd.DataFrame(rows)
    if res.empty:
        return res, pd.DataFrame()

    for c in ('lh_p_vs0', 'rh_p_vs0', 'ptvspt_p'):
        res['q_' + c] = fdr(res[c].values)

    z = pd.DataFrame(zlong,
                     columns=['sub', 'tag', 'side', 'cell', 'level', 'z'])

    def tests_on(block, tag):
        pt = block[block.tag != 'ctrl']
        w = pt.pivot_table(index=['sub', 'tag'], columns='level',
                           values='z', aggfunc='mean')
        w.index = w.index.set_names(['sub', 'grp'])
        inter = profile(w, 'grp', 'LH-intact', 'RH-intact', rng)

        ct = block[block.tag == 'ctrl']
        wc = ct.pivot_table(index=['sub', 'side'], columns='level',
                            values='z', aggfunc='mean')
        san = profile(wc, 'side', LEFT, RIGHT, rng)
        return dict(measure=label, scope=tag, n_levels=inter['k'],
                    n_patients=inter['n'], profile_stat=inter['stat'],
                    profile_p=inter['p'],
                    ctrl_check_n=san['n'], ctrl_check_stat=san['stat'],
                    ctrl_check_p=san['p'])

    omni = []
    if kind == 'pair':
        for cell in CELLS:
            b = z[z.cell == cell]
            if not b.empty:
                omni.append(tests_on(b, cell))
        pooled = (z.groupby(['sub', 'tag', 'side', 'level'])['z']
                   .mean().reset_index())
        pooled['cell'] = 'POOLED'
        omni.append(tests_on(pooled, 'POOLED (over cells)'))
    else:
        omni.append(tests_on(z, 'across the four cells'))

    return res, pd.DataFrame(omni)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='store_true', help='write output')
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    warnings.simplefilter('ignore', category=RuntimeWarning)

    base = [COL['sub'], COL['ses'], COL['status'], COL['hemi'],
            COL['intact'], COL['cell']]
    univar = load(UNIVAR_CSV, base + ['peak_z', 'volume'])
    rsa = load(RSA_CSV, base + [COL['pair'], 'fisher_r'])

    univar = apply_cohort(univar, 'univariate')
    rsa = apply_cohort(rsa, 'rsa       ')

    # overall separability: mean of the six pairs, per subject x side x cell
    rsa = rsa[rsa[COL['pair']].isin(PAIRS)].copy()
    n_pairs = (rsa.groupby([COL['sub'], 'side', COL['cell']])[COL['pair']]
                  .nunique())
    if len(n_pairs) and n_pairs.min() < len(PAIRS):
        print('   NOTE: %d of %d subject x side x cell blocks have fewer than '
              'six pairs; separability is the mean of those present'
              % (int((n_pairs < len(PAIRS)).sum()), len(n_pairs)))
    rsa[SEP6] = (rsa.groupby([COL['sub'], 'side', COL['cell']])['fisher_r']
                    .transform('mean'))
    print()

    missing = [c for c in CELLS if c not in set(univar[COL['cell']])]
    if missing:
        print('WARNING: cells absent from univariate file: %s' % missing)
        print()

    all_res, all_omni = [], []
    for source, value_col, label, kind in MEASURES:
        d = univar if source == 'univar' else rsa
        res, omni = run(d, value_col, label, kind, rng)
        if res.empty:
            print('skip %s: no rows after filtering' % label)
            continue
        all_res.append(res)
        all_omni.append(omni)

        print('=' * 76)
        print(label.upper())
        print('=' * 76)
        print(omni.to_string(index=False, float_format=lambda v: '%.4f' % v))
        print()
        cols = ['cell', 'level', 'lh_n', 'lh_mean_z', 'lh_p_vs0',
                'rh_mean_z', 'rh_p_vs0', 'ptvspt_diff_z', 'ptvspt_d',
                'ptvspt_p', 'q_ptvspt_p']
        print(res[cols].to_string(index=False,
                                  float_format=lambda v: '%.3f' % v))
        print()

    if args.csv and all_res:
        pd.concat(all_res, ignore_index=True).to_csv(OUT_CSV, index=False)
        om = OUT_CSV.replace('.csv', '_omnibus.csv')
        pd.concat(all_omni, ignore_index=True).to_csv(om, index=False)
        print('wrote %s' % OUT_CSV)
        print('wrote %s' % om)
    elif all_res:
        print('(nothing written; pass --csv)')


if __name__ == '__main__':
    main()