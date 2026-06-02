#!/usr/bin/env python3
"""
run_xs_stats.py — self-contained cross-sectional stats with corrections.

Reads the Liu-replication CSV (one row per subject x category x hemi, or
per-pair rows that duplicate the columns we use) and runs:

  - Sum-selectivity (log10 sum_selec_norm): patient vs control, per ROI x hemi
  - Distinctiveness (liu_distinctiveness, Fisher-z, raw): patient vs control
  - Peak distance (2D MNI centroid): patient vs control       [if MNI cols exist]
  - Control within-hemisphere baseline (L vs R, paired): sum-sel + distinctiveness

Corrections (deliberately liberal; small, precious patient n):
  - Per-measure BH-FDR across the 4 PRIMARY ROIs, WITHIN each hemisphere.
    Family = 4. This is the smallest defensible family (primary ROIs only,
    one hemisphere at a time), i.e. the most liberal BH option.
  - rVWFA distinctiveness is ALSO printed standalone as the pre-specified
    single-case test (raw p, no family) per the manuscript.
  - Per-category WTA + TFCE are separate pipelines; not recomputed here.

All perm tests: 10,000 iterations, seed 42, two-sided.

Usage:
  python run_xs_stats.py                       # uses default CSV path
  python run_xs_stats.py --csv /path/to.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CSV = '/user_data/csimmon2/git_repos/sym_pt/liu_exact_replication_v2.csv'
PRIMARY_ROIS = ['face_FFA', 'house_PPA', 'object_LOC', 'word_VWFA']
N_PERM = 10000
SEED = 42

# ── BH-FDR (liberal, no positive-dependence penalty) ─────────────────────────
def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values. NaNs pass through as NaN."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    pv = p[idx]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    out_idx = np.empty(n, dtype=int)
    out_idx[order] = np.arange(n)
    out[idx] = adj[out_idx]
    return out

# ── Permutation tests ────────────────────────────────────────────────────────
def perm_unpaired(ctrl, pt, n_perm=N_PERM, seed=SEED):
    """Two-sided label-shuffle perm test of mean diff (pt - ctrl)."""
    ctrl = np.asarray(ctrl, float); ctrl = ctrl[~np.isnan(ctrl)]
    pt = np.asarray(pt, float);     pt = pt[~np.isnan(pt)]
    if len(ctrl) < 3 or len(pt) < 2:
        return np.nan, np.nan, len(ctrl), len(pt), np.nan
    obs = pt.mean() - ctrl.mean()
    sd = np.sqrt(((len(ctrl) - 1) * ctrl.var(ddof=1) +
                  (len(pt) - 1) * pt.var(ddof=1)) /
                 (len(ctrl) + len(pt) - 2))          # pooled SD
    d = obs / sd if sd > 0 else np.nan
    pool = np.concatenate([ctrl, pt]); n_pt = len(pt)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pool)
        null[i] = perm[:n_pt].mean() - perm[n_pt:].mean()
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, p, len(ctrl), len(pt), d

def perm_paired(a, b, n_perm=N_PERM, seed=SEED):
    """Two-sided sign-flip paired perm test of mean(a - b). a,b aligned arrays."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan, np.nan, len(a), np.nan
    diff = a - b
    obs = diff.mean()
    d = obs / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        null[i] = (diff * signs).mean()
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, p, len(a), d

# ── Data ─────────────────────────────────────────────────────────────────────
def load(csv):
    df = pd.read_csv(csv)
    # collapse per-pair duplicate rows
    df = df.drop_duplicates(subset=['subject_id', 'category', 'hemi']).copy()
    df['is_ctrl'] = df['group'].astype(str).str.lower().eq('control')
    return df

def vals(df, subset, roi, hemi, col, log10=False):
    v = df[subset & (df['category'] == roi) & (df['hemi'] == hemi)][col].dropna().values
    if log10:
        v = np.log10(v[v > 0])
    return v

# ── Patient-vs-control, per measure, BH within hemisphere (family=4) ─────────
def patient_vs_control(df, measure, col, log10):
    ctrl = df['is_ctrl']
    rows = []
    for hemi, intact in [('l', 'left'), ('r', 'right')]:
        pt = (~df['is_ctrl']) & (df['intact_hemi'] == intact)
        for roi in PRIMARY_ROIS:
            c = vals(df, ctrl, roi, hemi, col, log10)
            p = vals(df, pt,   roi, hemi, col, log10)
            obs, pval, n_c, n_p, d = perm_unpaired(c, p)
            rows.append(dict(measure=measure, hemi=hemi.upper(), roi=roi,
                             n_ctrl=n_c, n_pt=n_p,
                             ctrl_m=c.mean() if len(c) else np.nan,
                             pt_m=p.mean() if len(p) else np.nan,
                             delta=obs, d=d, p_raw=pval))
    out = pd.DataFrame(rows)
    # BH within each hemisphere across the 4 primary ROIs  (family = 4)
    out['q_bh'] = np.nan
    for hemi in ('L', 'R'):
        m = out['hemi'] == hemi
        out.loc[m, 'q_bh'] = bh_fdr(out.loc[m, 'p_raw'].values)
    return out

# ── Control within-hemisphere (L vs R, paired across subjects) ───────────────
def control_within_hemi(df, measure, col, log10):
    ctrl = df['is_ctrl']
    rows = []
    for roi in PRIMARY_ROIS:
        # align L and R by subject
        sub_l = df[ctrl & (df['category'] == roi) & (df['hemi'] == 'l')][['subject_id', col]]
        sub_r = df[ctrl & (df['category'] == roi) & (df['hemi'] == 'r')][['subject_id', col]]
        m = sub_l.merge(sub_r, on='subject_id', suffixes=('_l', '_r')).dropna()
        a = m[f'{col}_l'].values; b = m[f'{col}_r'].values
        if log10:
            ok = (a > 0) & (b > 0); a = np.log10(a[ok]); b = np.log10(b[ok])
        obs, pval, n, d = perm_paired(a, b)
        rows.append(dict(measure=measure, roi=roi, n=n,
                         L_m=a.mean() if len(a) else np.nan,
                         R_m=b.mean() if len(b) else np.nan,
                         delta_LmR=obs, d=d, p_raw=pval))
    out = pd.DataFrame(rows)
    out['q_bh'] = bh_fdr(out['p_raw'].values)   # family = 4 ROIs
    return out

# ── Distance to 2D control centroid (needs MNI x,y) ──────────────────────────
def distance_test(df):
    xc = next((c for c in ['peak_x_mni', 'peak_x_native'] if c in df.columns), None)
    yc = next((c for c in ['peak_y_mni', 'peak_y_native'] if c in df.columns), None)
    if xc is None or yc is None:
        return None, None
    used_native = 'native' in xc
    ctrl = df['is_ctrl']
    rows = []
    for hemi, intact in [('l', 'left'), ('r', 'right')]:
        pt = (~df['is_ctrl']) & (df['intact_hemi'] == intact)
        for roi in PRIMARY_ROIS:
            c = df[ctrl & (df['category'] == roi) & (df['hemi'] == hemi)][[xc, yc]].dropna().values
            if len(c) < 3:
                rows.append(dict(measure='distance', hemi=hemi.upper(), roi=roi,
                                 n_ctrl=len(c), n_pt=0, ctrl_m=np.nan, pt_m=np.nan,
                                 delta=np.nan, d=np.nan, p_raw=np.nan)); continue
            cen = c.mean(0)
            cd = np.linalg.norm(c - cen, axis=1)
            pd_ = []
            for s in df[pt]['subject_id'].unique():
                r = df[(df['subject_id'] == s) & (df['category'] == roi) &
                       (df['hemi'] == hemi)][[xc, yc]].dropna().values
                if len(r): pd_.append(np.linalg.norm(r[0] - cen))
            pd_ = np.array(pd_)
            obs, pval, n_c, n_p, d = perm_unpaired(cd, pd_)
            rows.append(dict(measure='distance', hemi=hemi.upper(), roi=roi,
                             n_ctrl=n_c, n_pt=n_p, ctrl_m=cd.mean(),
                             pt_m=pd_.mean() if len(pd_) else np.nan,
                             delta=obs, d=d, p_raw=pval))
    out = pd.DataFrame(rows)
    out['q_bh'] = np.nan
    for hemi in ('L', 'R'):
        mm = out['hemi'] == hemi
        out.loc[mm, 'q_bh'] = bh_fdr(out.loc[mm, 'p_raw'].values)
    return out, used_native

# ── Printing ─────────────────────────────────────────────────────────────────
def fmt(v, w=8, p=3):
    return f'{v:>{w}.{p}f}' if pd.notna(v) else ' ' * w

def print_pvc(out, title):
    print(f'\n=== {title}  (patient vs control; BH family = 4 ROIs within hemi) ===')
    print(f'{"hemi":>4} {"ROI":12} {"nC":>3} {"nP":>3} {"ctrl_m":>8} {"pt_m":>8} '
          f'{"delta":>8} {"d":>7} {"p_raw":>8} {"q_BH":>8} sig')
    for _, r in out.iterrows():
        sig = '*' if pd.notna(r['q_bh']) and r['q_bh'] < .05 else ''
        print(f'{r["hemi"]:>4} {r["roi"]:12} {int(r["n_ctrl"]):>3} {int(r["n_pt"]):>3} '
              f'{fmt(r["ctrl_m"])} {fmt(r["pt_m"])} {fmt(r["delta"])} '
              f'{fmt(r["d"],7,2)} {fmt(r["p_raw"],8,4)} {fmt(r["q_bh"],8,4)} {sig}')

def print_ctrl(out, title):
    print(f'\n=== {title}  (control L vs R, paired; BH family = 4 ROIs) ===')
    print(f'{"ROI":12} {"n":>3} {"L_m":>8} {"R_m":>8} {"L-R":>8} {"d":>7} '
          f'{"p_raw":>8} {"q_BH":>8} sig')
    for _, r in out.iterrows():
        sig = '*' if pd.notna(r['q_bh']) and r['q_bh'] < .05 else ''
        print(f'{r["roi"]:12} {int(r["n"]):>3} {fmt(r["L_m"])} {fmt(r["R_m"])} '
              f'{fmt(r["delta_LmR"])} {fmt(r["d"],7,2)} {fmt(r["p_raw"],8,4)} '
              f'{fmt(r["q_bh"],8,4)} {sig}')

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    args = ap.parse_args()

    df = load(args.csv)
    n_pt = df[~df['is_ctrl']]['subject_id'].nunique()
    n_ct = df[df['is_ctrl']]['subject_id'].nunique()
    print('=' * 78)
    print(f'CSV: {args.csv}')
    print(f'Patients: {n_pt}   Controls: {n_ct}   Primary ROIs: {PRIMARY_ROIS}')
    print(f'Perm: {N_PERM} iters, seed {SEED}, two-sided. BH-FDR family = 4 (per hemi).')
    print('=' * 78)

    # Patient vs control
    print_pvc(patient_vs_control(df, 'sum_sel', 'sum_selec_norm', log10=True),
              'SUM-SELECTIVITY (log10)')
    dist = patient_vs_control(df, 'distinctiveness', 'liu_distinctiveness', log10=False)
    print_pvc(dist, 'DISTINCTIVENESS (Fisher-z, raw metric)')

    # Pre-specified single test (raw, no family) per manuscript
    r = dist[(dist['hemi'] == 'R') & (dist['roi'] == 'word_VWFA')]
    if len(r):
        rr = r.iloc[0]
        print(f'\n  [pre-specified single-case] rVWFA RH-intact distinctiveness: '
              f'delta={rr["delta"]:.3f}, d={rr["d"]:.2f}, p_raw={rr["p_raw"]:.4f} '
              f'(report RAW, no family)')

    # Distance (if MNI coords present)
    dres = distance_test(df)
    if dres[0] is not None:
        out, native = dres
        tag = 'distance (NATIVE coords — confirm MNI intended!)' if native else 'distance (MNI 2D)'
        print_pvc(out, tag.upper())
    else:
        print('\n=== DISTANCE: skipped (no peak_x/_y columns found) ===')

    # Control within-hemisphere baseline
    print('\n' + '-' * 78)
    print('CONTROL WITHIN-HEMISPHERE BASELINE (descriptive; raw + BH provided)')
    print('-' * 78)
    print_ctrl(control_within_hemi(df, 'sum_sel', 'sum_selec_norm', True),
               'CONTROL sum-sel L vs R')
    print_ctrl(control_within_hemi(df, 'distinctiveness', 'liu_distinctiveness', False),
               'CONTROL distinctiveness L vs R')

    print('\nDone. * = q_BH < .05. Per-category WTA + TFCE run in their own scripts.')

if __name__ == '__main__':
    main()