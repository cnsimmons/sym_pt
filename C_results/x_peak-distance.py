#!/usr/bin/env python3
"""
Peak Distance Table: Cross-sectional peak displacement (mm).
Loads precomputed patient_distances.csv, bootstraps control distances,
prints tab-separated table for Word paste.

Save: /user_data/csimmon2/git_repos/sym_pt/C_results/figures/
Run:  python table_peak_distance.py
"""

import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path(processed_dir)
PEAK_DIR = BASE / 'group_results' / 'peak_coords'
EXCLUDE  = ['sub-017']
CATEGORIES = ['face', 'house', 'object', 'word']
N_ITER   = 10_000
RNG      = np.random.default_rng(42)

# ── Load ──────────────────────────────────────────────────────────────────────
dist_file = PEAK_DIR / 'patient_distances.csv'
if not dist_file.exists():
    print(f'ERROR: {dist_file} not found')
    print('Run 05_calc_peak_coords.py first.')
    sys.exit(1)

df = pd.read_csv(dist_file)
df = df[~df['sub'].isin(EXCLUDE)]
print(f'Loaded {len(df)} rows from patient_distances.csv')
print(f'Columns: {list(df.columns)}')
print(f'Groups: {df["group"].unique()}')
print(f'Categories: {sorted(df["category"].unique())}')

# ── Bootstrap helpers ─────────────────────────────────────────────────────────
def bootstrap_ci_replace(vals, n_iter=N_ITER, rng=RNG):
    """Bootstrap 95% CI with replacement (for patient groups)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    boot = np.array([rng.choice(v, size=len(v), replace=True).mean()
                     for _ in range(n_iter)])
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

# ── Also load bootstrap resamples for control CIs if available ────────────────
resample_file = PEAK_DIR / 'distance_resamples.csv'
has_resamples = resample_file.exists()
if has_resamples:
    df_resamp = pd.read_csv(resample_file)
    print(f'Loaded control resamples: {list(df_resamp.columns)[:5]}...')

# ── Build table ───────────────────────────────────────────────────────────────
print('\n' + '=' * 90)
print('PEAK DISTANCE TABLE: Mean Euclidean Distance to Control Peaks (mm)')
print('=' * 90)

# Separate by intact hemisphere for anatomical homolog
# L-res (intact RH) → compare to ctrl RH distances
# R-res (intact LH) → compare to ctrl LH distances

for hemi_label, ctrl_hemi, res_label in [
    ('right', 'right', 'RH'),   # L-res intact RH vs Ctrl RH
    ('left',  'left',  'LH'),   # R-res intact LH vs Ctrl LH
]:
    print(f'\n  {res_label}: L-res (intact {res_label})' if res_label == 'RH'
          else f'\n  {res_label}: R-res (intact {res_label})')

    for cat in CATEGORIES:
        # Patient values for this hemisphere × category
        if 'intact_hemi' in df.columns:
            pt = df[(df['category'] == cat) &
                    (df['intact_hemi'] == hemi_label) &
                    (df['group'] == 'OTC')]
        elif 'hemi' in df.columns:
            pt = df[(df['category'] == cat) &
                    (df['hemi'] == hemi_label) &
                    (df['group'] == 'OTC')]
        else:
            print(f'  ERROR: Cannot determine hemisphere column')
            continue

        pt_vals = pt['mean_dist_mm'].dropna().values
        pt_m = np.nanmean(pt_vals) if len(pt_vals) > 0 else np.nan
        pt_lo, pt_hi = bootstrap_ci_replace(pt_vals)

        # Control CI from resamples
        col_name = f'{cat}_{ctrl_hemi}'
        if has_resamples and col_name in df_resamp.columns:
            ctrl_boot = df_resamp[col_name].dropna().values
            ctrl_lo = np.percentile(ctrl_boot, 2.5)
            ctrl_hi = np.percentile(ctrl_boot, 97.5)
            ctrl_m = np.mean(ctrl_boot)
        else:
            ctrl_m, ctrl_lo, ctrl_hi = np.nan, np.nan, np.nan

        sig = '*' if (np.isfinite(pt_m) and np.isfinite(ctrl_hi)
                      and pt_m > ctrl_hi) else ''

        print(f'    {cat:<8} Ctrl: {ctrl_m:>5.1f} [{ctrl_lo:.1f}, {ctrl_hi:.1f}]  '
              f'OTC: {pt_m:>5.1f} [{pt_lo:.1f}, {pt_hi:.1f}]{sig}  n={len(pt_vals)}')

# ── Tab-separated for Word ────────────────────────────────────────────────────
print('\n' + '=' * 90)
print('TAB-SEPARATED TABLE (copy into Word):')
print('=' * 90)
print('\tFace\tHouse\tObject\tWord')

for hemi_label, ctrl_hemi, res_label in [
    ('right', 'right', 'RH'),
    ('left',  'left',  'LH'),
]:
    ctrl_line = f'{res_label}: Ctrl (n=24)'
    pt_line   = f'{res_label}: OTC (n=8)'

    for cat in CATEGORIES:
        # Control
        col_name = f'{cat}_{ctrl_hemi}'
        if has_resamples and col_name in df_resamp.columns:
            ctrl_boot = df_resamp[col_name].dropna().values
            ctrl_m = np.mean(ctrl_boot)
            ctrl_lo = np.percentile(ctrl_boot, 2.5)
            ctrl_hi = np.percentile(ctrl_boot, 97.5)
        else:
            ctrl_m, ctrl_lo, ctrl_hi = np.nan, np.nan, np.nan

        # Patient
        if 'intact_hemi' in df.columns:
            pt = df[(df['category'] == cat) &
                    (df['intact_hemi'] == hemi_label) &
                    (df['group'] == 'OTC')]
        else:
            pt = df[(df['category'] == cat) &
                    (df['hemi'] == hemi_label) &
                    (df['group'] == 'OTC')]

        pt_vals = pt['mean_dist_mm'].dropna().values
        pt_m = np.nanmean(pt_vals) if len(pt_vals) > 0 else np.nan
        pt_lo, pt_hi = bootstrap_ci_replace(pt_vals)

        sig = '*' if (np.isfinite(pt_m) and np.isfinite(ctrl_hi)
                      and pt_m > ctrl_hi) else ''

        ctrl_line += f'\t{ctrl_m:.1f} [{ctrl_lo:.1f}, {ctrl_hi:.1f}]'
        pt_line   += f'\t{pt_m:.1f} [{pt_lo:.1f}, {pt_hi:.1f}]{sig}'

    print(ctrl_line)
    print(pt_line)

print('\n* = OTC mean above control 97.5th percentile')
print('RH = L-resection patients (intact RH) vs Control RH')
print('LH = R-resection patients (intact LH) vs Control LH')