#!/usr/bin/env python3
"""
F30 Pilot Data Pull — All 5 pulls in one script.
Run from anywhere: python f30_data_pull.py
"""

import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path(processed_dir)
SEL_DIR  = BASE / 'group_results' / 'selectivity'
LIU_DIR  = BASE / 'group_results' / 'liu_distinctiveness'
GEO_DIR  = BASE / 'group_results' / 'geometry'

COPE_SET   = 'differential'
CATEGORIES = ['face', 'house', 'object', 'word']
EXCLUDE    = ['sub-017']

# Load all CSVs
df_sel = pd.read_csv(SEL_DIR / 'selectivity_summary.csv')
df_liu = pd.read_csv(LIU_DIR / f'liu_distinctiveness_{COPE_SET}.csv')
df_geo = pd.read_csv(GEO_DIR / f'geometry_{COPE_SET}.csv')
df_pw  = pd.read_csv(LIU_DIR / f'pairwise_correlations_{COPE_SET}.csv')

# Filter exclusions
df_liu = df_liu[~df_liu['subject_id'].isin(EXCLUDE)]
df_liu = df_liu[df_liu['category'].isin(CATEGORIES)]
df_geo = df_geo[~df_geo['subject_id'].isin(EXCLUDE)]
df_geo = df_geo[df_geo['category'].isin(CATEGORIES)]
df_pw  = df_pw[~df_pw['subject_id'].isin(EXCLUDE)]
df_pw  = df_pw[df_pw['category'].isin(CATEGORIES)]

# Sub info
sub_info = pd.read_csv('/user_data/csimmon2/git_repos/sym_pt/sub_info.csv')

# Pre-surgery sessions
PRE_SURG = {
    'sub-021': ['01', '1'], 'sub-045': ['01', '1'], 'sub-047': ['01', '1'],
    'sub-049': ['01', '1'], 'sub-070': ['01', '1'], 'sub-073': ['01', '1'],
    'sub-081': ['01', '1'], 'sub-086': ['01', '1'],
}

print("=" * 80)
print("F30 PILOT DATA PULL")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
# PULL 1: Identify best control exemplar
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PULL 1: Best control exemplar")
print("=" * 80)

ctrl_liu = df_liu[df_liu['group'] == 'control'].copy()
ctrl_liu['ses_num'] = pd.to_numeric(ctrl_liu['session'], errors='coerce').astype(int)
first_ses = ctrl_liu.groupby('subject_id')['ses_num'].min().reset_index().rename(columns={'ses_num': 'fs'})
ctrl_first = ctrl_liu.merge(first_ses, on='subject_id')
ctrl_first = ctrl_first[ctrl_first['ses_num'] == ctrl_first['fs']]

ctrl_means = ctrl_first.groupby('subject_id')['liu_distinctiveness'].mean()
group_mean = ctrl_means.mean()

ctrl_sessions = df_liu[df_liu['group'] == 'control'].groupby('subject_id')['session'].nunique()
multi_session = ctrl_sessions[ctrl_sessions >= 2].index.tolist()

best_all = (ctrl_means - group_mean).abs().idxmin()
print(f"  Best control (closest to group mean): {best_all}")
print(f"    Mean Liu = {ctrl_means[best_all]:.3f}, Group mean = {group_mean:.3f}")

candidates = ctrl_means[ctrl_means.index.isin(multi_session)]
best_long = (candidates - group_mean).abs().idxmin()
print(f"\n  Controls with 2+ sessions: {multi_session}")
print(f"  Best longitudinal control: {best_long}")
print(f"    Mean Liu = {candidates[best_long]:.3f}")

BEST_CTRL = 'sub-052'
TARGETS = ['sub-004', 'sub-007', 'sub-021', BEST_CTRL]

# ═══════════════════════════════════════════════════════════════════════════════
# PULL 2: Liu distinctiveness per category for 4 exemplar participants
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PULL 2: Liu distinctiveness per category")
print("=" * 80)

print(f"\n  {'Subject':<15} {'Face':>8} {'House':>8} {'Object':>8} {'Word':>8}  Hemi  Session")
print(f"  {'-'*70}")

for sid in TARGETS:
    sub_data = df_liu[(df_liu['subject_id'] == sid) & (df_liu['category'].isin(CATEGORIES))].copy()

    # Filter pre-surgery
    if sid in PRE_SURG:
        sub_data = sub_data[~sub_data['session'].astype(str).isin(PRE_SURG[sid])]

    if sub_data.empty:
        print(f"  {sid:<15} NO DATA")
        continue

    sub_data.loc[:, 'ses_num'] = pd.to_numeric(sub_data['session'], errors='coerce')
    first = sub_data['ses_num'].min()
    ses_data = sub_data[sub_data['ses_num'] == first]

    # For patients: intact hemisphere
    if sid in ['sub-004', 'sub-007', 'sub-021']:
        ses_intact = ses_data[ses_data['hemi_label'] == 'intact']
        if not ses_intact.empty:
            ses_data = ses_intact

    hemi = ses_data['hemi_label'].iloc[0] if len(ses_data) > 0 else '?'

    vals = {}
    for cat in CATEGORIES:
        cd = ses_data[ses_data['category'] == cat]['liu_distinctiveness']
        vals[cat] = cd.values[0] if len(cd) > 0 else np.nan

    print(f"  {sid:<15} {vals['face']:>8.3f} {vals['house']:>8.3f} {vals['object']:>8.3f} {vals['word']:>8.3f}  {hemi:<6} ses-{int(first):02d}")

# Control group 95% CI per category
print(f"\n  Control group stats (n={ctrl_first['subject_id'].nunique()}):")
print(f"  {'Category':<10} {'Mean':>8} {'2.5%':>8} {'97.5%':>8}")
print(f"  {'-'*36}")
for cat in CATEGORIES:
    subj_means = ctrl_first[ctrl_first['category'] == cat].groupby('subject_id')['liu_distinctiveness'].mean()
    lo = np.percentile(subj_means, 2.5)
    hi = np.percentile(subj_means, 97.5)
    print(f"  {cat:<10} {subj_means.mean():>8.3f} {lo:>8.3f} {hi:>8.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PULL 3: 4x4 RDM pairwise values
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PULL 3: RDM pairwise values (upper triangle)")
print("=" * 80)

cat_pairs = [('face','house'), ('face','object'), ('face','word'),
             ('house','object'), ('house','word'), ('object','word')]

def get_pairwise(sub_df, c1, c2):
    """Find the fisher_r for a category pair, checking both orderings."""
    for pair_str in [f'{c1}-{c2}', f'{c2}-{c1}']:
        rows = sub_df[sub_df['pair'] == pair_str]
        if len(rows) > 0:
            return rows['fisher_r'].mean()
    return np.nan

# Print CSV info for debugging
print(f"\n  Pairwise CSV: {df_pw.shape[0]} rows, columns: {df_pw.columns.tolist()}")
print(f"  Unique pairs: {df_pw['pair'].unique().tolist()}")

for sid in TARGETS:
    sub_pw = df_pw[df_pw['subject_id'] == sid].copy()
    if sub_pw.empty:
        print(f"\n  {sid}: no pairwise data")
        continue

    # Filter to cope_set
    if 'cope_set' in sub_pw.columns:
        sub_pw = sub_pw[sub_pw['cope_set'] == COPE_SET]

    # Filter pre-surgery sessions
    if sid in PRE_SURG:
        sub_pw = sub_pw[~sub_pw['session'].astype(str).isin(PRE_SURG[sid])]

    # First session
    sub_pw.loc[:, 'ses_num'] = pd.to_numeric(sub_pw['session'], errors='coerce')
    first = sub_pw['ses_num'].min()
    sub_pw = sub_pw[sub_pw['ses_num'] == first]

    # For patients: intact hemisphere only
    if sid in ['sub-004', 'sub-007', 'sub-021']:
        intact = sub_pw[sub_pw['hemi_label'] == 'intact']
        if not intact.empty:
            sub_pw = intact

    hemi = sub_pw['hemi_label'].iloc[0] if len(sub_pw) > 0 else '?'
    n_rows = len(sub_pw)
    print(f"\n  {sid} (ses-{int(first)}, {hemi}, {n_rows} rows):")
    for c1, c2 in cat_pairs:
        val = get_pairwise(sub_pw, c1, c2)
        if not np.isnan(val):
            print(f"    {c1}-{c2}: {val:.3f}")
        else:
            print(f"    {c1}-{c2}: NOT FOUND")

# Control group average RDM
print(f"\n  Control group average RDM:")
ctrl_pw = df_pw[df_pw['group'] == 'control'].copy()
if 'cope_set' in ctrl_pw.columns:
    ctrl_pw = ctrl_pw[ctrl_pw['cope_set'] == COPE_SET]

# First session per control
ctrl_pw.loc[:, 'ses_num'] = pd.to_numeric(ctrl_pw['session'], errors='coerce')
ctrl_first_pw = ctrl_pw.groupby('subject_id')['ses_num'].min().reset_index().rename(columns={'ses_num': 'fs'})
ctrl_pw = ctrl_pw.merge(ctrl_first_pw, on='subject_id')
ctrl_pw = ctrl_pw[ctrl_pw['ses_num'] == ctrl_pw['fs']]

print(f"  {'Pair':<16} {'Mean':>8} {'SD':>8} {'n':>5}")
print(f"  {'-'*40}")
for c1, c2 in cat_pairs:
    vals = []
    for sid_c in ctrl_pw['subject_id'].unique():
        sub_data = ctrl_pw[ctrl_pw['subject_id'] == sid_c]
        v = get_pairwise(sub_data, c1, c2)
        if not np.isnan(v):
            vals.append(v)
    if vals:
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        print(f"  {c1+'-'+c2:<16} {m:>8.3f} {sd:>8.3f} {len(vals):>5}")
    else:
        print(f"  {c1+'-'+c2:<16} NOT FOUND")

# ═══════════════════════════════════════════════════════════════════════════════
# PULL 4: Selectivity values for 4 exemplar participants
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PULL 4: Selectivity (mean_act, volume, sum_selec_norm)")
print("=" * 80)

df_sel['ses_int'] = df_sel['ses'].astype(int)

for metric, label in [('mean_act', 'Mean Activation'), ('volume', 'Active Volume'),
                       ('sum_selec_norm', 'Sum Selectivity (normalized)')]:
    print(f"\n  {label}:")
    print(f"  {'Subject':<15} {'Face':>10} {'House':>10} {'Object':>10} {'Word':>10}  Hemi  Session")
    print(f"  {'-'*72}")

    for sid in TARGETS:
        sub_data = df_sel[df_sel['sub'] == sid]
        if sub_data.empty:
            sub_data = df_sel[df_sel['sub'] == sid.replace('sub-', '')]
        if sub_data.empty:
            print(f"  {sid:<15} NO DATA")
            continue

        # Filter pre-surgery
        si = sub_info[sub_info['sub'] == sid]
        pre_ses = []
        if 'pre_post' in si.columns:
            pre_ses = si[si['pre_post'] == 'pre']['ses'].str.replace('ses-', '').astype(int).tolist()
        sub_data_post = sub_data[~sub_data['ses_int'].isin(pre_ses)]
        if sub_data_post.empty:
            sub_data_post = sub_data

        first = sub_data_post['ses_int'].min()
        ses_data = sub_data_post[sub_data_post['ses_int'] == first]

        # For patients: intact hemisphere
        if 'intact_hemi' in ses_data.columns:
            intact = ses_data['intact_hemi'].iloc[0]
            if intact != 'control':
                ses_data = ses_data[ses_data['hemi'] == intact]

        hemi = ses_data['hemi'].iloc[0] if len(ses_data) > 0 else '?'

        vals = {}
        for cat in CATEGORIES:
            cd = ses_data[ses_data['category'] == cat][metric]
            vals[cat] = cd.values[0] if len(cd) > 0 else np.nan

        if metric == 'volume':
            print(f"  {sid:<15} {vals['face']:>10.0f} {vals['house']:>10.0f} {vals['object']:>10.0f} {vals['word']:>10.0f}  {hemi:<6} ses-{int(first):02d}")
        else:
            print(f"  {sid:<15} {vals['face']:>10.2f} {vals['house']:>10.2f} {vals['object']:>10.2f} {vals['word']:>10.2f}  {hemi:<6} ses-{int(first):02d}")

    # Control stats
    ctrl_sel = df_sel[df_sel['group'] == 'control']
    first_ctrl = ctrl_sel.groupby('sub')['ses_int'].min().reset_index().rename(columns={'ses_int': 'fs'})
    ctrl_first_sel = ctrl_sel.merge(first_ctrl, on='sub')
    ctrl_first_sel = ctrl_first_sel[ctrl_first_sel['ses_int'] == ctrl_first_sel['fs']]

    print(f"\n  Control stats:")
    print(f"  {'Category':<10} {'Mean':>10} {'2.5%':>10} {'97.5%':>10}")
    print(f"  {'-'*42}")
    for cat in CATEGORIES:
        pref = {'face': 'right', 'house': 'right', 'object': 'left', 'word': 'left'}
        vals_c = ctrl_first_sel[(ctrl_first_sel['category'] == cat) &
                              (ctrl_first_sel['hemi'] == pref[cat])][metric].dropna()
        if len(vals_c) == 0:
            vals_c = ctrl_first_sel[ctrl_first_sel['category'] == cat][metric].dropna()
        lo = np.percentile(vals_c, 2.5) if len(vals_c) > 0 else np.nan
        hi = np.percentile(vals_c, 97.5) if len(vals_c) > 0 else np.nan
        m = vals_c.mean() if len(vals_c) > 0 else np.nan
        if metric == 'volume':
            print(f"  {cat:<10} {m:>10.0f} {lo:>10.0f} {hi:>10.0f}")
        else:
            print(f"  {cat:<10} {m:>10.2f} {lo:>10.2f} {hi:>10.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PULL 5: Geometry preservation for sub-021 (confirm values)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PULL 5: Geometry preservation — confirm sub-021")
print("=" * 80)

for sid in ['sub-004', 'sub-021']:
    sub_geo = df_geo[(df_geo['subject_id'] == sid) & (df_geo['hemi_label'] == 'intact')]
    if sub_geo.empty:
        sub_geo = df_geo[df_geo['subject_id'] == sid]

    print(f"\n  {sid}:")
    for cat in CATEGORIES:
        val = sub_geo[sub_geo['category'] == cat]['geometry_preservation']
        if len(val) > 0:
            print(f"    {cat}: {val.values[0]:.3f}")
        else:
            print(f"    {cat}: NOT FOUND")

# All OTC patients for reference
def fv(v):
    """Format a value, handling NaN and non-numeric gracefully."""
    if isinstance(v, (int, float)) and not np.isnan(v):
        return f"{v:.3f}"
    return "—"

print(f"\n  All OTC patients:")
otc_geo = df_geo[(df_geo['group'] == 'OTC') & (df_geo['hemi_label'] == 'intact')]
for sid in sorted(otc_geo['subject_id'].unique()):
    sub_g = otc_geo[otc_geo['subject_id'] == sid]
    vals = {}
    for cat in CATEGORIES:
        cat_data = sub_g[sub_g['category'] == cat]['geometry_preservation']
        vals[cat] = cat_data.values[0] if len(cat_data) > 0 else np.nan

    sym = np.nanmean([vals.get('house', np.nan), vals.get('object', np.nan)])
    asym = np.nanmean([vals.get('face', np.nan), vals.get('word', np.nan)])
    diff = sym - asym if not (np.isnan(sym) or np.isnan(asym)) else np.nan

    print(f"    {sid}: F={fv(vals.get('face', np.nan))}  H={fv(vals.get('house', np.nan))}  "
          f"O={fv(vals.get('object', np.nan))}  W={fv(vals.get('word', np.nan))}  "
          f"| sym={fv(sym)} asym={fv(asym)} diff={fv(diff)}")

print("\n" + "=" * 80)
print("ALL PULLS COMPLETE")
print("=" * 80)