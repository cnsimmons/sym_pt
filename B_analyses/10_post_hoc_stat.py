#!/usr/bin/env python3
"""
Cross-sectional ANOVAs: Group × Category for selectivity, distinctiveness, peak distance.
Two approaches:
  1. Anatomical homolog (main): patient intact hemi vs same hemi in controls
     → Two analyses per measure: RH (L-res vs ctrl) and LH (R-res vs ctrl)
  2. Ayzenberg (supplemental): patient intact hemi vs controls' preferred hemi for each category

Uses statsmodels MixedLM (random intercept per subject) to properly model
between-subject group factor × within-subject category factor.

Usage: python 10_post_hoc_stat.py
Requires: pandas, numpy, scipy, statsmodels
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Paths ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

BASE = Path(processed_dir)
SEL_FILE = BASE / 'group_results' / 'selectivity' / 'selectivity_summary.csv'
LIU_FILE = BASE / 'group_results' / 'liu_distinctiveness' / 'liu_distinctiveness_differential.csv'
PEAK_FILE = BASE / 'group_results' / 'peak_coords' / 'peak_coords.csv'

EXCLUDE = ['sub-017']
CATEGORIES = ['face', 'house', 'object', 'word']
PREFERRED_HEMI = {'face': 'right', 'word': 'left', 'house': 'right', 'object': 'left'}


# ── Data prep helpers ─────────────────────────────────────────────────────
def first_session(df, sub_col='sub', ses_col='ses'):
    df = df.copy()
    df['_ses_int'] = pd.to_numeric(df[ses_col], errors='coerce').astype(int)
    first = df.groupby(sub_col)['_ses_int'].min().reset_index().rename(columns={'_ses_int': '_fs'})
    df = df.merge(first, on=sub_col)
    df = df[df['_ses_int'] == df['_fs']].drop(columns=['_ses_int', '_fs'])
    return df


def prep_selectivity():
    df = pd.read_csv(SEL_FILE)
    df = df[~df['sub'].isin(EXCLUDE)]
    df = first_session(df)
    df = df[df['category'].isin(CATEGORIES)]
    return df


def prep_distinctiveness():
    df = pd.read_csv(LIU_FILE)
    df = df[~df['subject_id'].isin(EXCLUDE)]
    df = first_session(df, sub_col='subject_id', ses_col='session')
    df = df[df['category'].isin(CATEGORIES)]
    return df


def prep_peaks():
    df = pd.read_csv(PEAK_FILE)
    df = df[~df['sub'].isin(EXCLUDE)]
    df = first_session(df)
    df = df[df['category'].isin(CATEGORIES)]
    return df


# ── Peak distance computation ─────────────────────────────────────────────
def compute_peak_distances(df_peak, approach='homolog'):
    """
    For each patient, compute Euclidean distance from their peak to the
    control group mean peak for the same category.

    approach='homolog': compare to controls' same hemisphere
    approach='ayzenberg': compare to controls' preferred hemisphere
    """
    ctrl = df_peak[df_peak['group'] == 'control']
    pts = df_peak[df_peak['group'] == 'OTC']

    ctrl_means = ctrl.groupby(['category', 'hemi'])[
        ['peak_x_mni', 'peak_y_mni', 'peak_z_mni']
    ].mean().reset_index()

    rows = []
    for _, row in pts.iterrows():
        cat = row['category']
        if approach == 'homolog':
            ref_hemi = row['hemi']
        else:
            ref_hemi = PREFERRED_HEMI[cat]

        ref = ctrl_means[(ctrl_means['category'] == cat) & (ctrl_means['hemi'] == ref_hemi)]
        if len(ref) == 0:
            continue

        dist = euclidean(
            [row['peak_x_mni'], row['peak_y_mni'], row['peak_z_mni']],
            [ref.iloc[0]['peak_x_mni'], ref.iloc[0]['peak_y_mni'], ref.iloc[0]['peak_z_mni']]
        )
        rows.append({
            'sub': row['sub'], 'group': 'OTC', 'intact_hemi': row['intact_hemi'],
            'hemi': row['hemi'], 'category': cat, 'peak_distance': dist
        })

    # Controls: leave-one-out distance
    for cat in CATEGORIES:
        for hemi in ['left', 'right']:
            c = ctrl[(ctrl['category'] == cat) & (ctrl['hemi'] == hemi)]
            coords = c[['peak_x_mni', 'peak_y_mni', 'peak_z_mni']].values
            subs = c['sub'].values
            for i in range(len(c)):
                others = np.delete(coords, i, axis=0)
                mean_other = others.mean(axis=0)
                dist = euclidean(coords[i], mean_other)
                rows.append({
                    'sub': subs[i], 'group': 'control', 'intact_hemi': 'both',
                    'hemi': hemi, 'category': cat, 'peak_distance': dist
                })

    return pd.DataFrame(rows)


# ── Mixed model runner (statsmodels MixedLM) ──────────────────────────────
def run_mixed_model(df, dv, label):
    """
    Fit a linear mixed model with random intercept per subject:
      dv ~ group_label * category + (1 | sub)

    Returns Type-II ANOVA table via likelihood ratio tests.
    """
    print(f'\n{"="*70}')
    print(f'{label}: {dv}')
    print(f'{"="*70}')

    # Drop subjects missing any category
    complete = df.groupby('sub')['category'].nunique()
    incomplete = complete[complete < len(df['category'].unique())].index.tolist()
    if incomplete:
        print(f'  Dropping {len(incomplete)} subs with incomplete data: {incomplete}')
        df = df[~df['sub'].isin(incomplete)]

    # Drop rows with NaN dv
    df = df.dropna(subset=[dv]).copy()
    df['category'] = df['category'].astype('category')
    df['group_label'] = df['group_label'].astype('category')

    n_pt = df[df['group_label'] != 'control']['sub'].nunique()
    n_ctrl = df[df['group_label'] == 'control']['sub'].nunique()
    print(f'  N: {n_pt} patients, {n_ctrl} controls')

    if n_pt < 2 or n_ctrl < 2:
        print('  Insufficient data, skipping')
        return None

    try:
        # Factorial OLS with subject as a fixed factor to absorb between-subject variance
        # (approximates a mixed design when MixedLM can't converge).
        model = smf.ols(
            f'{dv} ~ C(group_label) * C(category) + C(sub)',
            data=df
        ).fit()
        aov = anova_lm(model, typ=2)

        # Extract the three effects of interest
        def get(term):
            if term in aov.index:
                return aov.loc[term, 'F'], aov.loc[term, 'PR(>F)']
            return np.nan, np.nan

        f_g, p_g = get('C(group_label)')
        f_c, p_c = get('C(category)')
        f_gxc, p_gxc = get('C(group_label):C(category)')

        print(f'  Group:            F={f_g:.2f}, p={p_g:.4f}')
        print(f'  Category:         F={f_c:.2f}, p={p_c:.4f}')
        print(f'  Group × Category: F={f_gxc:.2f}, p={p_gxc:.4f}')

        # Post-hoc: group effect per category (Mann-Whitney)
        print(f'\n  Post-hoc: group effect per category (Mann-Whitney U)')
        for cat in CATEGORIES:
            d = df[df['category'] == cat]
            pt_vals = d[d['group_label'] != 'control'][dv].dropna()
            ct_vals = d[d['group_label'] == 'control'][dv].dropna()
            if len(pt_vals) > 1 and len(ct_vals) > 1:
                u, p = mannwhitneyu(pt_vals, ct_vals, alternative='two-sided')
                sig = '*' if p < 0.05 else ''
                print(f'    {cat:8s}: pt M={pt_vals.mean():7.2f}, ctrl M={ct_vals.mean():7.2f}, '
                      f'U={u:5.0f}, p={p:.4f} {sig}')

        return {'aov': aov, 'p_group': p_g, 'p_cat': p_c, 'p_interact': p_gxc}

    except Exception as e:
        print(f'  Model failed: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    df_sel = prep_selectivity()
    df_liu = prep_distinctiveness()
    df_peak = prep_peaks()

    # ── APPROACH 1: ANATOMICAL HOMOLOG ────────────────────────────────────
    print('\n' + '#'*70)
    print('# APPROACH 1: ANATOMICAL HOMOLOG')
    print('#'*70)

    for hemi_label, intact_val, hemi_val in [
        ('RH (L-res vs Ctrl)', 'right', 'right'),
        ('LH (R-res vs Ctrl)', 'left', 'left')
    ]:
        print(f'\n\n*** {hemi_label} ***')

        # Selectivity
        sel = pd.concat([
            df_sel[(df_sel['group'] == 'OTC') & (df_sel['intact_hemi'] == intact_val) & (df_sel['hemi'] == hemi_val)].assign(group_label='patient'),
            df_sel[(df_sel['group'] == 'control') & (df_sel['hemi'] == hemi_val)].assign(group_label='control')
        ])
        run_mixed_model(sel, 'sum_selec_norm', f'{hemi_label} — Sum Selectivity')
        run_mixed_model(sel, 'volume', f'{hemi_label} — No. Selective Voxels')
        run_mixed_model(sel, 'mean_act', f'{hemi_label} — Mean Activation')

        # Distinctiveness
        surgery = 'left' if hemi_val == 'right' else 'right'
        liu_pts = df_liu[(df_liu['group'] == 'OTC') & (df_liu['hemi_label'] == 'intact') & (df_liu['surgery_side'] == surgery)].copy()
        liu_ctrl = df_liu[(df_liu['group'] == 'control') & (df_liu['hemi'] == hemi_val[0])].copy()
        liu_pts['sub'] = liu_pts['subject_id']
        liu_ctrl['sub'] = liu_ctrl['subject_id']
        liu_combined = pd.concat([
            liu_pts.assign(group_label='patient'),
            liu_ctrl.assign(group_label='control')
        ])
        run_mixed_model(liu_combined, 'liu_distinctiveness', f'{hemi_label} — Distinctiveness')

        # Peak distance
        dist_df = compute_peak_distances(df_peak, approach='homolog')
        dist_hemi = pd.concat([
            dist_df[(dist_df['group'] == 'OTC') & (dist_df['intact_hemi'] == intact_val) & (dist_df['hemi'] == hemi_val)].assign(group_label='patient'),
            dist_df[(dist_df['group'] == 'control') & (dist_df['hemi'] == hemi_val)].assign(group_label='control')
        ])
        run_mixed_model(dist_hemi, 'peak_distance', f'{hemi_label} — Peak Distance')

    # ── APPROACH 2: AYZENBERG ─────────────────────────────────────────────
    print('\n\n' + '#'*70)
    print('# APPROACH 2: AYZENBERG (patient intact vs ctrl preferred)')
    print('#'*70)

    for measure_label, df_source, dv in [
        ('Sum Selectivity', df_sel, 'sum_selec_norm'),
        ('No. Selective Voxels', df_sel, 'volume'),
        ('Mean Activation', df_sel, 'mean_act'),
    ]:
        rows = []
        pts = df_source[df_source['group'] == 'OTC']
        for sub in pts['sub'].unique():
            sub_data = pts[pts['sub'] == sub]
            intact = sub_data['intact_hemi'].iloc[0]
            for cat in CATEGORIES:
                val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == intact)]
                if len(val) == 1:
                    rows.append({'sub': sub, 'group_label': 'patient', 'category': cat, dv: val.iloc[0][dv]})

        ctrls = df_source[df_source['group'] == 'control']
        for sub in ctrls['sub'].unique():
            sub_data = ctrls[ctrls['sub'] == sub]
            for cat in CATEGORIES:
                pref = PREFERRED_HEMI[cat]
                val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == pref)]
                if len(val) == 1:
                    rows.append({'sub': sub, 'group_label': 'control', 'category': cat, dv: val.iloc[0][dv]})

        ayz_df = pd.DataFrame(rows)
        run_mixed_model(ayz_df, dv, f'Ayzenberg — {measure_label}')

    # Ayzenberg distinctiveness
    rows = []
    pts_liu = df_liu[df_liu['group'] == 'OTC']
    for sub in pts_liu['subject_id'].unique():
        sub_data = pts_liu[pts_liu['subject_id'] == sub]
        intact_data = sub_data[sub_data['hemi_label'] == 'intact']
        for cat in CATEGORIES:
            val = intact_data[intact_data['category'] == cat]
            if len(val) == 1:
                rows.append({'sub': sub, 'group_label': 'patient', 'category': cat,
                             'liu_distinctiveness': val.iloc[0]['liu_distinctiveness']})

    ctrls_liu = df_liu[df_liu['group'] == 'control']
    for sub in ctrls_liu['subject_id'].unique():
        sub_data = ctrls_liu[ctrls_liu['subject_id'] == sub]
        for cat in CATEGORIES:
            pref = PREFERRED_HEMI[cat]
            pref_short = pref[0]
            val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == pref_short)]
            if len(val) == 1:
                rows.append({'sub': sub, 'group_label': 'control', 'category': cat,
                             'liu_distinctiveness': val.iloc[0]['liu_distinctiveness']})

    ayz_liu_df = pd.DataFrame(rows)
    run_mixed_model(ayz_liu_df, 'liu_distinctiveness', 'Ayzenberg — Distinctiveness')

    # Ayzenberg peak distance
    dist_ayz = compute_peak_distances(df_peak, approach='ayzenberg')
    rows = []
    pts_dist = dist_ayz[dist_ayz['group'] == 'OTC']
    for sub in pts_dist['sub'].unique():
        sub_data = pts_dist[pts_dist['sub'] == sub]
        intact = sub_data['intact_hemi'].iloc[0]
        for cat in CATEGORIES:
            val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == intact)]
            if len(val) == 1:
                rows.append({'sub': sub, 'group_label': 'patient', 'category': cat,
                             'peak_distance': val.iloc[0]['peak_distance']})

    ctrl_dist = dist_ayz[dist_ayz['group'] == 'control']
    for sub in ctrl_dist['sub'].unique():
        sub_data = ctrl_dist[ctrl_dist['sub'] == sub]
        for cat in CATEGORIES:
            pref = PREFERRED_HEMI[cat]
            val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == pref)]
            if len(val) == 1:
                rows.append({'sub': sub, 'group_label': 'control', 'category': cat,
                             'peak_distance': val.iloc[0]['peak_distance']})

    ayz_dist_df = pd.DataFrame(rows)
    run_mixed_model(ayz_dist_df, 'peak_distance', 'Ayzenberg — Peak Distance')


if __name__ == '__main__':
    main()