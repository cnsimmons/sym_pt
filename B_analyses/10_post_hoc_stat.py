#!/usr/bin/env python3
"""
Cross-sectional ANOVAs: Group × Category for selectivity, distinctiveness, peak distance.
Two approaches:
  1. Anatomical homolog (main): patient intact hemi vs same hemi in controls
     → Two ANOVAs per measure: RH (L-res vs ctrl) and LH (R-res vs ctrl)
  2. Ayzenberg (supplemental): patient intact hemi vs controls' preferred hemi for each category
     → One ANOVA per measure, but comparison hemisphere varies by category

Usage: python cross_sectional_anovas.py
Requires: pingouin, pandas, numpy, scipy
"""

import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from scipy.spatial.distance import euclidean

# ── Paths (adjust if needed) ──────────────────────────────────────────────
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

    # Control mean peaks per category × hemi
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

    # Controls: distance from each control to leave-one-out mean
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


# ── Mixed ANOVA runner ────────────────────────────────────────────────────
def run_mixed_anova(df, dv, label):
    """Run group(2) × category(4) mixed ANOVA. Returns pingouin result."""
    print(f'\n{"="*70}')
    print(f'{label}: {dv}')
    print(f'{"="*70}')

    # Check for missing data
    complete = df.groupby('sub')['category'].nunique()
    incomplete = complete[complete < 4].index.tolist()
    if incomplete:
        print(f'  Dropping {len(incomplete)} subs with incomplete data: {incomplete}')
        df = df[~df['sub'].isin(incomplete)]

    n_pt = df[df['group_label'] != 'control']['sub'].nunique()
    n_ctrl = df[df['group_label'] == 'control']['sub'].nunique()
    print(f'  N: {n_pt} patients, {n_ctrl} controls')

    aov = None
    try:
        aov = pg.mixed_anova(
            data=df, dv=dv, within='category', between='group_label',
            subject='sub', correction=True
        )
        print(aov.to_string(index=False))

        # Post-hoc: group effect per category
        print(f'\n  Post-hoc: group effect per category')
        for cat in CATEGORIES:
            d = df[df['category'] == cat]
            pt_vals = d[d['group_label'] != 'control'][dv].dropna()
            ct_vals = d[d['group_label'] == 'control'][dv].dropna()
            if len(pt_vals) > 1 and len(ct_vals) > 1:
                from scipy.stats import mannwhitneyu
                u, p = mannwhitneyu(pt_vals, ct_vals, alternative='two-sided')
                print(f'    {cat}: pt M={pt_vals.mean():.1f}, ctrl M={ct_vals.mean():.1f}, U={u:.0f}, p={p:.4f}')
    except Exception as e:
        print(f'  ANOVA failed: {e}')

    return aov


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

        # -- Selectivity: sum_selec_norm --
        sel = pd.concat([
            df_sel[(df_sel['group'] == 'OTC') & (df_sel['intact_hemi'] == intact_val) & (df_sel['hemi'] == hemi_val)].assign(group_label='patient'),
            df_sel[(df_sel['group'] == 'control') & (df_sel['hemi'] == hemi_val)].assign(group_label='control')
        ])
        run_mixed_anova(sel, 'sum_selec_norm', f'{hemi_label} — Sum Selectivity')
        run_mixed_anova(sel, 'volume', f'{hemi_label} — No. Selective Voxels')
        run_mixed_anova(sel, 'mean_act', f'{hemi_label} — Mean Activation')

        # -- Distinctiveness --
        # liu file uses 'l'/'r' for hemi, surgery_side='left' means L-res (intact R)
        if hemi_val == 'right':
            surgery = 'left'  # L-res → intact right
        else:
            surgery = 'right'  # R-res → intact left

        liu_pts = df_liu[(df_liu['group'] == 'OTC') & (df_liu['hemi_label'] == 'intact') & (df_liu['surgery_side'] == surgery)]
        liu_ctrl = df_liu[(df_liu['group'] == 'control') & (df_liu['hemi'] == hemi_val[0])]
        # Standardize sub column
        liu_pts = liu_pts.copy()
        liu_ctrl = liu_ctrl.copy()
        liu_pts['sub'] = liu_pts['subject_id']
        liu_ctrl['sub'] = liu_ctrl['subject_id']
        liu_combined = pd.concat([
            liu_pts.assign(group_label='patient'),
            liu_ctrl.assign(group_label='control')
        ])
        run_mixed_anova(liu_combined, 'liu_distinctiveness', f'{hemi_label} — Distinctiveness')

        # -- Peak distance (homolog) --
        dist_df = compute_peak_distances(df_peak, approach='homolog')
        dist_hemi = pd.concat([
            dist_df[(dist_df['group'] == 'OTC') & (dist_df['intact_hemi'] == intact_val) & (dist_df['hemi'] == hemi_val)].assign(group_label='patient'),
            dist_df[(dist_df['group'] == 'control') & (dist_df['hemi'] == hemi_val)].assign(group_label='control')
        ])
        run_mixed_anova(dist_hemi, 'peak_distance', f'{hemi_label} — Peak Distance')

    # ── APPROACH 2: AYZENBERG (supplemental) ──────────────────────────────
    print('\n\n' + '#'*70)
    print('# APPROACH 2: AYZENBERG (patient intact vs ctrl preferred)')
    print('#'*70)

    # For each category, patient's intact hemi vs controls' preferred hemi
    # This mixes hemisphere across categories, so we build the df manually
    for measure_label, df_source, dv, sub_col, hemi_col in [
        ('Sum Selectivity', df_sel, 'sum_selec_norm', 'sub', 'hemi'),
        ('No. Selective Voxels', df_sel, 'volume', 'sub', 'hemi'),
        ('Mean Activation', df_sel, 'mean_act', 'sub', 'hemi'),
    ]:
        rows = []
        # Patients: intact hemi value per category
        pts = df_source[df_source['group'] == 'OTC']
        for sub in pts['sub'].unique():
            sub_data = pts[pts['sub'] == sub]
            intact = sub_data['intact_hemi'].iloc[0]
            for cat in CATEGORIES:
                val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == intact)]
                if len(val) == 1:
                    rows.append({'sub': sub, 'group_label': 'patient', 'category': cat, dv: val.iloc[0][dv]})

        # Controls: preferred hemi per category
        ctrls = df_source[df_source['group'] == 'control']
        for sub in ctrls['sub'].unique():
            sub_data = ctrls[ctrls['sub'] == sub]
            for cat in CATEGORIES:
                pref = PREFERRED_HEMI[cat]
                val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == pref)]
                if len(val) == 1:
                    rows.append({'sub': sub, 'group_label': 'control', 'category': cat, dv: val.iloc[0][dv]})

        ayz_df = pd.DataFrame(rows)
        run_mixed_anova(ayz_df, dv, f'Ayzenberg — {measure_label}')

    # Ayzenberg distinctiveness
    rows = []
    pts_liu = df_liu[df_liu['group'] == 'OTC']
    for sub in pts_liu['subject_id'].unique():
        sub_data = pts_liu[pts_liu['subject_id'] == sub]
        if len(sub_data) == 0:
            continue
        intact = sub_data['hemi_label'].iloc[0]  # will use hemi_label == 'intact'
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
            pref_short = pref[0]  # 'l' or 'r'
            val = sub_data[(sub_data['category'] == cat) & (sub_data['hemi'] == pref_short)]
            if len(val) == 1:
                rows.append({'sub': sub, 'group_label': 'control', 'category': cat,
                             'liu_distinctiveness': val.iloc[0]['liu_distinctiveness']})

    ayz_liu_df = pd.DataFrame(rows)
    run_mixed_anova(ayz_liu_df, 'liu_distinctiveness', 'Ayzenberg — Distinctiveness')

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
    run_mixed_anova(ayz_dist_df, 'peak_distance', 'Ayzenberg — Peak Distance')


if __name__ == '__main__':
    main()
    