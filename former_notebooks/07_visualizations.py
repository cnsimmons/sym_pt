#!/usr/bin/env python3
"""
07_plot_results.py - Visualize selectivity and peak coordinate results

Generates key figures following Ayzenberg et al. (2023):
  1. Violin plots: bootstrapped control distributions + patient data (Fig 4-7C)
  2. Summary table: patient percentiles with significance markers (Table 2)
  3. Peak coordinate scatter: patient peaks on control distribution

Reads from:
  - group_results/selectivity/selectivity_summary.csv
  - group_results/selectivity/resamples/*_resamples.csv
  - group_results/selectivity/patient_percentiles.csv
  - group_results/peak_coords/peak_coords.csv
  - group_results/peak_coords/patient_distances.csv

Usage:
  python 07_plot_results.py
  python 07_plot_results.py --suffix _broad
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, _load_csv

# ── Configuration ────────────────────────────────────────────────────────────

CATEGORIES = ['face', 'word', 'house', 'object']
HEMIS = ['left', 'right']

# Colors
CTRL_COLOR = '#808080'
LEFT_COLOR = '#ee7183'
RIGHT_COLOR = '#7398af'

ALPHA = 0.05  # two-sided 95% CI


# ── Helper: get subject code from sub_info ───────────────────────────────────

def get_code(sub_clean):
    """Get patient code from sub_info CSV."""
    df = _load_csv()
    rows = df[df['sub_clean'] == sub_clean.replace('sub-', '')]
    if rows.empty:
        return sub_clean
    code = rows.iloc[0].get('code', '')
    return code if pd.notna(code) and code != '' else sub_clean


# ── Plot 1: Selectivity violin plots ────────────────────────────────────────

def plot_selectivity_violins(suffix=''):
    """
    For each category × hemisphere, plot:
      - Violin of bootstrapped control mean distribution
      - Individual control data points
      - Individual patient data points with labels
    """
    sel_file = f'{processed_dir}/group_results/selectivity/selectivity_summary{suffix}.csv'
    if not os.path.exists(sel_file):
        print(f'Selectivity summary not found: {sel_file}')
        return

    sel_df = pd.read_csv(sel_file)

    # Separate controls and patients, first session only
    controls = sel_df[sel_df['group'] == 'control'].copy()
    patients = sel_df[sel_df['group'] != 'control'].copy()

    # First session per subject
    for df in [controls, patients]:
        df['ses_int'] = df['ses'].astype(int)
        first = df.groupby('sub')['ses_int'].min().reset_index()
        first.columns = ['sub', 'first_ses']
        idx = df.merge(first, on='sub')
        df.drop(df.index, inplace=True)
        df_filtered = idx[idx['ses_int'] == idx['first_ses']].drop(columns=['first_ses'])
        for col in df_filtered.columns:
            if col in df.columns or col == 'ses_int':
                continue
        # Rebuild
        controls = sel_df[sel_df['group'] == 'control'].copy()
        patients = sel_df[sel_df['group'] != 'control'].copy()
        break

    # Simpler: just use first session
    controls['ses_int'] = controls['ses'].astype(int)
    first_ctrl = controls.groupby('sub')['ses_int'].min().reset_index()
    first_ctrl.columns = ['sub', 'fs']
    controls = controls.merge(first_ctrl, on='sub')
    controls = controls[controls['ses_int'] == controls['fs']]

    patients['ses_int'] = patients['ses'].astype(int)
    first_pt = patients.groupby('sub')['ses_int'].min().reset_index()
    first_pt.columns = ['sub', 'fs']
    patients = patients.merge(first_pt, on='sub')
    patients = patients[patients['ses_int'] == patients['fs']]

    # Load resamples for violin data
    metric = 'sum_selec_norm'
    resample_file = f'{processed_dir}/group_results/selectivity/resamples/{metric}_resamples{suffix}.csv'

    has_resamples = os.path.exists(resample_file)
    if has_resamples:
        resamples = pd.read_csv(resample_file)

    out_dir = f'{processed_dir}/group_results/figures{suffix}'
    os.makedirs(out_dir, exist_ok=True)

    for cat in CATEGORIES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        fig.suptitle(f'{cat.capitalize()} - Summed Selectivity (Normalized)',
                     fontsize=14, fontweight='bold')

        for i, hemi in enumerate(HEMIS):
            ax = axes[i]
            col_name = f'{cat}_{hemi}'

            # Violin from resamples
            if has_resamples and col_name in resamples.columns:
                boot_data = resamples[col_name].dropna()
                parts = ax.violinplot(boot_data, positions=[0], showmeans=True,
                                       showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(CTRL_COLOR)
                    pc.set_alpha(0.3)

            # Control individual points
            ctrl_sub = controls[(controls['category'] == cat) &
                                (controls['hemi'] == hemi)]
            if len(ctrl_sub) > 0:
                ctrl_vals = ctrl_sub[metric].values
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(ctrl_vals))
                ax.scatter(jitter, ctrl_vals, c=CTRL_COLOR, s=30, alpha=0.6,
                          zorder=3, label='Controls')

            # Patient points with labels
            pt_sub = patients[(patients['category'] == cat) &
                              (patients['hemi'] == hemi)]
            for _, row in pt_sub.iterrows():
                val = row[metric]
                color = LEFT_COLOR if row['intact_hemi'] == 'left' else RIGHT_COLOR
                code = get_code(row['sub'])
                jit = np.random.default_rng(hash(row['sub']) % 2**31).uniform(-0.15, 0.15)
                ax.scatter(jit, val, c=color, s=60, zorder=5, edgecolors='black')
                ax.annotate(code, (jit, val), fontsize=7, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                    ec='black', alpha=0.7))

            ax.set_title(f'{hemi.capitalize()} Hemisphere', fontsize=12)
            ax.set_xticks([])
            ax.set_ylabel('Summed Selectivity (norm)' if i == 0 else '')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Legend
        patches = [
            mpatches.Patch(color=CTRL_COLOR, alpha=0.6, label='Controls'),
            mpatches.Patch(color=LEFT_COLOR, label='Intact Left'),
            mpatches.Patch(color=RIGHT_COLOR, label='Intact Right'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        out_path = f'{out_dir}/selectivity_{cat}{suffix}.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {out_path}')
        plt.close()


# ── Plot 2: Summary table ───────────────────────────────────────────────────

def plot_summary_table(suffix=''):
    """
    Print and save a summary table mimicking Ayzenberg Table 2.
    Shows whether each patient's sum_selec_norm is below control 95% CI.
    """
    pt_file = f'{processed_dir}/group_results/selectivity/patient_percentiles{suffix}.csv'
    if not os.path.exists(pt_file):
        print(f'Patient percentiles not found: {pt_file}')
        return

    pt_df = pd.read_csv(pt_file)
    ssn = pt_df[pt_df['metric'] == 'sum_selec_norm']

    print('\n' + '=' * 80)
    print(f'SELECTIVITY SUMMARY TABLE (sum_selec_norm){" [" + suffix[1:] + "]" if suffix else ""}')
    print('* = below 95% CI, - = within/above')
    print('=' * 80)

    header = f'{"Patient":<12} {"Group":<8} {"Intact":<8}'
    for cat in CATEGORIES:
        header += f' {cat.capitalize():<12}'
    print(header)
    print('-' * 80)

    for sub in sorted(ssn['sub'].unique()):
        sub_data = ssn[ssn['sub'] == sub]
        intact = sub_data['intact_hemi'].iloc[0]
        group = sub_data['group'].iloc[0]
        code = get_code(sub)

        line = f'{code:<12} {group:<8} {intact:<8}'
        for cat in CATEGORIES:
            row = sub_data[sub_data['category'] == cat]
            if len(row) == 0:
                line += f' {"N/A":<12}'
            else:
                r = row.iloc[0]
                marker = '*' if r['below_ci'] else '-'
                line += f' {r["percentile"]:>5.1f}%{marker:<5}'
        print(line)

    print('=' * 80)


# ── Plot 3: Peak coordinate distances ───────────────────────────────────────

def plot_peak_distances(suffix=''):
    """
    For each category × hemisphere, show patient peak distances
    vs bootstrapped control distribution.
    """
    dist_file = f'{processed_dir}/group_results/peak_coords/patient_distances{suffix}.csv'
    resample_file = f'{processed_dir}/group_results/peak_coords/distance_resamples{suffix}.csv'

    if not os.path.exists(dist_file) or not os.path.exists(resample_file):
        print('Peak coordinate results not found. Run 05_calc_peak_coords.py first.')
        return

    dist_df = pd.read_csv(dist_file)
    resamples = pd.read_csv(resample_file)

    out_dir = f'{processed_dir}/group_results/figures{suffix}'
    os.makedirs(out_dir, exist_ok=True)

    for cat in CATEGORIES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.suptitle(f'{cat.capitalize()} - Peak Distance from Controls (mm)',
                     fontsize=14, fontweight='bold')

        for i, hemi in enumerate(HEMIS):
            ax = axes[i]
            col_name = f'{cat}_{hemi}'

            # Bootstrap distribution
            if col_name in resamples.columns:
                boot = resamples[col_name].dropna()
                ax.hist(boot, bins=50, color=CTRL_COLOR, alpha=0.4,
                       density=True, label='Control bootstrap')

                # 97.5th percentile line
                ci_97 = np.percentile(boot, 97.5)
                ax.axvline(ci_97, color='red', linestyle='--', alpha=0.7,
                          label=f'97.5th pctl ({ci_97:.1f}mm)')

            # Patient distances
            pt_sub = dist_df[(dist_df['category'] == cat) &
                             (dist_df['hemi'] == hemi)]
            for _, row in pt_sub.iterrows():
                color = LEFT_COLOR if row['intact_hemi'] == 'left' else RIGHT_COLOR
                code = get_code(row['sub'])
                ax.axvline(row['mean_dist_mm'], color=color, linewidth=2, alpha=0.8)
                ax.text(row['mean_dist_mm'], ax.get_ylim()[1] * 0.9, code,
                       fontsize=7, ha='center', rotation=90,
                       bbox=dict(fc='white', ec=color, alpha=0.7, boxstyle='round'))

            ax.set_title(f'{hemi.capitalize()} Hemisphere')
            ax.set_xlabel('Mean Distance (mm)')
            if i == 0:
                ax.set_ylabel('Density')
            ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        out_path = f'{out_dir}/peak_distance_{cat}{suffix}.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {out_path}')
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--suffix', type=str, default='',
                        help='File suffix (e.g., _broad, _wholehemi)')
    args = parser.parse_args()

    print('=' * 60)
    print(f'GENERATING RESULT FIGURES{" [" + args.suffix[1:] + "]" if args.suffix else ""}')
    print('=' * 60)

    # 1. Selectivity violin plots
    print('\n--- Selectivity Violin Plots ---')
    plot_selectivity_violins(args.suffix)

    # 2. Summary table
    print('\n--- Summary Table ---')
    plot_summary_table(args.suffix)

    # 3. Peak distance plots
    print('\n--- Peak Distance Plots ---')
    plot_peak_distances(args.suffix)

    print('\nDone!')


if __name__ == '__main__':
    main()