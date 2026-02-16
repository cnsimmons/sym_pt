#!/usr/bin/env python3
"""
03_resample_selectivity.py - Bootstrap resample control data for patient comparisons

Following Ayzenberg et al. (2023):
  - For each category × hemisphere, bootstrap sample n_subs controls
  - Compute mean of each selectivity metric per resample
  - Repeat 10,000 times to create null distribution
  - Each patient's value is then compared to the 95% CI of this distribution

Uses first post-surgical session per subject for controls.
Outputs one CSV per metric with columns for each category_hemisphere combination.

Usage:
  python 03_resample_selectivity.py
  python 03_resample_selectivity.py --n-subs 4 --iter 10000
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

# ── Configuration ────────────────────────────────────────────────────────────

N_SUBS = 4          # Controls per resample (Ayzenberg used 4)
ITER = 10000         # Number of bootstrap iterations
METRICS = ['mean_act', 'volume', 'sum_selec', 'sum_selec_norm']
ALPHA = 0.05         # For two-sided 95% CI: 2.5th and 97.5th percentiles

# ── Core Functions ───────────────────────────────────────────────────────────

def load_and_prepare(selectivity_file):
    """
    Load selectivity summary CSV.
    For controls with multiple sessions, use only the first session.
    Returns separate control and patient DataFrames.
    """
    df = pd.read_csv(selectivity_file)
    
    # Split by group
    controls = df[df['group'] == 'control'].copy()
    patients = df[df['group'] != 'control'].copy()
    
    # For controls: keep only first session per subject
    controls['ses_int'] = controls['ses'].astype(int)
    first_ses = controls.groupby('sub')['ses_int'].min().reset_index()
    first_ses.columns = ['sub', 'first_ses']
    controls = controls.merge(first_ses, on='sub')
    controls = controls[controls['ses_int'] == controls['first_ses']]
    controls = controls.drop(columns=['ses_int', 'first_ses'])
    
    # For patients: keep only first post-surgical session per subject
    patients['ses_int'] = patients['ses'].astype(int)
    first_ses_pt = patients.groupby('sub')['ses_int'].min().reset_index()
    first_ses_pt.columns = ['sub', 'first_ses']
    patients = patients.merge(first_ses_pt, on='sub')
    patients = patients[patients['ses_int'] == patients['first_ses']]
    patients = patients.drop(columns=['ses_int', 'first_ses'])
    
    print(f'Controls: {controls["sub"].nunique()} subjects, '
          f'{len(controls)} rows')
    print(f'Patients: {patients["sub"].nunique()} subjects, '
          f'{len(patients)} rows')
    
    return controls, patients


def resample_selectivity(controls, n_subs=N_SUBS, n_iter=ITER):
    """
    Bootstrap resample control data.
    
    For each category × hemisphere combination:
      - Sample n_subs controls with replacement
      - Compute mean of each metric
      - Repeat n_iter times
    
    Returns dict of DataFrames, one per metric.
    """
    categories = controls['category'].unique()
    hemis = controls['hemi'].unique()
    
    # All category_hemi combos
    combos = []
    for cat in sorted(categories):
        for hemi in sorted(hemis):
            combos.append(f'{cat}_{hemi}')
    
    # Initialize result DataFrames
    resample_dfs = {metric: pd.DataFrame(columns=combos) for metric in METRICS}
    
    print(f'\nResampling: {n_iter} iterations, {n_subs} controls per sample')
    print(f'Combinations: {combos}')
    
    for cat in sorted(categories):
        for hemi in sorted(hemis):
            col_name = f'{cat}_{hemi}'
            
            # Get control data for this category × hemisphere
            subset = controls[
                (controls['category'] == cat) & 
                (controls['hemi'] == hemi)
            ]
            
            if len(subset) < n_subs:
                print(f'  WARNING: {col_name} has only {len(subset)} controls '
                      f'(need {n_subs}), skipping')
                continue
            
            print(f'  {col_name}: {len(subset)} control values')
            
            # Bootstrap
            for metric in METRICS:
                vals = subset[metric].dropna().values
                
                if len(vals) < n_subs:
                    continue
                
                # Vectorized bootstrap: sample indices, compute means
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(len(vals), size=(n_iter, n_subs), replace=True)
                boot_means = vals[idx].mean(axis=1)
                
                resample_dfs[metric][col_name] = boot_means
    
    return resample_dfs


def compute_patient_percentiles(patients, resample_dfs):
    """
    For each patient, compute where their value falls in the 
    bootstrapped control distribution (as a percentile).
    """
    results = []
    
    for _, row in patients.iterrows():
        col_name = f'{row["category"]}_{row["hemi"]}'
        
        for metric in METRICS:
            if col_name not in resample_dfs[metric].columns:
                continue
            
            boot_dist = resample_dfs[metric][col_name].dropna().values
            patient_val = row[metric]
            
            if np.isnan(patient_val) or len(boot_dist) == 0:
                percentile = np.nan
                below_ci = np.nan
            else:
                percentile = float(np.mean(boot_dist <= patient_val) * 100)
                # Two-sided: below if < 2.5th percentile
                ci_low = np.percentile(boot_dist, (ALPHA / 2) * 100)
                ci_high = np.percentile(boot_dist, (1 - ALPHA / 2) * 100)
                below_ci = patient_val < ci_low
            
            results.append({
                'sub': row['sub'],
                'ses': row['ses'],
                'group': row['group'],
                'intact_hemi': row['intact_hemi'],
                'hemi': row['hemi'],
                'category': row['category'],
                'metric': metric,
                'value': patient_val,
                'percentile': percentile,
                'below_ci': below_ci,
                'ci_low': ci_low if not np.isnan(patient_val) else np.nan,
                'ci_high': ci_high if not np.isnan(patient_val) else np.nan,
            })
    
    return pd.DataFrame(results)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Bootstrap resample control selectivity data')
    parser.add_argument('--n-subs', type=int, default=N_SUBS,
                        help=f'Controls per resample (default: {N_SUBS})')
    parser.add_argument('--iter', type=int, default=ITER,
                        help=f'Bootstrap iterations (default: {ITER})')
    parser.add_argument('--suffix', type=str, default='',
                        help='Input/output file suffix')
    args = parser.parse_args()
    
    print('=' * 60)
    print('BOOTSTRAP RESAMPLE SELECTIVITY DATA')
    print('=' * 60)
    print(f'N controls per resample: {args.n_subs}')
    print(f'Iterations: {args.iter}')
    print()
    
    # Load data
    selectivity_file = f'{processed_dir}/group_results/selectivity/selectivity_summary{args.suffix}.csv'
    
    if not os.path.exists(selectivity_file):
        print(f'ERROR: {selectivity_file} not found')
        print('Run 02_calc_summary_vals.py first')
        sys.exit(1)
    
    controls, patients = load_and_prepare(selectivity_file)
    
    # Bootstrap resample controls
    resample_dfs = resample_selectivity(controls, args.n_subs, args.iter)
    
    # Save resampled distributions
    out_dir = f'{processed_dir}/group_results/selectivity/resamples'
    os.makedirs(out_dir, exist_ok=True)
    
    for metric, rdf in resample_dfs.items():
        out_file = f'{out_dir}/{metric}_resamples{args.suffix}.csv'
        rdf.to_csv(out_file, index=False)
        print(f'  Saved: {out_file}')
    
    # Compute patient percentiles
    if len(patients) > 0:
        print('\nComputing patient percentiles...')
        patient_results = compute_patient_percentiles(patients, resample_dfs)
        
        pt_file = f'{processed_dir}/group_results/selectivity/patient_percentiles{args.suffix}.csv'
        patient_results.to_csv(pt_file, index=False)
        print(f'Saved: {pt_file}')
        
        # Summary table (like Ayzenberg Table 2)
        print('\n' + '=' * 60)
        print('PATIENT SUMMARY (sum_selec_norm, * = below 95% CI)')
        print('=' * 60)
        
        ssn = patient_results[patient_results['metric'] == 'sum_selec_norm']
        for sub in sorted(ssn['sub'].unique()):
            sub_data = ssn[ssn['sub'] == sub]
            intact = sub_data['intact_hemi'].iloc[0]
            group = sub_data['group'].iloc[0]
            
            line = f'{sub} ({group}, intact {intact}): '
            for _, r in sub_data.iterrows():
                marker = '*' if r['below_ci'] else '-'
                line += f'{r["category"]}_{r["hemi"]}={r["percentile"]:.1f}%{marker} '
            print(line)
    
    print('\nDone!')


if __name__ == '__main__':
    main()