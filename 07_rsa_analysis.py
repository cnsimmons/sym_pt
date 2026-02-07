#!/usr/bin/env python3
"""
07_rsa_analysis.py - RSA analysis following Liu 2025

ROI definition: cope_selective (category > all others)
RSA input: cope_identity (raw per-condition estimates)
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy.spatial.distance import squareform
from nilearn import image
from long_pt_params import (
    processed_dir, csv_file, task, categories,
    cope_identity, cope_selective, roi_threshold,
    skip_subs, get_sessions, get_runs, get_cope
)


def load_cope(sub, ses, run, cope_num, space='standard'):
    """Load cope image"""
    path = get_cope(sub, ses, run, cope_num, space)
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata()


def get_roi_mask(sub, ses, run, category, threshold_pct=90):
    """Create ROI mask from selective contrast (top X%)"""
    cope_num = cope_selective[category]
    data = load_cope(sub, ses, run, cope_num)
    
    if data is None:
        return None
    
    # Threshold at percentile
    thresh = np.percentile(data[data > 0], threshold_pct)
    mask = data >= thresh
    
    return mask


def extract_patterns(sub, ses, run, mask):
    """Extract activation patterns for all categories within mask"""
    patterns = {}
    for cat in categories:
        cope_num = cope_identity[cat]
        data = load_cope(sub, ses, run, cope_num)
        if data is None:
            return None
        patterns[cat] = data[mask]
    return patterns


def compute_rdm(patterns):
    """Compute RDM from patterns (1 - Pearson correlation)"""
    cats = list(patterns.keys())
    n = len(cats)
    rdm = np.zeros((n, n))
    
    for i, cat_i in enumerate(cats):
        for j, cat_j in enumerate(cats):
            if i < j:
                r = stats.pearsonr(patterns[cat_i], patterns[cat_j])[0]
                rdm[i, j] = 1 - r
                rdm[j, i] = 1 - r
    
    return rdm, cats


def compute_selectivity(patterns, preferred_cat):
    """Compute representational selectivity (Liu 2025 method)
    
    Returns Fisher-transformed correlation between preferred 
    and non-preferred categories (lower = more selective)
    """
    pref_pattern = patterns[preferred_cat]
    non_pref_cats = [c for c in patterns.keys() if c != preferred_cat]
    
    correlations = []
    for cat in non_pref_cats:
        r = stats.pearsonr(pref_pattern, patterns[cat])[0]
        correlations.append(np.arctanh(r))  # Fisher transform
    
    return np.mean(correlations)


def analyze_session(sub, ses):
    """Analyze one session"""
    sub_clean = sub.replace('sub-', '')
    runs = get_runs(sub_clean, ses)
    
    results = {cat: {'rdm': [], 'selectivity': []} for cat in categories}
    
    for run in runs:
        for cat in categories:
            # Get ROI mask for this category
            mask = get_roi_mask(sub_clean, ses, run, cat, roi_threshold)
            if mask is None or mask.sum() == 0:
                continue
            
            # Extract patterns
            patterns = extract_patterns(sub_clean, ses, run, mask)
            if patterns is None:
                continue
            
            # Compute RDM
            rdm, labels = compute_rdm(patterns)
            results[cat]['rdm'].append(rdm)
            
            # Compute selectivity
            sel = compute_selectivity(patterns, cat)
            results[cat]['selectivity'].append(sel)
    
    return results


def compute_geometry_preservation(rdm_t1, rdm_t2):
    """Compute geometry preservation between two RDMs"""
    # Vectorize upper triangles
    vec_t1 = squareform(rdm_t1)
    vec_t2 = squareform(rdm_t2)
    
    # Pearson correlation
    r, p = stats.pearsonr(vec_t1, vec_t2)
    return r, p


def main():
    print('Running RSA analysis (Liu 2025 method)...')
    print(f'ROI threshold: top {100 - roi_threshold}%')
    print(f'Categories: {categories}')
    
    df = pd.read_csv(csv_file)
    all_results = []
    
    for _, row in df.iterrows():
        sub = row['sub'].replace('sub-', '')
        
        if sub in skip_subs:
            continue
        
        sessions = get_sessions(sub, df)
        if len(sessions) < 2:
            print(f'\nsub-{sub}: <2 sessions, skipping')
            continue
        
        print(f'\nsub-{sub}')
        
        # Analyze each session
        session_results = {}
        for ses in sessions:
            print(f'  Session {ses}...')
            session_results[ses] = analyze_session(sub, ses)
        
        # Compute geometry preservation (T1 vs T2)
        t1, t2 = sessions[0], sessions[1]
        
        for cat in categories:
            rdms_t1 = session_results[t1][cat]['rdm']
            rdms_t2 = session_results[t2][cat]['rdm']
            
            if not rdms_t1 or not rdms_t2:
                continue
            
            # Average RDMs across runs
            avg_rdm_t1 = np.mean(rdms_t1, axis=0)
            avg_rdm_t2 = np.mean(rdms_t2, axis=0)
            
            # Geometry preservation
            gp, gp_p = compute_geometry_preservation(avg_rdm_t1, avg_rdm_t2)
            
            # Average selectivity
            sel_t1 = np.mean(session_results[t1][cat]['selectivity'])
            sel_t2 = np.mean(session_results[t2][cat]['selectivity'])
            
            all_results.append({
                'subject': sub,
                'category': cat,
                'geometry_preservation': gp,
                'gp_pvalue': gp_p,
                'selectivity_t1': sel_t1,
                'selectivity_t2': sel_t2,
                'selectivity_change': sel_t2 - sel_t1
            })
            
            print(f'    {cat}: GP={gp:.3f}, Sel_T1={sel_t1:.3f}, Sel_T2={sel_t2:.3f}')
    
    # Save results
    results_df = pd.DataFrame(all_results)
    out_file = f'{processed_dir}/rsa_results.csv'
    results_df.to_csv(out_file, index=False)
    print(f'\nResults saved to {out_file}')
    
    # Summary stats
    print('\n=== Summary ===')
    for cat in categories:
        cat_data = results_df[results_df['category'] == cat]
        if len(cat_data) > 0:
            gp_mean = cat_data['geometry_preservation'].mean()
            gp_std = cat_data['geometry_preservation'].std()
            print(f'{cat}: GP = {gp_mean:.3f} ± {gp_std:.3f}')


if __name__ == '__main__':
    main()
