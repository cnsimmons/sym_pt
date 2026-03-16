#!/usr/bin/env python3
"""Pull per-ROI pairwise values for F30 pilot figure heatmaps."""

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

# ── Config ────────────────────────────────────────────────────────────────────
TARGETS = ['sub-004', 'sub-007', 'sub-052']
ROIS = ['face', 'house', 'object', 'word']
PAIRS = ['face-house', 'face-object', 'face-word', 'house-object', 'house-word', 'object-word']
COPE_SET = 'differential'  # adjust if your primary is different

# ── Load pairwise CSV ─────────────────────────────────────────────────────────
pairwise_file = f'{processed_dir}/group_results/liu_distinctiveness/pairwise_{COPE_SET}.csv'
if not os.path.exists(pairwise_file):
    # try alternate naming
    for f in os.listdir(f'{processed_dir}/group_results/liu_distinctiveness/'):
        if 'pairwise' in f and COPE_SET in f:
            pairwise_file = f'{processed_dir}/group_results/liu_distinctiveness/{f}'
            break

print(f"Loading: {pairwise_file}")
df = pd.read_csv(pairwise_file)
print(f"Columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")
print(f"Unique categories (ROIs): {sorted(df['category'].unique())}")
print()

# ── Per-ROI pairwise values for each target ───────────────────────────────────
print("=" * 80)
print("PER-ROI PAIRWISE VALUES")
print("=" * 80)

for sid in TARGETS:
    sub = df[df['subject_id'] == sid]
    if len(sub) == 0:
        print(f"\n{sid}: NO DATA")
        continue
    
    # Get first relevant session
    sessions = sorted(sub['session'].unique())
    # For patients, skip pre-surgical if applicable
    first_ses = sessions[0] if len(sessions) == 1 else sessions[0]
    # sub-021 uses ses-02 (ses-01 is pre-op), but our targets don't include 021
    ses_data = sub[sub['session'] == first_ses]
    
    hemi_options = ses_data['hemi_label'].unique()
    # prefer 'intact' for patients, 'left' or 'right' for controls
    if 'intact' in hemi_options:
        ses_data = ses_data[ses_data['hemi_label'] == 'intact']
    
    print(f"\n{'='*60}")
    print(f"{sid} (session: {first_ses}, rows: {len(ses_data)})")
    print(f"{'='*60}")
    
    for roi in ROIS:
        roi_data = ses_data[ses_data['category'] == roi]
        if len(roi_data) == 0:
            print(f"\n  {roi} ROI: NO DATA")
            continue
        
        print(f"\n  {roi} ROI:")
        for pair in PAIRS:
            pair_data = roi_data[roi_data['pair'] == pair]
            if len(pair_data) == 0:
                print(f"    {pair}: —")
            else:
                val = pair_data['fisher_r'].values[0]
                # Flag if this pair involves the ROI's preferred category
                is_preferred = roi in pair.split('-')
                marker = " ← preferred" if is_preferred else ""
                print(f"    {pair}: {val:.3f}{marker}")

# ── Control group per-ROI averages ────────────────────────────────────────────
print(f"\n{'='*80}")
print("CONTROL GROUP PER-ROI AVERAGES")
print("=" * 80)

ctrl = df[df['group'] == 'control']
# First session per control
ctrl_first = ctrl.groupby('subject_id')['session'].min().reset_index()
ctrl_first.columns = ['subject_id', 'first_ses']
ctrl = ctrl.merge(ctrl_first, on='subject_id')
ctrl = ctrl[ctrl['session'] == ctrl['first_ses']]

for roi in ROIS:
    roi_data = ctrl[ctrl['category'] == roi]
    print(f"\n  {roi} ROI (n={roi_data['subject_id'].nunique()} controls):")
    for pair in PAIRS:
        pair_data = roi_data[roi_data['pair'] == pair]
        vals = pair_data['fisher_r'].dropna()
        if len(vals) == 0:
            print(f"    {pair}: —")
        else:
            is_preferred = roi in pair.split('-')
            marker = " ← preferred" if is_preferred else ""
            print(f"    {pair}: M={vals.mean():.3f} SD={vals.std():.3f} [{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]{marker}")

print(f"\n{'='*80}")
print("DONE")
print("=" * 80)