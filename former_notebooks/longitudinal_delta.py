#!/usr/bin/env python3
"""Longitudinal delta: earliest vs latest session per patient (and per control).

For each subject × measure × parcel × hemi: Δ = latest - earliest value.
Sub-021 ses-02 excluded (unreliable per PI). Controls included as drift baseline.

Output: long CSV + printed summary highlighting persistent/increasing/decreasing patterns.
"""
import numpy as np
import pandas as pd
from pathlib import Path

CSV = Path('/user_data/csimmon2/sym_pt/group_results/longitudinal/longitudinal_metrics.csv')
OUT = Path('/user_data/csimmon2/sym_pt/group_results/longitudinal/longitudinal_delta.csv')

df = pd.read_csv(CSV)
EXCLUDE = {('sub-021', 2)}  # ses-02 unreliable

df = df[~df.apply(lambda r: (r['subject_id'], r['session']) in EXCLUDE, axis=1)]

MEASURES = ['wta_territory_pct', 'log10_sum_sel', 'distinctiveness']

rows = []
for sub in df['subject_id'].unique():
    sdf = df[df['subject_id'] == sub]
    sessions = sorted(sdf['session'].unique())
    if len(sessions) < 2:
        continue
    early_ses, late_ses = sessions[0], sessions[-1]
    group  = sdf['group'].iloc[0]
    intact = sdf['intact_hemi'].iloc[0]
    surg   = sdf['surgery_side'].iloc[0]
    # Special label for TC pre→post case
    spans_surgery = (sub == 'sub-021' and early_ses == 1)
    for measure in MEASURES:
        m = sdf[sdf['measure'] == measure]
        early_vals = m[m['session'] == early_ses].set_index(['parcel', 'hemi'])['value']
        late_vals  = m[m['session'] == late_ses].set_index(['parcel', 'hemi'])['value']
        common = early_vals.index.intersection(late_vals.index)
        for parcel, hemi in common:
            e, l = early_vals[(parcel, hemi)], late_vals[(parcel, hemi)]
            rows.append({
                'subject_id':     sub,
                'group':          group,
                'intact_hemi':    intact,
                'surgery_side':   surg,
                'spans_surgery':  spans_surgery,
                'early_session':  early_ses,
                'late_session':   late_ses,
                'measure':        measure,
                'parcel':         parcel,
                'hemi':           hemi,
                'early':          round(float(e), 3),
                'late':           round(float(l), 3),
                'delta':          round(float(l - e), 3),
            })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print(f'Saved: {OUT}  ({len(out_df)} rows)')

# ── Per-patient summary table per measure ───────────────────────────────────
print('\n' + '=' * 90)
print('LONGITUDINAL Δ (latest − earliest) PER PATIENT')
print('=' * 90)
for measure in MEASURES:
    print(f'\n--- {measure} ---')
    for grp in ('OTC', 'control'):
        sub_df = out_df[(out_df['measure'] == measure) & (out_df['group'] == grp)]
        if not len(sub_df):
            continue
        # Per patient: mean Δ in intact hemi (for patients) or both hemis (for controls)
        for sub in sorted(sub_df['subject_id'].unique()):
            s = sub_df[sub_df['subject_id'] == sub]
            intact_short = s['intact_hemi'].iloc[0][0] if s['intact_hemi'].iloc[0] != 'both' else None
            tag = ''
            if s['spans_surgery'].iloc[0]:
                tag = ' [PRE→POST]'
            elif grp == 'OTC':
                tag = f' [post→post, intact={intact_short}H]'
            # Limit to intact-hemi for patients
            display = s if grp == 'control' else s[s['hemi'] == intact_short]
            if not len(display):
                continue
            deltas = display.set_index('parcel')['delta'].to_dict()
            line = f'  {sub:>10s}{tag}: '
            line += ', '.join(f'{p}={d:+.2f}' for p, d in deltas.items())
            print(line)


# ── TC-specific compact: pre→stable post in intact RH ──────────────────────
print('\n' + '=' * 70)
print('SUB-021 (TC) PRE→POST in intact RH')
print('=' * 70)
tc = out_df[(out_df['subject_id'] == 'sub-021') & (out_df['hemi'] == 'r')]
for measure in MEASURES:
    sub = tc[tc['measure'] == measure].set_index('parcel')
    if not len(sub):
        continue
    print(f'\n{measure}:')
    for p, r in sub.iterrows():
        print(f'  {p:14s} pre={r["early"]:>7.2f}  post={r["late"]:>7.2f}  '
              f'Δ={r["delta"]:+7.2f}')