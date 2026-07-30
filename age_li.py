#!/usr/bin/env python3
"""age_li.py — Does age predict degree of lateralization (|LI|) per category in controls?"""
import numpy as np, pandas as pd
from scipy.stats import linregress

CSV   = '/user_data/csimmon2/sym_pt/group_results/roi_betas/selective_voxel_counts.csv'
SUBINFO = '/user_data/csimmon2/git_repos/sym_pt/sub_info.csv'
EXCLUDE = {'sub-027', 'sub-084'}
MIN_TOTAL, THR = 10, 2.33          # 2.33 = Ayzenberg resection precedent
CATS = ['face', 'house', 'object', 'word']

df = pd.read_csv(CSV)
df = df[(df['group'] == 'control') & (df['threshold'] == THR) & (~df['subject_id'].isin(EXCLUDE))]

# age merge (normalize session strings on both sides)
si = pd.read_csv(SUBINFO)[['sub', 'ses', 'age']].rename(columns={'sub': 'subject_id'})
si['ses'] = si['ses'].astype(str).str.replace('ses-', '', regex=False).str.zfill(2)
df['ses'] = df['session'].astype(str).str.replace('ses-', '', regex=False).str.zfill(2)
df = df.merge(si, on=['subject_id', 'ses'], how='left')
if df['age'].isna().any():
    print("WARN: unmatched age for:", df[df['age'].isna()]['subject_id'].unique())

print(f"threshold z>{THR}   n controls = {df['subject_id'].nunique()}\n")
print(f"{'cat':>7} {'n':>4} {'slope(|LI|/yr)':>15} {'r':>7} {'p':>8}   direction")
for cat in CATS:
    w = df[df['category'] == cat].pivot_table(index='subject_id', columns='hemi',
                                              values='n_selective', aggfunc='first')
    a = df[df['category'] == cat].groupby('subject_id')['age'].first()
    w = w.join(a).dropna(subset=['l', 'r', 'age'])
    w = w[(w['l'] + w['r']) >= MIN_TOTAL]
    absli = ((w['l'] - w['r']) / (w['l'] + w['r'])).abs()
    lr = linregress(w['age'], absli)
    arrow = 'strengthens ↑' if (lr.pvalue < .05 and lr.slope > 0) else \
            ('weakens ↓' if (lr.pvalue < .05 and lr.slope < 0) else 'flat')
    print(f"{cat:>7} {len(w):>4} {lr.slope:>15.4f} {lr.rvalue:>7.3f} {lr.pvalue:>8.4f}   {arrow}")