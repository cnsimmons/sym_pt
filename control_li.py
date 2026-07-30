#!/usr/bin/env python3
"""
control_li.py — Summed-count laterality index in controls, with across-threshold robustness.
LI = (L - R) / (L + R) on n_selective voxels, per control x category x threshold.
Positive = LH-dominant. Controls only. Raw counts (no ComBat needed for within-subject ratio).
Prints only; no files written.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── config ────────────────────────────────────────────────────────────────────
CSV_PATH   = '/user_data/csimmon2/sym_pt/group_results/roi_betas/selective_voxel_counts.csv'
EXCLUDE    = {'sub-027', 'sub-084'}   # current-cohort control exclusions
MIN_TOTAL  = 10        # drop a subject×category×threshold cell if L+R < this (LI unstable)
CATS       = ['face', 'house', 'object', 'word']

df = pd.read_csv(CSV_PATH)
df = df[df['group'] == 'control'].copy()
df = df[~df['subject_id'].isin(EXCLUDE)]

print(f"Controls in file after exclusions: {df['subject_id'].nunique()}  "
      f"(expected 38 for current cohort)")
print(f"Thresholds present: {sorted(df['threshold'].unique())}\n")

# ── pivot L/R, compute LI ───────────────────────────────────────────────────────
rows = []
for thr in sorted(df['threshold'].unique()):
    for cat in CATS:
        sub = df[(df['threshold'] == thr) & (df['category'] == cat)]
        w = sub.pivot_table(index='subject_id', columns='hemi',
                            values='n_selective', aggfunc='first')
        if 'l' not in w or 'r' not in w:
            print(f"  WARN {cat} thr={thr}: missing a hemisphere column"); continue
        w = w.dropna(subset=['l', 'r'])
        tot = w['l'] + w['r']
        w = w[tot >= MIN_TOTAL]
        li = (w['l'] - w['r']) / (w['l'] + w['r'])
        for sid, v in li.items():
            rows.append({'threshold': thr, 'category': cat,
                         'subject_id': sid, 'LI': v, 'absLI': abs(v)})

li_df = pd.DataFrame(rows)

# ── per-category summary + one-sample test (is it lateralized?) ──────────────────
print("="*72)
print("PER-CATEGORY LI  (LI>0 = LH-dominant; test = Wilcoxon vs 0)")
print("="*72)
print(f"{'thr':>5} {'cat':>7} {'n':>4} {'meanLI':>8} {'medLI':>7} "
      f"{'mean|LI|':>9} {'p_vs0':>8}")
for thr in sorted(li_df['threshold'].unique()):
    for cat in CATS:
        s = li_df[(li_df['threshold'] == thr) & (li_df['category'] == cat)]['LI']
        if len(s) < 2:
            print(f"{thr:>5} {cat:>7} {len(s):>4}  (too few)"); continue
        try:    p = wilcoxon(s, zero_method='wilcox').pvalue
        except  ValueError: p = np.nan   # all zeros
        print(f"{thr:>5} {cat:>7} {len(s):>4} {s.mean():>8.3f} {s.median():>7.3f} "
              f"{s.abs().mean():>9.3f} {p:>8.4f}")
    print()

# ── degree-of-lateralization ranking + the face-vs-{object,house} test ──────────
print("="*72)
print("DEGREE OF LATERALIZATION  |LI|  (two-level story: word >> face≈object≈house)")
print("="*72)
for thr in sorted(li_df['threshold'].unique()):
    print(f"\n-- threshold z>{thr} --")
    means = {c: li_df[(li_df['threshold']==thr) & (li_df['category']==c)]['absLI'].mean()
             for c in CATS}
    for c in sorted(means, key=means.get, reverse=True):
        print(f"   {c:>7}  mean|LI| = {means[c]:.3f}")

    # paired within-subject: face vs object, face vs house, word vs face
    wide = li_df[li_df['threshold']==thr].pivot_table(
        index='subject_id', columns='category', values='absLI')
    for a, b in [('face','object'), ('face','house'), ('word','face')]:
        pair = wide[[a, b]].dropna()
        if len(pair) < 2:
            print(f"   {a} vs {b}: too few pairs"); continue
        try:    p = wilcoxon(pair[a], pair[b]).pvalue
        except  ValueError: p = np.nan
        print(f"   |LI| {a} vs {b}: n={len(pair)}, "
              f"med {pair[a].median():.3f} vs {pair[b].median():.3f}, p={p:.4f}")