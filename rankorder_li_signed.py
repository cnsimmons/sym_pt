#!/usr/bin/env python3
"""
rankorder_li.py — Per-patient rank-order test, split by intact hemisphere.
Two a priori orders (fixed; data never edits them):
  SUM-SEL (extent)      : object = house > face > word   (bilateral lose redundancy)
  DISTINCTIVENESS       : word > face > object = house    (lateralized displaced → less distinct)
Effect = loss of the good quantity in patients:
  sum_selec_norm      higher=better → loss = ctrl - pt
  liu_distinctiveness stores SIMILARITY, higher=worse → loss of distinctiveness = pt - ctrl
Per patient: Spearman(effect, expected order); Wilcoxon on rho's, per subgroup.
Prints only. CSV provenance not yet verified.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

UNIV = '/user_data/csimmon2/git_repos/sym_pt/D_liu/univariate_v1_harmonized.csv'
RSA  = '/user_data/csimmon2/git_repos/sym_pt/D_liu/rsa_v1_harmonized.csv'
EXCLUDE = {'sub-027', 'sub-084'}
CATS = ['object', 'house', 'face', 'word']
ROI  = {'object': 'object_LOC', 'house': 'house_PPA', 'face': 'face_FFA', 'word': 'word_VWFA'}

# two fixed hypotheses (higher value = more effect expected). object=house always tied.
EXP_SUMSEL = {'object': 3, 'house': 3, 'face': 2, 'word': 1}     # tie at top
EXP_DIST   = {'word': 3, 'face': 2, 'object': 1, 'house': 1}     # tie at bottom (reverse)

def exp_vector(d): return np.array([d[c] for c in CATS])

# self-tests: perfect follow = +1, exact reverse = -1, for BOTH orders
for d in (EXP_SUMSEL, EXP_DIST):
    v = exp_vector(d)
    assert spearmanr(v, v)[0] == 1.0 and spearmanr(-v, v)[0] == -1.0

def prep(df):
    df = df.drop_duplicates(['subject_id', 'hemi', 'category'])
    keep = {v: k for k, v in ROI.items()}
    df = df[df['category'].isin(keep)].copy()
    df['category'] = df['category'].map(keep)
    return df

def run(csv, valcol, measure, exp_dict, higher_is_better):
    if isinstance(next(iter(exp_dict.values())), dict):
        exp_vecs = {h: exp_vector(v) for h, v in exp_dict.items()}
        ord_ref = exp_dict['l']
    else:
        exp_vecs = {'l': exp_vector(exp_dict), 'r': exp_vector(exp_dict)}
        ord_ref = exp_dict
    df = prep(pd.read_csv(csv).pipe(lambda d: d[~d['subject_id'].isin(EXCLUDE)]))
    ctrl = (df[df['group'] == 'control']
            .groupby(['hemi', 'category'])[valcol].mean().rename('cm').reset_index())
    pts = sorted(df[df['group'] == 'OTC']['subject_id'].unique())

    ord_str = ' > '.join(k for k, _ in sorted(ord_ref.items(), key=lambda x: -x[1]))
    print(f"\n{'='*70}\n{measure}  col='{valcol}'  (higher_is_better={higher_is_better})"
          f"\n  predicted: {ord_str}\n{'='*70}")
    print(f"{'patient':>12} {'hemi':>5} {'rho':>7} {'p':>7}   effect(obj,hou,fac,wor)")
    rec = {'l': [], 'r': []}          # rho by intact hemi
    word_rh = []
    for sid in pts:
        p = df[(df['subject_id'] == sid) & (df['group'] == 'OTC')]
        hemi = p['hemi'].iloc[0]
        eff = []
        for c in CATS:
            pv = p[p['category'] == c][valcol]
            cm = ctrl[(ctrl['hemi'] == hemi) & (ctrl['category'] == c)]['cm']
            val = (cm.mean() - pv.mean()) if higher_is_better else (pv.mean() - cm.mean())
            eff.append(val if len(pv) and len(cm) else np.nan)
        eff = np.array(eff)
        if np.isnan(eff).any():
            print(f"{sid:>12} {hemi:>5}   (missing category)"); continue
        if hemi == 'r': word_rh.append(eff[CATS.index('word')])
        rho, pv = spearmanr(eff, exp_vecs[hemi])
        rec[hemi].append(rho)
        print(f"{sid:>12} {hemi:>5} {rho:>+7.2f} {pv:>7.3f}   " + ','.join(f'{v:+.2f}' for v in eff))

    def summarize(label, arr):
        arr = np.array(arr)
        if not len(arr):
            print(f"  [{label}] no data"); return
        line = f"  [{label}]  n={len(arr)}  mean rho={arr.mean():+.3f}  median={np.median(arr):+.3f}"
        if len(arr) >= 3:
            w = wilcoxon(arr)
            line += f"  Wilcoxon W={w.statistic:.1f} p={w.pvalue:.4f}  ({(arr>0).sum()}/{len(arr)} pos)"
        print(line)

    print()
    summarize('LH-intact', rec['l'])
    summarize('RH-intact', rec['r'])
    summarize('POOLED',    rec['l'] + rec['r'])
    if word_rh:
        m = np.nanmean(word_rh)
        print(f"  [check] word RH-intact mean effect = {m:+.3f}  "
              f"{'OK (matches rVWFA)' if m > 0 else '*** SIGN WRONG ***'}")


SIGNED_LI = {'word': 0.368, 'face': -0.203, 'object': 0.114, 'house': -0.109}
# expected impact = dependence on the RESECTED hemisphere
EXP_SIGNED = {'l': {c: -v for c, v in SIGNED_LI.items()},   # LH-intact: RH resected
              'r': {c: +v for c, v in SIGNED_LI.items()}}   # RH-intact: LH resected

run(UNIV, 'sum_selec_norm',      'SUM-SEL (extent) - SIGNED LI', EXP_SIGNED, higher_is_better=True)
run(RSA,  'liu_distinctiveness', 'DISTINCTIVENESS - SIGNED LI',  EXP_SIGNED, higher_is_better=False)
