#!/usr/bin/env python3
"""
combat_06b_harmonize_rsa_scramble.py — harmonize the SCRAMBLE RSA (v2).

Same structure as combat_06_harmonize_rsa.py, which it deliberately mirrors line
for line where it can. It wraps verified/05_stats so the analyzed rows are
exactly the manuscript's (controls = first session, OTC = last; sub-017 out,
sub-108 ses-2 out, nonOTC out).

THREE feature blocks harmonized separately, per hemisphere:
  (1) liu_distinctiveness   one value per subject x ROI. Feature = category.
                            Scramble is NOT in this mean, so this block is the
                            same quantity as in combat_06 and stays comparable
                            to rsa_v1_harmonized.
  (2) dist_incl_scramble    the 4-condition version, scramble included in the
                            mean. NEW. Feature = category.
  (3) fisher_r              the pairwise correlations. Feature = roi__pair.
                            With scramble in the RDM this is 10 pairs per ROI
                            rather than 6; the block enumerates whatever pairs
                            are present, so the four scramble pairs are picked
                            up without any change to the logic.

Covariates group+age+sex, batch=scanner, winsorized 5th/95th within group.
Identical to combat_06.

Input:  D_liu/rsa_v2_scramble.csv
Output: D_liu/rsa_v2_scramble_harmonized.csv

VALIDATION after running. Block (1) is the same quantity combat_06 harmonizes,
computed from the same betas and spheres, so for the four primary ROIs it should
land close to rsa_v1_harmonized.csv. It will not match exactly — ComBat is fit on
a different feature set here (12 ROIs, 10 pairs), so the empirical Bayes shrinkage
differs. Large divergence means something else moved:

  python -c "
  import pandas as pd
  k=['subject_id','session','hemi','category']
  a=pd.read_csv('D_liu/rsa_v1_harmonized.csv')[k+['liu_distinctiveness']].drop_duplicates(k)
  b=pd.read_csv('D_liu/rsa_v2_scramble_harmonized.csv')[k+['liu_distinctiveness']].drop_duplicates(k)
  m=a.merge(b,on=k,suffixes=('_v1','_v2'))
  d=(m.liu_distinctiveness_v1-m.liu_distinctiveness_v2).abs()
  print(len(m),'shared rows  mean|diff|=%.4f  max=%.4f'%(d.mean(),d.max()))
  "

Usage:
  python combat_06b_harmonize_rsa_scramble.py
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from neuroHarmonize import harmonizationLearn

GIT     = Path('/user_data/csimmon2/git_repos/sym_pt')
RSA     = GIT / 'D_liu' / 'rsa_v2_scramble.csv'
SCANNER = GIT / 'F_harmonization' / 'sub_info_scanner.csv'
OUT     = GIT / 'D_liu' / 'rsa_v2_scramble_harmonized.csv'
WINSOR  = (5, 95)

# verified/05_stats has been renamed before; resolve rather than hard-code
_STATS_CANDIDATES = [
    GIT / 'D_liu' / 'verified' / '05_stats.py',
    GIT / 'D_liu' / 'verified' / '05_stats_harmony.py',
]
STATS = next((p for p in _STATS_CANDIDATES if p.exists()), None)
if STATS is None:
    _found = sorted((GIT / 'D_liu' / 'verified').glob('05_stats*.py'))
    sys.exit('Cannot find the verified stats module.\n'
             f'  Looked for: {[p.name for p in _STATS_CANDIDATES]}\n'
             f'  Present:    {[p.name for p in _found] or "none"}')

sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('verified_stats', str(STATS))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)                       # apply_exclusions, select_sessions


def winsorize_within_group(col, groups, lo, hi):
    """Clip each group's values to its [lo, hi] percentiles (lab: by group)."""
    out = col.astype(float).copy()
    for g in groups.unique():
        m = (groups == g) & col.notna()
        if m.sum() < 3:
            continue
        a, b = np.nanpercentile(col[m], [lo, hi])
        out[m] = col[m].clip(a, b)
    return out


def build_covariates(df, scan):
    """One selected session per subject -> scanner/age/sex covariates."""
    gcol = 'group' if 'group' in df.columns else 'status'
    df = df.copy()
    df['_grp'] = np.where(df[gcol].astype(str).str.upper().str.contains('OTC'),
                          'OTC', 'control')
    sel = df[['subject_id', 'session', '_grp']].drop_duplicates('subject_id')
    rows = []
    for _, r in sel.iterrows():
        sval = pd.to_numeric(r['session'], errors='coerce')
        ses = f"ses-{int(sval):02d}" if pd.notna(sval) else str(r['session'])
        m = scan[(scan['sub'] == r['subject_id']) & (scan['ses'] == ses)]
        if not len(m):
            print(f"  WARNING: no scanner ROW for {r['subject_id']} {ses} -> dropped")
            continue
        m = m.iloc[0]
        if pd.isna(m['scanner']):
            print(f"  WARNING: scanner LABEL missing for {r['subject_id']} {ses} -> dropped")
            continue
        rows.append({'subject_id': r['subject_id'], 'group': r['_grp'],
                     'scanner': str(m['scanner']), 'age': m['age'], 'sex': m['sex']})
    return df, pd.DataFrame(rows).set_index('subject_id')


def harmonize_block(dh, value_col, feature_col, subs, cov, hemi, label):
    """Harmonize one long-format value column; feature_col defines the feature set.
    Returns {(subject_id, feature): harmonized_value} for valid cells."""
    feats = sorted(dh[feature_col].dropna().unique())
    groups = cov.loc[subs, 'group']
    wide, colnames = [], []
    for f in feats:
        piv = (dh[dh[feature_col] == f]
               .pivot_table(index='subject_id', columns=feature_col,
                            values=value_col, aggfunc='first')
               .reindex(subs))
        col = winsorize_within_group(piv[f], groups, *WINSOR)
        wide.append(col.values); colnames.append(f)
    X = np.array(wide, dtype=float).T                # [n_subj x n_feat]

    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"  [{label}] {int(nan_mask.sum())} NaN cells imputed for fit (restored after)")
        for j in range(X.shape[1]):
            colj = X[:, j]
            for g in groups.unique():
                gm = (groups.values == g)
                mu = np.nanmean(colj[gm])
                colj[gm & np.isnan(colj)] = mu if not np.isnan(mu) else np.nanmean(colj)
            X[:, j] = colj

    sex_d = pd.get_dummies(cov.loc[subs, 'sex'], prefix='sex',
                           drop_first=True).astype(int).reset_index(drop=True)
    grp_d = pd.get_dummies(cov.loc[subs, 'group'], prefix='group',
                           drop_first=True).astype(int).reset_index(drop=True)
    design = pd.DataFrame({'SITE': cov.loc[subs, 'scanner'].values,
                           'age':  cov.loc[subs, 'age'].values})
    design = pd.concat([design, sex_d, grp_d], axis=1)

    _, adj = harmonizationLearn(X, design, smooth_terms=[])
    adj[nan_mask] = np.nan

    site = cov.loc[subs, 'scanner'].values
    ctrl = (cov.loc[subs, 'group'] == 'control').values
    def gap(M, a, b):
        return float(np.nanmean(np.abs(np.nanmean(M[a], 0) - np.nanmean(M[b], 0))))
    print(f"  [{label}] {len(colnames)} features")
    print(f"  [{label}] site gap : {gap(X, site=='Verio', site=='Prisma'):.3f} -> "
          f"{gap(adj, site=='Verio', site=='Prisma'):.3f}")
    print(f"  [{label}] group gap: {gap(X, ctrl, ~ctrl):.3f} -> "
          f"{gap(adj, ctrl, ~ctrl):.3f}")

    out = {}
    for j, f in enumerate(colnames):
        for i, sid in enumerate(subs):
            if not nan_mask[i, j]:
                out[(sid, f)] = adj[i, j]
    return out


def main():
    df = pd.read_csv(RSA)
    print(f'{RSA.name}: {len(df)} rows, {df.subject_id.nunique()} subjects, '
          f'{df["pair"].nunique()} pairs, {df.category.nunique()} ROIs')
    df = s.apply_exclusions(df)
    df = s.select_sessions(df, pt_rule='last')       # controls=first, OTC=last
    scan = pd.read_csv(SCANNER)
    df, cov = build_covariates(df, scan)

    df_out = df.copy()
    for c in ['liu_distinctiveness', 'dist_incl_scramble', 'fisher_r']:
        df_out[c] = df_out[c].astype(float)

    for hemi in ['l', 'r']:
        dh = df[df['hemi'] == hemi]
        subs = [sid for sid in cov.index if sid in dh['subject_id'].values]
        bad = [sid for sid in subs
               if cov.loc[sid, ['scanner', 'age', 'sex']].isna().any()]
        if bad:
            print(f"  [{hemi.upper()}H] dropping {len(bad)} subj w/ incomplete covars: {bad}")
            subs = [sid for sid in subs if sid not in bad]
        if len(subs) < 5:
            continue
        print(f"\n[{hemi.upper()}H] {len(subs)} subjects")

        # (1) distinctiveness, scramble EXCLUDED — comparable to rsa_v1_harmonized
        dd = (dh.dropna(subset=['liu_distinctiveness'])
                .drop_duplicates(['subject_id', 'category']))
        for (sid, cat), val in harmonize_block(
                dd, 'liu_distinctiveness', 'category', subs, cov, hemi, 'distinct').items():
            m = ((df_out['subject_id'] == sid) & (df_out['hemi'] == hemi)
                 & (df_out['category'] == cat))
            df_out.loc[m, 'liu_distinctiveness'] = val

        # (2) distinctiveness, scramble INCLUDED — new 4-condition version
        ds = (dh.dropna(subset=['dist_incl_scramble'])
                .drop_duplicates(['subject_id', 'category']))
        for (sid, cat), val in harmonize_block(
                ds, 'dist_incl_scramble', 'category', subs, cov, hemi, 'dist+scr').items():
            m = ((df_out['subject_id'] == sid) & (df_out['hemi'] == hemi)
                 & (df_out['category'] == cat))
            df_out.loc[m, 'dist_incl_scramble'] = val

        # (3) geometry: feature = category__pair, now 10 pairs per ROI
        dg = dh.dropna(subset=['fisher_r']).copy()
        dg['_feat'] = dg['category'].astype(str) + '__' + dg['pair'].astype(str)
        for (sid, feat), val in harmonize_block(
                dg, 'fisher_r', '_feat', subs, cov, hemi, 'geometry').items():
            cat, pair = feat.split('__', 1)
            m = ((df_out['subject_id'] == sid) & (df_out['hemi'] == hemi)
                 & (df_out['category'] == cat) & (df_out['pair'] == pair))
            df_out.loc[m, 'fisher_r'] = val

    df_out.drop(columns=['_grp'], errors='ignore').to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df_out)} rows)")

    prim = ['object_LOC', 'house_PPA_strict', 'face_FFA', 'word_VWFA']
    c = (df_out[(df_out['status'] == 'control') & df_out['category'].isin(prim)]
         .drop_duplicates(['subject_id', 'hemi', 'category']))
    print("\nHarmonized control means, primary ROIs:")
    print(c.groupby(['category', 'hemi'])[['liu_distinctiveness',
                                           'dist_incl_scramble']]
           .mean().round(3).to_string())


if __name__ == '__main__':
    main()
