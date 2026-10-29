#!/usr/bin/env python3
"""
combat_06_harmonize_rsa.py — harmonize the RSA measures (distinctiveness + geometry).

WRAPPER around verified/05_stats: reuses its apply_exclusions + select_sessions so
the analyzed rows are EXACTLY the manuscript's (controls=first session, OTC=last;
sub-017 out, sub-108 ses-2 out, nonOTC out).

Two feature blocks harmonized separately, per hemisphere:
  (1) liu_distinctiveness  — one value per subject x ROI(category)
  (2) fisher_r             — the 6 geometry pairs per subject x ROI; harmonized
                             with the ROI's pair as the feature (roi__pair).
Covariates group+age+sex, batch=scanner, winsorized 5th/95th within group (lab procedure).

Output: D_liu/rsa_v1_harmonized.csv  (same schema; distinctiveness + fisher_r harmonized).
Then run 05_stats_harmony pointed at it (--rsa) for the with/without comparison.
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from neuroHarmonize import harmonizationLearn

GIT      = Path('/user_data/csimmon2/git_repos/sym_pt')
STATS    = GIT / 'D_liu' / 'verified' / '05_stats.py'
RSA      = GIT / 'D_liu' / 'rsa_v1.csv'
SCANNER  = GIT / 'F_harmonization' / 'sub_info_scanner.csv'
OUT      = GIT / 'D_liu' / 'rsa_v1_harmonized.csv'
WINSOR   = (5, 95)

sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('verified_stats', str(STATS))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)                       # defines apply_exclusions, select_sessions


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
    df['_grp'] = np.where(df[gcol].astype(str).str.upper().str.contains('OTC'), 'OTC', 'control')
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
    """Harmonize one long-format value column, with feature_col defining the feature set.
    Returns dict {(subject_id, feature): harmonized_value} for valid cells."""
    feats = sorted(dh[feature_col].dropna().unique())
    groups = cov.loc[subs, 'group']
    wide, colnames = [], []
    for f in feats:
        piv = (dh[dh[feature_col] == f]
               .pivot_table(index='subject_id', columns=feature_col, values=value_col,
                            aggfunc='first')
               .reindex(subs))
        col = winsorize_within_group(piv[f], groups, *WINSOR)
        wide.append(col.values); colnames.append(f)
    X = np.array(wide, dtype=float).T               # [n_subj x n_feat]

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

    sex_d = pd.get_dummies(cov.loc[subs, 'sex'],   prefix='sex',   drop_first=True).astype(int).reset_index(drop=True)
    grp_d = pd.get_dummies(cov.loc[subs, 'group'], prefix='group', drop_first=True).astype(int).reset_index(drop=True)
    design = pd.DataFrame({'SITE': cov.loc[subs, 'scanner'].values,
                           'age':  cov.loc[subs, 'age'].values})
    design = pd.concat([design, sex_d, grp_d], axis=1)

    _, adj = harmonizationLearn(X, design, smooth_terms=[])
    adj[nan_mask] = np.nan

    site = cov.loc[subs, 'scanner'].values
    ctrl = (cov.loc[subs, 'group'] == 'control').values
    def gap(M, a, b): return float(np.nanmean(np.abs(np.nanmean(M[a], 0) - np.nanmean(M[b], 0))))
    print(f"  [{label}] site gap : {gap(X, site=='Verio', site=='Prisma'):.3f} -> {gap(adj, site=='Verio', site=='Prisma'):.3f}")
    print(f"  [{label}] group gap: {gap(X, ctrl, ~ctrl):.3f} -> {gap(adj, ctrl, ~ctrl):.3f}")

    out = {}
    for j, f in enumerate(colnames):
        for i, sid in enumerate(subs):
            if not nan_mask[i, j]:
                out[(sid, f)] = adj[i, j]
    return out


def main():
    df = pd.read_csv(RSA)
    df = s.apply_exclusions(df)
    df = s.select_sessions(df, pt_rule='last')      # controls=first, OTC=last (manuscript)
    scan = pd.read_csv(SCANNER)
    df, cov = build_covariates(df, scan)

    df_out = df.copy()
    df_out['liu_distinctiveness'] = df_out['liu_distinctiveness'].astype(float)
    df_out['fisher_r'] = df_out['fisher_r'].astype(float)

    for hemi in ['l', 'r']:
        dh = df[df['hemi'] == hemi]
        subs = [sid for sid in cov.index if sid in dh['subject_id'].values]
        bad = [sid for sid in subs if cov.loc[sid, ['scanner', 'age', 'sex']].isna().any()]
        if bad:
            print(f"  [{hemi.upper()}H] dropping {len(bad)} subj w/ incomplete covars: {bad}")
            subs = [sid for sid in subs if sid not in bad]
        if len(subs) < 5:
            continue
        print(f"\n[{hemi.upper()}H] {len(subs)} subjects")

        # (1) distinctiveness: one row per subject x category -> feature = category
        dd = dh.dropna(subset=['liu_distinctiveness']).drop_duplicates(['subject_id', 'category'])
        dist_map = harmonize_block(dd, 'liu_distinctiveness', 'category', subs, cov, hemi, 'distinct')
        for (sid, cat), val in dist_map.items():
            m = (df_out['subject_id'] == sid) & (df_out['hemi'] == hemi) & (df_out['category'] == cat)
            df_out.loc[m, 'liu_distinctiveness'] = val

        # (2) geometry: feature = category__pair (pairwise fisher_r within each ROI)
        dg = dh.dropna(subset=['fisher_r']).copy()
        dg['_feat'] = dg['category'].astype(str) + '__' + dg['pair'].astype(str)
        geo_map = harmonize_block(dg, 'fisher_r', '_feat', subs, cov, hemi, 'geometry')
        for (sid, feat), val in geo_map.items():
            cat, pair = feat.split('__', 1)
            m = ((df_out['subject_id'] == sid) & (df_out['hemi'] == hemi)
                 & (df_out['category'] == cat) & (df_out['pair'] == pair))
            df_out.loc[m, 'fisher_r'] = val

    df_out.drop(columns=['_grp'], errors='ignore').to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print("Next: run 05_stats_harmony pointed at it (--rsa rsa_v1_harmonized.csv).")


if __name__ == '__main__':
    main()