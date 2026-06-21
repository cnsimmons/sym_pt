#!/usr/bin/env python3
"""
combat_05_harmonize_univariate.py — harmonize the parcel extent measures (step 6).

WRAPPER around verified/05_stats: reuses its apply_exclusions + select_sessions so
the analyzed rows are EXACTLY the manuscript's (controls=first session, OTC=last;
sub-017 out, sub-108 ses-2 out, nonOTC out).

Harmonizes sum_selec_norm, volume, mean_act at PARCEL level (each ROI = a feature),
per hemisphere, covariates group+age+sex, winsorized 5th/95th within group
(lab procedure). peak/coords are left untouched (relative).

Output: D_liu/univariate_v1_harmonized.csv  (same schema, 3 measures harmonized).
Then run 05_stats pointed at it for the with/without comparison.
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from neuroHarmonize import harmonizationLearn

GIT      = Path('/user_data/csimmon2/git_repos/sym_pt')
STATS    = GIT / 'D_liu' / 'verified' / '05_stats.py'
UNIVAR   = GIT / 'D_liu' / 'univariate_v1.csv'
SCANNER  = GIT / 'F_harmonization' / 'sub_info_scanner.csv'
OUT      = GIT / 'D_liu' / 'univariate_v1_harmonized.csv'
MEASURES = ['sum_selec_norm', 'volume', 'mean_act']
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


def main():
    df = pd.read_csv(UNIVAR)
    df = s.apply_exclusions(df)
    df = s.select_sessions(df, pt_rule='last')   # controls=first, OTC=last (manuscript)
    scan = pd.read_csv(SCANNER)

    # group label (control vs OTC) — 05 uses group=='OTC' / status=='control'
    gcol = 'group' if 'group' in df.columns else 'status'
    df['_grp'] = np.where(df[gcol].astype(str).str.upper().str.contains('OTC'), 'OTC', 'control')

    # one selected session per subject -> covariates from scanner csv
    sel = df[['subject_id', 'session', '_grp']].drop_duplicates('subject_id')
    cov_rows = []
    for _, r in sel.iterrows():
        sval = pd.to_numeric(r['session'], errors='coerce')
        ses = f"ses-{int(sval):02d}" if pd.notna(sval) else str(r['session'])
        m = scan[(scan['sub'] == r['subject_id']) & (scan['ses'] == ses)]
        if not len(m):
            print(f"  WARNING: no scanner ROW for {r['subject_id']} {ses} (selected session) -> dropped")
            continue
        m = m.iloc[0]
        if pd.isna(m['scanner']):
            print(f"  WARNING: scanner LABEL missing for {r['subject_id']} {ses} (selected session) -> dropped")
            continue
        cov_rows.append({'subject_id': r['subject_id'], 'group': r['_grp'],
                         'scanner': str(m['scanner']), 'age': m['age'], 'sex': m['sex']})
    cov = pd.DataFrame(cov_rows).set_index('subject_id')

    df_out = df.copy()
    df_out[MEASURES] = df_out[MEASURES].astype(float)   # volume is int64 -> keep harmonized decimals
    for hemi in ['l', 'r']:
        dh = df[df['hemi'] == hemi]
        rois = sorted(dh['category'].unique())
        subs = [sid for sid in cov.index if sid in dh['subject_id'].values]
        bad = [sid for sid in subs if cov.loc[sid, ['scanner', 'age', 'sex']].isna().any()]
        if bad:
            print(f"  [{hemi.upper()}H] dropping {len(bad)} subj w/ incomplete covars: {bad}")
            subs = [sid for sid in subs if sid not in bad]
        if len(subs) < 5:
            continue
        print(f"\n[{hemi.upper()}H] {len(subs)} subjects x {len(rois)} ROIs x {len(MEASURES)} measures")

        # wide matrix [n_subj x (ROI x measure)], winsorized per column within group
        wide, colnames = [], []
        groups = cov.loc[subs, 'group']
        for meas in MEASURES:
            piv = dh.pivot_table(index='subject_id', columns='category', values=meas,
                                 aggfunc='first').reindex(subs)
            for roi in rois:
                col = winsorize_within_group(piv[roi], groups, *WINSOR)
                wide.append(col.values); colnames.append(f'{roi}__{meas}')
        X = np.array(wide, dtype=float).T          # [n_subj x n_feat]

        # NaN -> impute within-group column mean for the FIT; restore NaN after
        nan_mask = np.isnan(X)
        if nan_mask.any():
            print(f"  {int(nan_mask.sum())} NaN cells imputed for fit (originals restored after)")
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
        adj[nan_mask] = np.nan                      # don't fabricate where data was missing

        # diagnostics
        site = cov.loc[subs, 'scanner'].values
        ctrl = (cov.loc[subs, 'group'] == 'control').values
        def gap(M, a, b): return float(np.nanmean(np.abs(np.nanmean(M[a], 0) - np.nanmean(M[b], 0))))
        print(f"  site gap : {gap(X, site=='Verio', site=='Prisma'):.3f} -> {gap(adj, site=='Verio', site=='Prisma'):.3f}")
        print(f"  group gap: {gap(X, ctrl, ~ctrl):.3f} -> {gap(adj, ctrl, ~ctrl):.3f}")

        # write harmonized values back into the long CSV
        for j, name in enumerate(colnames):
            roi, meas = name.split('__')
            for i, sid in enumerate(subs):
                sel_idx = (df_out['subject_id'] == sid) & (df_out['hemi'] == hemi) & (df_out['category'] == roi)
                df_out.loc[sel_idx, meas] = adj[i, j]

    df_out.drop(columns=['_grp'], errors='ignore').to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print("Next: run 05_stats with UNIVAR_CSV pointed at univariate_v1_harmonized.csv "
          "for the with/without comparison.")


if __name__ == '__main__':
    main()