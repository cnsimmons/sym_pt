#!/usr/bin/env python3
"""
02_harmonize.py - all ComBat harmonization for the pipeline, in one script.

Replaces four scripts, kept in 3_harmonization/reference/, and reproduces what they
produced for the committed results. Their code is carried over unchanged except
for paths, the fit-set rule below, and saving the fits.

  univariate  (was combat_05b_harmonize_univariate_sqrt.py)
              -> outputs/D_liu/univariate_v1_harmonized_sqrt.csv
  RSA         (was combat_06_harmonize_rsa.py)
              -> outputs/D_liu/rsa_v1_harmonized.csv
  TFCE maps   (was combat_01_build_matrix.py + combat_02_harmonize.py)
              -> outputs/F_harmonization/combat_inputs/   covars.csv, votc masks, features
              -> outputs/F_harmonization/combat_harmonized/ harmonized_{l,r}.npz, model_{l,r}

ComBat setup, as before: neuroHarmonize harmonizationLearn; SITE = scanner; age, sex
and group preserved; one fit per hemisphere; ROI measures winsorized 5/95 within
group; scanner, age and sex from outputs/F_harmonization/sub_info_scanner.csv.

Fit sets, as committed (they differ by measure, as they did originally):
  univariate, RSA  sub-017 out, sub-108 ses-02 out, nonOTC out, controls' first
                   session, patients' last session: 65 subjects, including the five
                   later excluded from the analyses.
  TFCE maps        lib_tfce.load_subjects(): 60 subjects, first post-op session.

The fits are saved to outputs/F_harmonization/fits/ (univariate and RSA models,
winsorizing limits, fitted subjects); the TFCE models stay in combat_harmonized/.
Existing model files are never overwritten: move them aside to rerun.

Usage:   python 02_harmonize.py                # all three
         python 02_harmonize.py --only rsa     # univariate | rsa | tfce
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn, saveHarmonizationModel

PIPE     = Path('/user_data/csimmon2/git_repos/sym_pt/pipeline')
OUT      = PIPE / 'outputs'
SCANNER  = OUT / 'F_harmonization' / 'sub_info_scanner.csv'
FITS     = OUT / 'F_harmonization' / 'fits'
UNIVAR, UNIVAR_OUT = OUT / 'D_liu' / 'univariate_v1.csv', OUT / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv'
RSA, RSA_OUT       = OUT / 'D_liu' / 'rsa_v1.csv', OUT / 'D_liu' / 'rsa_v1_harmonized.csv'
TFCE_IN, TFCE_OUT  = OUT / 'F_harmonization' / 'combat_inputs', OUT / 'F_harmonization' / 'combat_harmonized'

WINSOR        = (5, 95)
MEASURES      = ['sum_selec_norm', 'volume', 'mean_act', 'peak_z']
SQRT_MEASURES = ['sum_selec_norm', 'volume']       # sqrt before ComBat, left on the sqrt scale
ROI_FIT_EXCLUDE = ['sub-017']                      # the list the committed ROI fit used
TFCE_MODE     = 'others'                           # category-vs-all-others, copes 6-9
SMOOTH_AGE    = False                              # True -> ComBat-GAM smooth age term (TFCE)

sys.path.insert(0, str(PIPE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s = _load('lib_stats', PIPE / '3_harmonization' / 'lib_stats.py')   # apply_exclusions, select_sessions
LIMITS = []                                                           # winsorizing limits, for fits/


# ---------------------------------------------------------------- shared
def roi_fit_rows(df):
    """The committed univariate/RSA fit set: lib_stats' own exclusion and session
    functions, run with the exclusion list the committed fit used."""
    kept = s.EXCLUDE
    s.EXCLUDE = ROI_FIT_EXCLUDE
    try:
        df = s.apply_exclusions(df)              # also drops sub-108 ses-02 (EXCLUDE_SES)
    finally:
        s.EXCLUDE = kept
    return s.select_sessions(df, pt_rule='last')   # controls = first, OTC = last


def winsorize_within_group(col, groups, lo, hi, record=None):
    """Clip each group's values to its [lo, hi] percentiles (lab: by group)."""
    out = col.astype(float).copy()
    for g in groups.unique():
        m = (groups == g) & col.notna()
        if m.sum() < 3:
            continue
        a, b = np.nanpercentile(col[m], [lo, hi])
        out[m] = col[m].clip(a, b)
        if record is not None:
            LIMITS.append({**record, 'group': g, 'lo': a, 'hi': b})
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


def roi_subjects(dh, cov, hemi):
    subs = [sid for sid in cov.index if sid in dh['subject_id'].values]
    bad = [sid for sid in subs if cov.loc[sid, ['scanner', 'age', 'sex']].isna().any()]
    if bad:
        print(f"  [{hemi.upper()}H] dropping {len(bad)} subj w/ incomplete covars: {bad}")
        subs = [sid for sid in subs if sid not in bad]
    return subs


def impute_for_fit(X, groups, label):
    """NaN -> within-group column mean for the FIT; caller restores NaN after."""
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
    return nan_mask


def roi_design(cov, subs):
    sex_d = pd.get_dummies(cov.loc[subs, 'sex'],   prefix='sex',   drop_first=True).astype(int).reset_index(drop=True)
    grp_d = pd.get_dummies(cov.loc[subs, 'group'], prefix='group', drop_first=True).astype(int).reset_index(drop=True)
    design = pd.DataFrame({'SITE': cov.loc[subs, 'scanner'].values,
                           'age':  cov.loc[subs, 'age'].values})
    return pd.concat([design, sex_d, grp_d], axis=1)


def save_fit(model, name, subs, cov, features):
    FITS.mkdir(parents=True, exist_ok=True)
    saveHarmonizationModel(model, str(FITS / name))
    cov.loc[subs].reset_index().to_csv(FITS / f'{name}_subjects.csv', index=False)
    (FITS / f'{name}_features.txt').write_text('\n'.join(features) + '\n')


def gap(M, a, b):
    return float(np.nanmean(np.abs(np.nanmean(M[a], 0) - np.nanmean(M[b], 0))))


# ---------------------------------------------------------------- univariate (was combat_05b)
def _fwd(col, meas):
    """Forward transform for a measure column (no-op unless in SQRT_MEASURES)."""
    if meas not in SQRT_MEASURES:
        return col.astype(float)
    v = col.astype(float)
    if (v < 0).any():
        n = int((v < 0).sum())
        raise ValueError(f"{meas}: {n} negative raw value(s); sqrt undefined. "
                         "Raw values should be >= 0 - investigate extraction.")
    return np.sqrt(v)


def univariate():
    print('\n========== univariate ==========')
    df = roi_fit_rows(pd.read_csv(UNIVAR))
    df, cov = build_covariates(df, pd.read_csv(SCANNER))
    df_out = df.copy()
    df_out[MEASURES] = df_out[MEASURES].astype(float)   # volume is int64 -> keep harmonized decimals

    for hemi in ['l', 'r']:
        dh = df[df['hemi'] == hemi]
        rois = sorted(dh['category'].unique())
        subs = roi_subjects(dh, cov, hemi)
        if len(subs) < 5:
            continue
        print(f"\n[{hemi.upper()}H] {len(subs)} subjects x {len(rois)} ROIs x {len(MEASURES)} measures")

        wide, colnames = [], []
        groups = cov.loc[subs, 'group']
        for meas in MEASURES:
            piv = dh.pivot_table(index='subject_id', columns='category', values=meas,
                                 aggfunc='first').reindex(subs)
            for roi in rois:
                col = _fwd(piv[roi], meas)
                col = winsorize_within_group(col, groups, *WINSOR,
                                             record={'block': 'univariate', 'hemi': hemi,
                                                     'feature': f'{roi}__{meas}'})
                wide.append(col.values); colnames.append(f'{roi}__{meas}')
        X = np.array(wide, dtype=float).T
        nan_mask = impute_for_fit(X, groups, 'univariate')

        model, adj = harmonizationLearn(X, roi_design(cov, subs), smooth_terms=[])
        adj[nan_mask] = np.nan
        save_fit(model, f'univariate_{hemi}', subs, cov, colnames)

        site = cov.loc[subs, 'scanner'].values
        ctrl = (cov.loc[subs, 'group'] == 'control').values
        print(f"  site gap : {gap(X, site=='Verio', site=='Prisma'):.3f} -> {gap(adj, site=='Verio', site=='Prisma'):.3f}")
        print(f"  group gap: {gap(X, ctrl, ~ctrl):.3f} -> {gap(adj, ctrl, ~ctrl):.3f}")
        for meas in SQRT_MEASURES:
            cols = [j for j, nm in enumerate(colnames) if nm.endswith(f'__{meas}')]
            if cols:
                sub_adj = adj[:, cols]
                print(f"  SQRT {meas}: {int(np.nansum(sub_adj < 0))}/{int(np.sum(~np.isnan(sub_adj)))} "
                      f"harmonized cells < 0 on sqrt scale")

        for j, name in enumerate(colnames):
            roi, meas = name.split('__')
            for i, sid in enumerate(subs):
                sel_idx = (df_out['subject_id'] == sid) & (df_out['hemi'] == hemi) & (df_out['category'] == roi)
                df_out.loc[sel_idx, meas] = adj[i, j]

    for meas in SQRT_MEASURES:
        df_out[f'{meas}_scale'] = 'sqrt'
    df_out.drop(columns=['_grp'], errors='ignore').to_csv(UNIVAR_OUT, index=False)
    print(f"\nWrote {UNIVAR_OUT}  ({df_out['subject_id'].nunique()} subjects)")


# ---------------------------------------------------------------- RSA (was combat_06)
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
        col = winsorize_within_group(piv[f], groups, *WINSOR,
                                     record={'block': f'rsa_{label}', 'hemi': hemi, 'feature': f})
        wide.append(col.values); colnames.append(f)
    X = np.array(wide, dtype=float).T
    nan_mask = impute_for_fit(X, groups, label)

    model, adj = harmonizationLearn(X, roi_design(cov, subs), smooth_terms=[])
    adj[nan_mask] = np.nan
    save_fit(model, f'rsa_{label}_{hemi}', subs, cov, colnames)

    site = cov.loc[subs, 'scanner'].values
    ctrl = (cov.loc[subs, 'group'] == 'control').values
    print(f"  [{label}] site gap : {gap(X, site=='Verio', site=='Prisma'):.3f} -> {gap(adj, site=='Verio', site=='Prisma'):.3f}")
    print(f"  [{label}] group gap: {gap(X, ctrl, ~ctrl):.3f} -> {gap(adj, ctrl, ~ctrl):.3f}")

    out = {}
    for j, f in enumerate(colnames):
        for i, sid in enumerate(subs):
            if not nan_mask[i, j]:
                out[(sid, f)] = adj[i, j]
    return out


def rsa():
    print('\n========== RSA ==========')
    df = roi_fit_rows(pd.read_csv(RSA))
    df, cov = build_covariates(df, pd.read_csv(SCANNER))
    df_out = df.copy()
    df_out['liu_distinctiveness'] = df_out['liu_distinctiveness'].astype(float)
    df_out['fisher_r'] = df_out['fisher_r'].astype(float)

    for hemi in ['l', 'r']:
        dh = df[df['hemi'] == hemi]
        subs = roi_subjects(dh, cov, hemi)
        if len(subs) < 5:
            continue
        print(f"\n[{hemi.upper()}H] {len(subs)} subjects")

        dd = dh.dropna(subset=['liu_distinctiveness']).drop_duplicates(['subject_id', 'category'])
        dist_map = harmonize_block(dd, 'liu_distinctiveness', 'category', subs, cov, hemi, 'distinct')
        for (sid, cat), val in dist_map.items():
            m = (df_out['subject_id'] == sid) & (df_out['hemi'] == hemi) & (df_out['category'] == cat)
            df_out.loc[m, 'liu_distinctiveness'] = val

        dg = dh.dropna(subset=['fisher_r']).copy()
        dg['_feat'] = dg['category'].astype(str) + '__' + dg['pair'].astype(str)
        geo_map = harmonize_block(dg, 'fisher_r', '_feat', subs, cov, hemi, 'geometry')
        for (sid, feat), val in geo_map.items():
            cat, pair = feat.split('__', 1)
            m = ((df_out['subject_id'] == sid) & (df_out['hemi'] == hemi)
                 & (df_out['category'] == cat) & (df_out['pair'] == pair))
            df_out.loc[m, 'fisher_r'] = val

    df_out.drop(columns=['_grp'], errors='ignore').to_csv(RSA_OUT, index=False)
    print(f"\nWrote {RSA_OUT}  ({df_out['subject_id'].nunique()} subjects)")


# ---------------------------------------------------------------- TFCE maps (was combat_01 + combat_02)
def tfce_design(cov):
    """neuroHarmonize covars: 'SITE' column + numeric covariates to preserve."""
    design = pd.DataFrame({'SITE': cov['scanner'].values,
                           'age':  cov['age'].values})
    design = pd.concat([
        design,
        pd.get_dummies(cov['sex'],   prefix='sex',   drop_first=True).astype(int),
        pd.get_dummies(cov['group'], prefix='group', drop_first=True).astype(int),
    ], axis=1)
    return design


def tfce():
    print('\n========== TFCE maps ==========')
    v = _load('lib_tfce', PIPE / '2_extraction' / 'lib_tfce.py')
    copes = v.COPES_BY_MODE[TFCE_MODE]          # {'face':6,'house':7,'object':8,'word':9}
    categories, hemis = v.CATEGORIES, v.HEMIS

    # build the feature matrix (was combat_01)
    TFCE_IN.mkdir(parents=True, exist_ok=True)
    v.OUT_DIR = TFCE_IN
    print('Building VOTC masks...')
    masks = v.build_votc_masks_and_save()
    print('Loading subjects (lib_tfce.load_subjects)...')
    subjects = v.load_subjects()
    scan = pd.read_csv(SCANNER)

    rows = []
    for sid, info in subjects.items():
        ses = f"ses-{info['session']}"
        r = scan[(scan['sub'] == sid) & (scan['ses'] == ses)]
        if not len(r):
            print(f'  WARNING: no scanner row for {sid} {ses} -- skipped')
            continue
        r = r.iloc[0]
        rows.append({'subject_id': sid, 'session': info['session'],
                     'group': info['group'], 'scanner': r['scanner'],
                     'age': r['age'], 'sex': r['sex']})
    covars = pd.DataFrame(rows).sort_values('subject_id').reset_index(drop=True)
    covars.to_csv(TFCE_IN / 'covars.csv', index=False)
    print(f"  covars: {len(covars)} subjects | "
          f"scanner={covars.scanner.value_counts().to_dict()} | "
          f"group={covars.group.value_counts().to_dict()}")

    for hemi in hemis:
        mimg = nib.load(str(masks[hemi]))
        mask = mimg.get_fdata().astype(bool)
        nvox = int(mask.sum())
        print(f"\n[{hemi.upper()}H] VOTC = {nvox:,} vox")
        saved = {'mask': mask, 'affine': mimg.affine, 'shape': np.array(mimg.shape)}
        for cat in categories:
            cope = copes[cat]
            X, subs, miss = [], [], []
            for sid in covars['subject_id']:
                info = subjects[sid]
                p = v.get_zstat_path(sid, info['session'], info['first_session'], cope)
                if not p.exists():
                    miss.append(sid); continue
                X.append(nib.load(str(p)).get_fdata()[mask].astype(np.float32))
                subs.append(sid)
            saved[f'X_{cat}']    = np.vstack(X) if X else np.empty((0, nvox), np.float32)
            saved[f'subs_{cat}'] = np.array(subs)
            msg = f"  {cat:7s} cope{cope}: {len(subs)} subj"
            if miss:
                msg += f"  (missing {len(miss)}: {miss[:3]})"
            print(msg)
        np.savez_compressed(TFCE_IN / f'features_{hemi}.npz', **saved)
        print(f"  -> features_{hemi}.npz")

    # harmonize (was combat_02)
    TFCE_OUT.mkdir(parents=True, exist_ok=True)
    cov_all = pd.read_csv(TFCE_IN / 'covars.csv').set_index('subject_id')
    smooth = ['age'] if SMOOTH_AGE else []
    print(f"Covariates preserved: group, age, sex  |  age term: "
          f"{'SMOOTH (ComBat-GAM)' if SMOOTH_AGE else 'linear'}")
    for hemi in ['l', 'r']:
        npz  = np.load(TFCE_IN / f'features_{hemi}.npz', allow_pickle=True)
        subs = list(npz['subs_face'])
        for c in ['face', 'house', 'object', 'word']:
            assert list(npz[f'subs_{c}']) == subs, f'subject mismatch in {c}'
        cov    = cov_all.loc[subs].reset_index()
        design = tfce_design(cov)
        nvox   = npz['X_face'].shape[1]
        data   = np.hstack([npz[f'X_{c}'] for c in ['face', 'house', 'object', 'word']])
        print(f"\n[{hemi.upper()}H] n={len(subs)}  features={data.shape[1]:,}  "
              f"site={cov['scanner'].value_counts().to_dict()}  "
              f"group={cov['group'].value_counts().to_dict()}")

        model, adj = harmonizationLearn(data, design, smooth_terms=smooth)

        out = {'subs': np.array(subs), 'mask': npz['mask'],
               'affine': npz['affine'], 'shape': npz['shape']}
        for i, c in enumerate(['face', 'house', 'object', 'word']):
            out[f'X_{c}'] = adj[:, i*nvox:(i+1)*nvox].astype(np.float32)
        np.savez_compressed(TFCE_OUT / f'harmonized_{hemi}.npz', **out)
        saveHarmonizationModel(model, str(TFCE_OUT / f'model_{hemi}'))
        FITS.mkdir(parents=True, exist_ok=True)
        cov.to_csv(FITS / f'tfce_{hemi}_subjects.csv', index=False)

        site = cov['scanner'].values
        ctrl = (cov['group'] == 'control').values
        def gap_t(M, a, b): return float(np.abs(M[a].mean(0) - M[b].mean(0)).mean())
        print(f"  site  gap (Verio vs Prisma): {gap_t(data, site=='Verio', site=='Prisma'):.4f}"
              f"  ->  {gap_t(adj, site=='Verio', site=='Prisma'):.4f}   (should shrink)")
        print(f"  group gap (ctrl vs patient): {gap_t(data, ctrl, ~ctrl):.4f}"
              f"  ->  {gap_t(adj, ctrl, ~ctrl):.4f}   (should hold)")
        print(f"  -> harmonized_{hemi}.npz, model_{hemi}")


# ---------------------------------------------------------------- main
def existing_models(blocks):
    names = []
    if 'univariate' in blocks:
        names += [FITS / f'univariate_{h}' for h in 'lr']
    if 'rsa' in blocks:
        names += [FITS / f'rsa_{b}_{h}' for b in ('distinct', 'geometry') for h in 'lr']
    if 'tfce' in blocks:
        names += [TFCE_OUT / f'model_{h}' for h in 'lr']
    return [str(p) for p in names if p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['univariate', 'rsa', 'tfce'])
    blocks = [ap.parse_args().only] if ap.parse_args().only else ['univariate', 'rsa', 'tfce']

    found = existing_models(blocks)
    if found:
        sys.exit('STOPPED, nothing run: these model files already exist; move them aside first:\n  '
                 + '\n  '.join(found))

    if 'univariate' in blocks:
        univariate()
    if 'rsa' in blocks:
        rsa()
    if LIMITS:
        FITS.mkdir(parents=True, exist_ok=True)
        lim = FITS / 'winsor_limits.csv'
        new = pd.DataFrame(LIMITS)
        if lim.exists():
            old = pd.read_csv(lim)
            new = pd.concat([old[~old['block'].isin(new['block'].unique())], new], ignore_index=True)
        new.to_csv(lim, index=False)
    if 'tfce' in blocks:
        tfce()
    print('\nDone.')


if __name__ == '__main__':
    main()
