#!/usr/bin/env python3
"""
05_statistics.py — unified cross-sectional statistics for all measures.

Consumes the extractor CSVs (01/03/04 + 05_calc_peak_coords) and the TFCE
output dir; runs no extraction. One long-format results CSV + a TFCE cluster
table, both written to D_liu/.

Session selection (deferred to the cross-sectional notebook's rule):
  - controls  -> FIRST session (min ses_num)
  - OTC pts   -> LAST  session (max ses_num)
Exclusions (notebook): sub-017 entirely; (sub-108, ses-2).
nonOTC never enter (filtered by group=='OTC' / status=='control').

Framework (locked):
  - LMM omnibus (joint Wald chi2) on every measure.
      * GATING for WTA (category x group, df=3) and geometry (pair x group, df=5)
        -- categories/pairs are coupled within a unit.
      * REPORTED-BUT-NOT-GATING for peak/sum-sel/distinctiveness -- category==ROI,
        so the omnibus is a family-wise context test, not a profile test; per-ROI
        FDR results stand on their own.
  - Per-level post-hoc: permutation (10,000). Independent = label-shuffle;
    paired (control L-vs-R) = sign-flip. Two-tailed.
  - Effect size: Cohen's d (pooled SD); sum-sel additionally reports delta(log10).
  - Bootstrap 95% CI on the effect size (subject resampling).
  - Multiple comparisons: BH-FDR within each measure's PATIENT-vs-CONTROL family.
    Control L-vs-R asymmetry tests are corrected as a SEPARATE family.
    (BY available via FDR_METHOD; matches notebook fdr_correct.)

Usage:
  python 05_statistics.py [--fdr bh|by] [--n-perm 10000] [--n-boot 5000]
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, mannwhitneyu
import statsmodels.formula.api as smf

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from params import processed_dir

# ── Paths ────────────────────────────────────────────────────────────────────
D_LIU       = Path('/user_data/csimmon2/git_repos/sym_pt/D_liu')
UNIVAR_CSV  = D_LIU / 'univariate_v1.csv'
RSA_CSV     = D_LIU / 'rsa_v1.csv'
WTA_CSV     = Path(processed_dir) / 'group_results' / 'wta_percentages.csv'
PEAK_MNI    = Path(processed_dir) / 'group_results' / 'peak_coords' / 'peak_coords_mni.csv'
TFCE_DIR    = Path(processed_dir) / 'group_results' / 'tfce_votc_fdr'

OUT_RESULTS = D_LIU / 'stats_results.csv'
OUT_TFCE    = D_LIU / 'tfce_clusters.csv'

# ── Notebook constants ───────────────────────────────────────────────────────
EXCLUDE     = ['sub-017']
EXCLUDE_SES = [('sub-108', 2)]

PRIMARY_ROIS = ['face_FFA', 'house_PPA', 'object_LOC', 'word_VWFA']
CATEGORIES   = ['face', 'house', 'object', 'word']
PAIRS        = ['face-house', 'face-object', 'face-word',
                'house-object', 'house-word', 'object-word']

# OTC voxel counts per hemisphere (manuscript §75) for % OTC in TFCE table.
OTC_VOXELS = {'l': 11340, 'r': 11540}

# Surviving TFCE clusters (cat, hemi, tstat). tstat1=ctrl>pt, tstat2=pt>ctrl.
TFCE_CLUSTERS = [('object', 'l', 1), ('house', 'r', 1), ('word', 'r', 2)]

N_PERM = 10000
N_BOOT = 5000
FDR_METHOD = 'bh'
RNG = np.random.RandomState(42)

# =============================================================================
# Shared helpers
# =============================================================================
def apply_exclusions(df):
    df = df[~df['subject_id'].isin(EXCLUDE)].copy()
    df['ses_num'] = pd.to_numeric(df['session'], errors='coerce').astype('Int64')
    for bad_sub, bad_ses in EXCLUDE_SES:
        df = df[~((df['subject_id'] == bad_sub) & (df['ses_num'] == bad_ses))]
    return df

def select_sessions(df, pt_rule='last'):
    """Controls -> first session. OTC patients -> last ('last') or first
    ('first') session. WTA reproduces the manuscript at first-post, so it
    passes pt_rule='first'; the other measures use the notebook's last-session
    rule."""
    ctrl = df[df['status'] == 'control'].copy()
    fs = ctrl.groupby('subject_id')['ses_num'].min().rename('keep').reset_index()
    ctrl = ctrl.merge(fs, on='subject_id')
    ctrl = ctrl[ctrl['ses_num'] == ctrl['keep']].drop(columns='keep')

    pt = df[df['group'] == 'OTC'].copy()
    agg = 'min' if pt_rule == 'first' else 'max'
    ls = pt.groupby('subject_id')['ses_num'].agg(agg).rename('keep').reset_index()
    pt = pt.merge(ls, on='subject_id')
    pt = pt[pt['ses_num'] == pt['keep']].drop(columns='keep')
    return pd.concat([ctrl, pt], ignore_index=True)

def fdr_bh_by(pvals, method='bh', alpha=0.05):
    """Benjamini-Hochberg ('bh') or Benjamini-Yekutieli ('by') q-values.
    Returns (q_values, sig_mask). NaNs preserved."""
    p = np.asarray(pvals, float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[valid]
    n = len(pv)
    if n == 0:
        return q, np.zeros_like(p, bool)
    order = np.argsort(pv)
    ranked = pv[order]
    c_n = np.sum(1.0 / np.arange(1, n + 1)) if method == 'by' else 1.0
    # step-up q-values
    q_ranked = ranked * n * c_n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_valid = np.empty(n)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    sig = np.zeros_like(p, bool)
    sig[valid] = q_valid <= alpha
    return q, sig

def cohens_d_independent(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

def cohens_d_paired(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)

def perm_independent(a, b, n_perm=N_PERM, rng=RNG):
    """Two-tailed label-shuffle permutation on mean difference (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pool))
        null[i] = pool[idx[:na]].mean() - pool[idx[na:]].mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p

def perm_paired(diff, n_perm=N_PERM, rng=RNG):
    """Two-tailed sign-flip permutation on the mean of paired differences."""
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return np.nan, np.nan
    obs = diff.mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        null[i] = (diff * signs).mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p

def boot_ci_independent(a, b, n_boot=N_BOOT, rng=RNG):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ds[i] = cohens_d_independent(rng.choice(a, len(a), replace=True),
                                     rng.choice(b, len(b), replace=True))
    ds = ds[~np.isnan(ds)]
    if len(ds) == 0:
        return np.nan, np.nan
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def boot_ci_paired(diff, n_boot=N_BOOT, rng=RNG):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return np.nan, np.nan
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ds[i] = cohens_d_paired(rng.choice(diff, len(diff), replace=True))
    ds = ds[~np.isnan(ds)]
    if len(ds) == 0:
        return np.nan, np.nan
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def lmm_omnibus(df, value_col, factor_col, group_col):
    """Fit value ~ factor*group + (1|sid); joint Wald chi2 on the interaction.
    Returns (chi2, df, p, mse). df must contain subject_id."""
    d = df[['subject_id', value_col, factor_col, group_col]].dropna().copy()
    d = d.rename(columns={value_col: 'y', factor_col: 'f', group_col: 'g'})
    if d['g'].nunique() < 2 or d['f'].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    try:
        m = smf.mixedlm('y ~ C(f) * C(g)', d, groups=d['subject_id']).fit(reml=True)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    inter = [i for i, n in enumerate(m.params.index) if ':' in n]
    if not inter:
        return np.nan, np.nan, np.nan, float(m.scale)
    beta = m.params.values[inter]
    V = m.cov_params().values[np.ix_(inter, inter)]
    try:
        W = float(beta @ np.linalg.solve(V, beta))
    except np.linalg.LinAlgError:
        W = float(beta @ np.linalg.pinv(V) @ beta)
    dfree = len(inter)
    return W, dfree, float(chi2_dist.sf(W, dfree)), float(m.scale)

def row(**kw):
    base = dict(measure=None, model=None, comparison=None, hemi=None, level=None,
                omnibus_chi2=np.nan, omnibus_df=np.nan, omnibus_p=np.nan, mse=np.nan,
                gating=None, diff=np.nan, cohens_d=np.nan, delta_log10=np.nan,
                p_perm=np.nan, q_fdr=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                n_a=np.nan, n_b=np.nan, note='')
    base.update(kw)
    return base

# =============================================================================
# Per-ROI scalar measures (peak, sum-sel, distinctiveness)
# =============================================================================
def scalar_measure(df_long, value_col, measure, rois, paired_log=False,
                   results=None, fdr_method='bh'):
    """Patient-vs-control per ROI (FDR family across rois) + control L-vs-R
    (separate FDR family) + reported-not-gating LMM omnibus per hemi group."""
    # intact hemi per patient: 'intact' label rows; controls have l/r.
    for grp_hemi, hemi_label in [('l', 'LH-intact'), ('r', 'RH-intact')]:
        # --- patient vs control, per ROI ---
        pc_rows, pvals = [], []
        for roi in rois:
            pt = df_long[(df_long['group'] == 'OTC') & (df_long['category'] == roi) &
                         (df_long['hemi'] == grp_hemi)][value_col].dropna().values
            ct = df_long[(df_long['status'] == 'control') & (df_long['category'] == roi) &
                         (df_long['hemi'] == grp_hemi)][value_col].dropna().values
            if len(pt) < 2 or len(ct) < 3:
                continue
            obs, p = perm_independent(pt, ct)
            d = cohens_d_independent(pt, ct)
            lo, hi = boot_ci_independent(pt, ct)
            r = row(measure=measure, model=f'{hemi_label}_pt_vs_ctrl',
                    comparison='patient_vs_control', hemi=grp_hemi, level=roi,
                    gating='not_gating', diff=obs, cohens_d=d, ci_lo=lo, ci_hi=hi,
                    p_perm=p, n_a=len(pt), n_b=len(ct))
            if measure == 'sum_selectivity':
                r['delta_log10'] = obs
            pc_rows.append(r); pvals.append(p)
        q, _ = fdr_bh_by(pvals, fdr_method)
        for r, qi in zip(pc_rows, q):
            r['q_fdr'] = qi
        results.extend(pc_rows)

        # --- reported-but-not-gating LMM omnibus (category==ROI) ---
        sub = df_long[(df_long['category'].isin(rois)) &
                      (((df_long['group'] == 'OTC') & (df_long['hemi'] == grp_hemi)) |
                       ((df_long['status'] == 'control') & (df_long['hemi'] == grp_hemi)))].copy()
        sub['grp'] = np.where(sub['group'] == 'OTC', 'pt', 'ctrl')
        chi, dfree, pomni, mse = lmm_omnibus(sub, value_col, 'category', 'grp')
        results.append(row(measure=measure, model=f'{hemi_label}_pt_vs_ctrl',
                           comparison='omnibus', hemi=grp_hemi, level='ALL',
                           gating='not_gating', omnibus_chi2=chi, omnibus_df=dfree,
                           omnibus_p=pomni, mse=mse,
                           note='category==ROI: family-wise context, not a profile test'))

    # --- control L vs R (paired), SEPARATE FDR family ---
    ca_rows, pvals = [], []
    ctrl = df_long[df_long['status'] == 'control']
    for roi in rois:
        wide = ctrl[ctrl['category'] == roi].pivot_table(
            index='subject_id', columns='hemi', values=value_col, aggfunc='first')
        if 'l' not in wide or 'r' not in wide:
            continue
        diff = (wide['l'] - wide['r']).dropna().values
        if len(diff) < 2:
            continue
        obs, p = perm_paired(diff)
        d = cohens_d_paired(diff)
        lo, hi = boot_ci_paired(diff)
        r = row(measure=measure, model='ctrl_LvsR', comparison='control_asymmetry',
                hemi='lr', level=roi, gating='separate_family', diff=obs,
                cohens_d=d, ci_lo=lo, ci_hi=hi, p_perm=p, n_a=len(diff))
        if measure == 'sum_selectivity':
            r['delta_log10'] = obs
        ca_rows.append(r); pvals.append(p)
    q, _ = fdr_bh_by(pvals, fdr_method)
    for r, qi in zip(ca_rows, q):
        r['q_fdr'] = qi
    results.extend(ca_rows)

    # control asymmetry omnibus (own context)
    co = ctrl[ctrl['category'].isin(rois)]
    chi, dfree, pomni, mse = lmm_omnibus(co, value_col, 'category', 'hemi')
    results.append(row(measure=measure, model='ctrl_LvsR', comparison='omnibus',
                       hemi='lr', level='ALL', gating='separate_family',
                       omnibus_chi2=chi, omnibus_df=dfree, omnibus_p=pomni, mse=mse))

# =============================================================================
# Peak distance (MNI, 2D x-y) — special input (peak_coords_mni.csv)
# =============================================================================
def peak_distance(results, fdr_method='bh'):
    if not PEAK_MNI.exists():
        results.append(row(measure='peak_distance', note=f'MISSING input: {PEAK_MNI}'))
        return
    mni = apply_exclusions(pd.read_csv(PEAK_MNI))
    mni = select_sessions(mni, pt_rule='last')   # uniform last-session (Liu)
    for grp_hemi, hemi_label in [('l', 'LH-intact'), ('r', 'RH-intact')]:
        pc_rows, pvals = [], []
        for roi in PRIMARY_ROIS:
            c = mni[(mni['status'] == 'control') & (mni['category'] == roi) &
                    (mni['hemi'] == grp_hemi)].drop_duplicates('subject_id')[
                ['peak_x_mni', 'peak_y_mni']].dropna().values
            if len(c) < 3:
                continue
            cen = c.mean(0)
            ct_d = np.linalg.norm(c - cen, axis=1)
            p = mni[(mni['group'] == 'OTC') & (mni['category'] == roi) &
                    (mni['hemi'] == grp_hemi)].drop_duplicates('subject_id')[
                ['peak_x_mni', 'peak_y_mni']].dropna().values
            if len(p) < 2:
                continue
            pt_d = np.linalg.norm(p - cen, axis=1)
            obs, pval = perm_independent(pt_d, ct_d)
            d = cohens_d_independent(pt_d, ct_d)
            lo, hi = boot_ci_independent(pt_d, ct_d)
            pc_rows.append(row(measure='peak_distance', model=f'{hemi_label}_pt_vs_ctrl',
                               comparison='patient_vs_control', hemi=grp_hemi, level=roi,
                               gating='not_gating', diff=obs, cohens_d=d, ci_lo=lo, ci_hi=hi,
                               p_perm=pval, n_a=len(pt_d), n_b=len(ct_d)))
            pvals.append(pval)
        q, _ = fdr_bh_by(pvals, fdr_method)
        for r, qi in zip(pc_rows, q):
            r['q_fdr'] = qi
        results.extend(pc_rows)

# =============================================================================
# WTA composition — 4 LMM models, omnibus GATES per-category post-hocs
# =============================================================================
def wta_composition(results, fdr_method='bh'):
    wta = apply_exclusions(pd.read_csv(WTA_CSV))
    wta = wta[(wta['region'] == 'otc') & (wta['denominator'] == 'selective')].copy()
    wta = select_sessions(wta, pt_rule='last')    # uniform last-session (Liu)

    def lmm_and_posthoc(sub, factor_col, model_name, comparison, paired=False):
        chi, dfree, pomni, mse = lmm_omnibus(sub, 'wta_pct', 'category', factor_col)
        results.append(row(measure='wta', model=model_name, comparison='omnibus',
                           hemi='', level='ALL', gating='gates_posthoc',
                           omnibus_chi2=chi, omnibus_df=dfree, omnibus_p=pomni, mse=mse))
        gated_off = not (pomni < 0.05)
        pc_rows, pvals = [], []
        levels = sorted(sub[factor_col].unique())
        a_lev, b_lev = levels[0], levels[1]
        for cat in CATEGORIES:
            av = sub[(sub[factor_col] == a_lev) & (sub['category'] == cat)]
            bv = sub[(sub[factor_col] == b_lev) & (sub['category'] == cat)]
            if paired:
                wide = sub[sub['category'] == cat].pivot_table(
                    index='subject_id', columns=factor_col, values='wta_pct', aggfunc='first')
                if a_lev not in wide or b_lev not in wide:
                    continue
                diff = (wide[a_lev] - wide[b_lev]).dropna().values
                obs, p = perm_paired(diff)
                d = cohens_d_paired(diff); lo, hi = boot_ci_paired(diff)
                na = len(diff); nb = len(diff)
            else:
                a = av['wta_pct'].dropna().values; b = bv['wta_pct'].dropna().values
                obs, p = perm_independent(a, b)
                d = cohens_d_independent(a, b); lo, hi = boot_ci_independent(a, b)
                na, nb = len(a), len(b)
            pc_rows.append(row(measure='wta', model=model_name, comparison=comparison,
                               hemi='', level=cat, gating='gated' if not gated_off else 'omnibus_ns',
                               diff=obs, cohens_d=d, ci_lo=lo, ci_hi=hi, p_perm=p,
                               n_a=na, n_b=nb,
                               note='' if not gated_off else 'omnibus n.s.; post-hoc exploratory'))
            pvals.append(p)
        results.extend(pc_rows)
        return pc_rows, pvals

    # Model 1: LH ctrl vs LH pt
    m1 = wta[((wta['group'] == 'OTC') & (wta['hemi'] == 'l')) |
             ((wta['status'] == 'control') & (wta['hemi'] == 'l'))].copy()
    m1['grp'] = np.where(m1['group'] == 'OTC', 'pt', 'ctrl')
    m1_rows, m1_p = lmm_and_posthoc(m1, 'grp', 'M1_LH_ctrl_vs_pt', 'patient_vs_control')

    # Model 2: RH ctrl vs RH pt
    m2 = wta[((wta['group'] == 'OTC') & (wta['hemi'] == 'r')) |
             ((wta['status'] == 'control') & (wta['hemi'] == 'r'))].copy()
    m2['grp'] = np.where(m2['group'] == 'OTC', 'pt', 'ctrl')
    m2_rows, m2_p = lmm_and_posthoc(m2, 'grp', 'M2_RH_ctrl_vs_pt', 'patient_vs_control')

    # FDR family = patient-vs-control = M1 + M2 pooled (the locked scope).
    pooled_rows = m1_rows + m2_rows
    pooled_q, _ = fdr_bh_by(m1_p + m2_p, fdr_method)
    for r, qi in zip(pooled_rows, pooled_q):
        r['q_fdr'] = qi

    # Model 3: LH pt vs RH pt (subgroup) — separate family
    m3 = wta[wta['group'] == 'OTC'].copy()
    m3['intact'] = np.where(m3['hemi'] == 'l', 'LH', 'RH')
    m3_rows, m3_p = lmm_and_posthoc(m3, 'intact', 'M3_LH_vs_RH_pt', 'patient_subgroup')
    q3, _ = fdr_bh_by(m3_p, fdr_method)
    for r, qi in zip(m3_rows, q3):
        r['q_fdr'] = qi

    # Model 4: ctrl L vs R (paired) — control-asymmetry family, separate
    m4 = wta[wta['status'] == 'control'].copy()
    m4_rows, m4_p = lmm_and_posthoc(m4, 'hemi', 'M4_ctrl_LvsR', 'control_asymmetry', paired=True)
    q4, _ = fdr_bh_by(m4_p, fdr_method)
    for r, qi in zip(m4_rows, q4):
        r['q_fdr'] = qi

    # Overall selective proportion: Mann-Whitney per hemisphere (denominator='total')
    tot = apply_exclusions(pd.read_csv(WTA_CSV))
    tot = tot[(tot['region'] == 'otc') & (tot['denominator'] == 'total') &
              (tot['category'] == 'non-selective')].copy()
    tot = select_sessions(tot, pt_rule='last')
    tot['sel_prop'] = 100.0 - tot['wta_pct']  # selective = 100 - non-selective
    for hemi, label in [('l', 'LH'), ('r', 'RH')]:
        pt = tot[(tot['group'] == 'OTC') & (tot['hemi'] == hemi)]['sel_prop'].dropna()
        ct = tot[(tot['status'] == 'control') & (tot['hemi'] == hemi)]['sel_prop'].dropna()
        if len(pt) < 2 or len(ct) < 2:
            continue
        U, p = mannwhitneyu(ct, pt, alternative='two-sided')
        results.append(row(measure='wta_selective_prop', model=f'{label}_pt_vs_ctrl',
                           comparison='patient_vs_control', hemi=hemi, level='selective',
                           diff=float(ct.median() - pt.median()), p_perm=p,
                           n_a=len(pt), n_b=len(ct),
                           note=f'Mann-Whitney U={U:.0f}; medians ctrl/pt'))

    # Within-cluster composition — descriptive only
    cl = apply_exclusions(pd.read_csv(WTA_CSV))
    cl = cl[cl['region'].str.startswith('cluster_')].copy()
    cl = select_sessions(cl, pt_rule='last')
    for region in sorted(cl['region'].unique()):
        for cat in CATEGORIES:
            for status, lab in [('control', 'ctrl'), ('patient', 'pt')]:
                v = cl[(cl['region'] == region) & (cl['category'] == cat) &
                       (cl['status'] == status)]['wta_pct'].dropna()
                if len(v) == 0:
                    continue
                results.append(row(measure='wta_within_cluster', model=region,
                                   comparison='descriptive', level=f'{cat}_{lab}',
                                   diff=float(v.mean()), n_a=len(v),
                                   note='descriptive mean % (no test)'))

# =============================================================================
# Between-category geometry — per-ROI LMM (pair x group, df=5) GATES per-pair
# =============================================================================
def geometry(results, fdr_method='bh'):
    rsa = apply_exclusions(pd.read_csv(RSA_CSV))
    rsa = select_sessions(rsa, pt_rule='last')  # RSA = last session (matches notebook); keeps pair/fisher_r

    def per_roi(sub_roi, factor_col, model_name, comparison, paired=False):
        chi, dfree, pomni, mse = lmm_omnibus(sub_roi, 'fisher_r', 'pair', factor_col)
        results.append(row(measure='geometry', model=model_name, comparison='omnibus',
                           hemi='', level='ALL', gating='gates_posthoc',
                           omnibus_chi2=chi, omnibus_df=dfree, omnibus_p=pomni, mse=mse))
        gated_off = not (pomni < 0.05)
        pc_rows, pvals = [], []
        levels = sorted(sub_roi[factor_col].unique())
        a_lev, b_lev = levels[0], levels[1]
        for pair in PAIRS:
            if paired:
                wide = sub_roi[sub_roi['pair'] == pair].pivot_table(
                    index='subject_id', columns=factor_col, values='fisher_r', aggfunc='first')
                if a_lev not in wide or b_lev not in wide:
                    continue
                diff = (wide[a_lev] - wide[b_lev]).dropna().values
                obs, p = perm_paired(diff)
                d = cohens_d_paired(diff); lo, hi = boot_ci_paired(diff)
                na = nb = len(diff)
            else:
                a = sub_roi[(sub_roi[factor_col] == a_lev) & (sub_roi['pair'] == pair)]['fisher_r'].dropna().values
                b = sub_roi[(sub_roi[factor_col] == b_lev) & (sub_roi['pair'] == pair)]['fisher_r'].dropna().values
                obs, p = perm_independent(a, b)
                d = cohens_d_independent(a, b); lo, hi = boot_ci_independent(a, b)
                na, nb = len(a), len(b)
            pc_rows.append(row(measure='geometry', model=model_name, comparison=comparison,
                               level=pair, gating='gated' if not gated_off else 'omnibus_ns',
                               diff=obs, cohens_d=d, ci_lo=lo, ci_hi=hi, p_perm=p,
                               n_a=na, n_b=nb,
                               note='' if not gated_off else 'omnibus n.s.; not interpreted'))
            pvals.append(p)
        q, _ = fdr_bh_by(pvals, fdr_method)  # within ROI x hemi
        for r, qi in zip(pc_rows, q):
            r['q_fdr'] = qi
        results.extend(pc_rows)

    for roi in PRIMARY_ROIS:
        for grp_hemi, hemi_label in [('l', 'LH-intact'), ('r', 'RH-intact')]:
            sub = rsa[(rsa['category'] == roi) &
                      (((rsa['group'] == 'OTC') & (rsa['hemi'] == grp_hemi)) |
                       ((rsa['status'] == 'control') & (rsa['hemi'] == grp_hemi)))].copy()
            sub['grp'] = np.where(sub['group'] == 'OTC', 'pt', 'ctrl')
            if sub['grp'].nunique() == 2:
                per_roi(sub, 'grp', f'{roi}_{hemi_label}_pt_vs_ctrl', 'patient_vs_control')
        # control L vs R for this ROI (paired)
        cc = rsa[(rsa['category'] == roi) & (rsa['status'] == 'control')].copy()
        if cc['hemi'].nunique() == 2:
            per_roi(cc, 'hemi', f'{roi}_ctrl_LvsR', 'control_asymmetry', paired=True)

# =============================================================================
# TFCE — harvest already-corrected maps into a cluster table (no test)
# =============================================================================
def tfce_clusters():
    import nibabel as nib
    from scipy.ndimage import label as nd_label
    rows = []
    mni_affine = None
    for cat, hemi, tstat in TFCE_CLUSTERS:
        corrp_f = TFCE_DIR / f'{cat}_{hemi}_pt_vs_ctrl' / f'rand_tfce_corrp_tstat{tstat}.nii.gz'
        tstat_f = TFCE_DIR / f'{cat}_{hemi}_pt_vs_ctrl' / f'rand_tstat{tstat}.nii.gz'
        if not corrp_f.exists():
            rows.append({'category': cat, 'hemi': hemi, 'note': f'missing {corrp_f}'})
            continue
        corrp_img = nib.load(corrp_f)
        corrp = corrp_img.get_fdata()
        tmap = nib.load(tstat_f).get_fdata() if tstat_f.exists() else None
        sig = corrp > 0.95
        lab, n = nd_label(sig)
        for c in range(1, n + 1):
            m = lab == c
            nvox = int(m.sum())
            # peak = voxel with max (1 - p) = max corrp
            idx = np.unravel_index(np.argmax(np.where(m, corrp, -np.inf)), corrp.shape)
            peak_mni = nib.affines.apply_affine(corrp_img.affine, np.array(idx))
            rows.append({
                'category': cat, 'hemi': hemi,
                'direction': 'ctrl>pt' if tstat == 1 else 'pt>ctrl',
                'n_voxels': nvox, 'volume_mm3': nvox * 8,  # 2mm iso
                'pct_otc': round(100.0 * nvox / OTC_VOXELS[hemi], 2),
                'peak_t': round(float(tmap[idx]), 2) if tmap is not None else np.nan,
                'peak_x_mni': round(float(peak_mni[0]), 1),
                'peak_y_mni': round(float(peak_mni[1]), 1),
                'peak_z_mni': round(float(peak_mni[2]), 1),
                'max_corrp': round(float(corrp[idx]), 4),
            })
    return pd.DataFrame(rows)

# =============================================================================
# Main
# =============================================================================
def main():
    global N_PERM, N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--fdr', choices=['bh', 'by'], default=FDR_METHOD)
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--univar', default=str(UNIVAR_CSV))   # swap to univariate_v1_harmonized.csv for the harmonized run
    ap.add_argument('--tag', default='')                   # suffix appended to output filenames (e.g. _harmonized)
    args = ap.parse_args()
    N_PERM, N_BOOT = args.n_perm, args.n_boot

    results = []

    # ── peak (special MNI input) ──
    print('Peak distance...')
    peak_distance(results, args.fdr)

    # ── sum-selectivity (univariate CSV, log10) ──
    print(f'Sum-selectivity...  (univar: {args.univar})')
    uni = apply_exclusions(pd.read_csv(args.univar))
    uni = uni[uni['group'] != 'nonOTC']          # 01 includes nonOTC; drop here
    uni = select_sessions(uni, pt_rule='last')   # uniform last-session (Liu)
    uni = uni[uni['sum_selec_norm'] > 0].copy()  # log10 needs positive
    uni['log_sumsel'] = np.log10(uni['sum_selec_norm'])
    scalar_measure(uni, 'log_sumsel', 'sum_selectivity', PRIMARY_ROIS,
                   results=results, fdr_method=args.fdr)

    # ── distinctiveness (rsa CSV, per-ROI scalar) ──
    print('Distinctiveness...')
    rsa = apply_exclusions(pd.read_csv(RSA_CSV))
    rsa_summary = rsa.drop(columns=['pair', 'fisher_r']).drop_duplicates()
    rsa_summary = select_sessions(rsa_summary, pt_rule='last')  # RSA = last session (matches notebook)
    scalar_measure(rsa_summary, 'liu_distinctiveness', 'distinctiveness', PRIMARY_ROIS,
                   results=results, fdr_method=args.fdr)

    # ── WTA ──
    print('WTA composition...')
    wta_composition(results, args.fdr)

    # ── geometry ──
    print('Between-category geometry...')
    geometry(results, args.fdr)

    # ── write results ──
    out_results = OUT_RESULTS.with_name(f'{OUT_RESULTS.stem}{args.tag}.csv')
    out_tfce    = OUT_TFCE.with_name(f'{OUT_TFCE.stem}{args.tag}.csv')
    df = pd.DataFrame(results)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_results, index=False)
    print(f'\nSaved: {out_results} ({len(df)} rows)')

    # ── TFCE harvest ──
    print('TFCE cluster table...')
    tdf = tfce_clusters()
    tdf.to_csv(out_tfce, index=False)
    print(f'Saved: {out_tfce} ({len(tdf)} clusters)')

if __name__ == '__main__':
    main()