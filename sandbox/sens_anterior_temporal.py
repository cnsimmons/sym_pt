#!/usr/bin/env python3
"""
sens_anterior_temporal.py - do the results change without the anterior temporal cases?

Cohorts, all on the paper's inputs (P2_nosmooth_c20: no smoothing, word ROIs cope 20):
  A1  record exclusions + sub-010                               (the paper's cohort)
  S3  A1 + sub-039 (KQ), sub-075 (ED), sub-076 (JH)             (documented anterior temporal)
  S5  S3 + sub-078 (IT), sub-082 (KB)                           (surgeries unchecked)
IDs-to-initials and surgery types: OTC_classification_note (not re-checked).

Runs D_liu/verified/05_stats_harmony.py UNCHANGED, the way sandbox/rerun_peaks_wta.py does
(fresh module per cohort, EXCLUDE / EXCLUDE_SES and input paths set on the module).
With statsmodels (cluster, fmri env): full run, LMM omnibus with age.
Without statsmodels (offline): LMM cells come out blank; every permutation cell is exact.
Writes only into <base>/sandbox/sens_AT/.

  python sens_anterior_temporal.py [--base /user_data/csimmon2/git_repos/sym_pt]
"""
import argparse
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

BASE_EXCL = ['sub-017', 'sub-091', 'sub-095', 'sub-096', 'sub-027', 'sub-084']
AT3 = ['sub-039', 'sub-075', 'sub-076']
AT2 = ['sub-078', 'sub-082']
COHORTS = {'A1': BASE_EXCL + ['sub-010'],
           'S3': BASE_EXCL + ['sub-010'] + AT3,
           'S5': BASE_EXCL + ['sub-010'] + AT3 + AT2}
KEYS = ['measure', 'model', 'comparison', 'hemi', 'level']


def _have_statsmodels():
    try:
        import statsmodels.formula.api  # noqa: F401
        return True
    except ImportError:
        return False


def _stub_statsmodels():
    """Offline only: lets the stats script import; every mixedlm call raises, which
    lmm_omnibus already turns into NaN. Permutation tests never touch it."""
    def _raise(*a, **k):
        raise RuntimeError('statsmodels not available offline')
    sm = types.ModuleType('statsmodels'); f = types.ModuleType('statsmodels.formula')
    api = types.ModuleType('statsmodels.formula.api'); api.mixedlm = _raise
    sm.formula, f.api = f, api
    sys.modules.update({'statsmodels': sm, 'statsmodels.formula': f, 'statsmodels.formula.api': api})


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run_stats(base, out, offline):
    stats_py = base / 'D_liu' / 'verified' / '05_stats_harmony.py'
    hz, more = base / 'sandbox' / 'rerun_harmonize' / 'new20', base / 'sandbox' / 'rerun_more'
    uni, rsa = hz / 'univariate_v1_harmonized_sqrt.csv', hz / 'rsa_v1_harmonized.csv'
    sys.path.insert(0, str(base))
    files = {}
    for c, excl in COHORTS.items():
        mod = _load(stats_py, f'stats_P2_{c}')                       # fresh module, fresh RNG
        mod.UNIVAR_CSV, mod.RSA_CSV = uni, rsa
        mod.PEAK_MNI = more / 'peaks_P2_nosmooth_c20.csv'
        mod.WTA_CSV = more / 'wta_nosmooth.csv'
        mod.EXCLUDE, mod.EXCLUDE_SES = list(excl), [('sub-108', 2)]
        mod.OUT_RESULTS, mod.OUT_TFCE = out / f'stats_P2__{c}.csv', out / f'tfce_P2__{c}.csv'
        if offline:   # no sub_info ages and no TFCE maps here; neither feeds a permutation test
            mod._age_table = lambda: pd.DataFrame(columns=['subject_id', 'ses_num', 'age'])
            mod.tfce_clusters = lambda: pd.DataFrame()
        argv, sys.argv = sys.argv, [str(stats_py)]
        try:
            with open(out / f'stats_P2__{c}.log', 'w') as log, contextlib.redirect_stdout(log):
                mod.main()
        finally:
            sys.argv = argv
        files[c] = mod.OUT_RESULTS
        print(f'stats {c}: {len(pd.read_csv(files[c]))} rows')
    return files


def inclusion(base, out):
    """Which of the five are in the analysed data, per measure, using the stats script's
    own exclusion and session rules (A1)."""
    mod = sys.modules.get('stats_P2_A1') or _load(base / 'D_liu' / 'verified' / '05_stats_harmony.py', 'incl')
    mod._age_table = lambda: pd.DataFrame(columns=['subject_id', 'ses_num', 'age'])
    mod.EXCLUDE, mod.EXCLUDE_SES = list(COHORTS['A1']), [('sub-108', 2)]
    hz = base / 'sandbox' / 'rerun_harmonize' / 'new20'
    u = mod.select_sessions(mod.apply_exclusions(pd.read_csv(hz / 'univariate_v1_harmonized_sqrt.csv')), 'last')
    r = mod.select_sessions(mod.apply_exclusions(pd.read_csv(hz / 'rsa_v1_harmonized.csv')), 'last')
    rows = []
    for s in AT3 + AT2:
        uu, rr = u[u.subject_id == s], r[r.subject_id == s]
        prim = ['word_VWFA', 'face_FFA', 'house_PPA_strict', 'object_LOC']
        rows.append(dict(subject=s,
                         intact=uu.intact_hemi.iloc[0] if len(uu) else '',
                         session=int(uu.ses_num.iloc[0]) if len(uu) else None,
                         peak_rois=int(uu[uu.category.isin(prim)].peak_z.notna().sum()),
                         voxel_rois=int((uu[uu.category.isin(prim)].volume >= 0).sum()),
                         pref_sim_rois=int(rr[rr.category.isin(prim)].drop_duplicates('category')
                                           .liu_distinctiveness.notna().sum()),
                         pair_values=int(rr[rr.category.isin(prim)].fisher_r.notna().sum())))
    t = pd.DataFrame(rows); t.to_csv(out / 'inclusion_A1.csv', index=False)
    n = u[u.group == 'OTC'].drop_duplicates('subject_id').groupby('intact_hemi').size()
    print('A1 patients by intact hemisphere:', n.to_dict())
    print(t.to_string(index=False))


def _sig(df):
    """What the paper reads each row by: q for the per-ROI families, p for
    the mean-over-ROIs/pairs effects, omnibus p for omnibus rows."""
    s = np.where(df.comparison == 'omnibus', df.omnibus_p,
                 np.where(df.level.astype(str).str.startswith('MEAN_OVER'), df.p_perm,
                          np.where(df.q_fdr.notna(), df.q_fdr, df.p_perm)))
    return pd.Series(s, index=df.index)


def compare(files, out):
    t = {}
    for c, f in files.items():
        d = pd.read_csv(f)
        for k in KEYS:
            d[k] = d[k].fillna('').astype(str)
        d = d[d.measure != 'sum_selectivity']                    # retired measure
        d['crit'] = _sig(d); t[c] = d.set_index(KEYS)
    m = t['A1'][['crit', 'n_a', 'n_b', 'cohens_d']].join(
        t['S3'][['crit', 'n_a', 'n_b', 'cohens_d']], rsuffix='_S3').join(
        t['S5'][['crit', 'n_a', 'n_b', 'cohens_d']], rsuffix='_S5')
    blank = int(m.crit.isna().sum())                          # LMM omnibus rows when offline
    m = m[m.crit.notna() | m.crit_S3.notna()]
    for c in ['S3', 'S5']:
        m[f'flip_{c}'] = (m.crit < .05) != (m[f'crit_{c}'] < .05)
    m.reset_index().to_csv(out / 'compare_A1_S3_S5.csv', index=False)
    print(f'\nrows compared: {len(m)}  (rows with no value, e.g. LMM omnibus offline: {blank})')
    print('note: p moves by up to ~.01 from permutation draws alone (control rows, identical data)')
    for c in ['S3', 'S5']:
        f = m[m[f'flip_{c}']]
        print(f'\n{c}: {len(f)} rows cross .05')
        if len(f):
            print(f.reset_index()[KEYS + ['crit', f'crit_{c}']].to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/user_data/csimmon2/git_repos/sym_pt')
    a = ap.parse_args()
    base = Path(a.base); out = base / 'sandbox' / 'sens_AT'; out.mkdir(parents=True, exist_ok=True)
    offline = not _have_statsmodels()
    if offline:
        _stub_statsmodels(); print('OFFLINE: statsmodels missing; LMM omnibus cells will be blank')
    files = run_stats(base, out, offline)
    inclusion(base, out)
    compare(files, out)
