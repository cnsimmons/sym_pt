#!/usr/bin/env python
"""
Interaction evidence ratios for the two patient-vs-control cells that lack them:

  1. preferred-category similarity, LH-intact patients vs left-hemisphere controls
  2. peak selectivity,              RH-intact patients vs right-hemisphere controls

Same route as the patient-vs-patient factors already in the manuscript: fit two
Bayesian mixed models differing only in the category x group interaction, compare
by leave-one-out expected log predictive density, exponentiate the difference.

Run from the repo root:
    python bf_patient_vs_control.py
"""
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az

# ---------------------------------------------------------------- config
RSA      = 'D_liu/rsa_v1_harmonized.csv'
UNIV     = 'D_liu/univariate_v1_harmonized_sqrt.csv'
SUB_INFO = 'sub_info.csv'

EXCLUDE = ['sub-017', 'sub-091', 'sub-095', 'sub-096', 'sub-027', 'sub-084']
EXCLUDE_SES = [('sub-108', 2)]

PPA_PREF = 'house_PPA_strict'      # falls back to house_PPA if absent
DRAWS, TUNE, CHAINS, SEED = 2000, 2000, 4, 11


# ---------------------------------------------------------------- helpers
def ages():
    """age per subject-session, from the cohort table"""
    s = pd.read_csv(SUB_INFO)
    s['ses_num'] = s.ses.str.extract(r'(\d+)').astype(int)
    return s[['sub', 'ses_num', 'age']].rename(columns={'sub': 'subject_id'})


def load(path, value_col):
    d = pd.read_csv(path)
    d = d[~d.subject_id.isin(EXCLUDE)]
    for sub, ses in EXCLUDE_SES:
        d = d[~((d.subject_id == sub) & (d.ses_num == ses))]

    ppa = PPA_PREF if PPA_PREF in set(d.category) else 'house_PPA'
    rois = ['word_VWFA', 'face_FFA', ppa, 'object_LOC']
    print(f'  {path}: using {ppa} as the PPA parcel')
    d = d[d.category.isin(rois)].copy()

    # session selection: patients last session, controls first
    is_ctrl = d.groupby('subject_id').group.first().eq('control')
    mn = d.groupby('subject_id').ses_num.min()
    mx = d.groupby('subject_id').ses_num.max()
    keep = mn.where(is_ctrl, mx)
    d = d[d.ses_num.values == keep.reindex(d.subject_id).values]

    d = d.merge(ages(), on=['subject_id', 'ses_num'], how='left')
    missing = d.loc[d.age.isna(), 'subject_id'].unique()
    if len(missing):
        print(f'  WARNING no age for: {list(missing)}')

    d = d.dropna(subset=[value_col, 'age'])
    keepcols = ['subject_id', 'group', 'intact_hemi', 'hemi', 'category', 'age', value_col]
    return d[keepcols].rename(columns={value_col: 'y'})


def cell(d, hemi, intact):
    """patients with the given intact hemisphere, plus controls' matching hemisphere"""
    pt = d[(d.group != 'control') & (d.intact_hemi == intact) & (d.hemi == hemi)].copy()
    ct = d[(d.group == 'control') & (d.hemi == hemi)].copy()
    pt['grp'] = 'patient'
    ct['grp'] = 'control'
    out = pd.concat([pt, ct], ignore_index=True).drop_duplicates(
        subset=['subject_id', 'category'])
    out['age_c'] = out.age - out.age.mean()
    print(f'  n patients = {pt.subject_id.nunique()}, n controls = {ct.subject_id.nunique()}, '
          f'rows = {len(out)}')
    return out


def evidence_ratio(df, label):
    full = bmb.Model('y ~ category * grp + age_c + (1|subject_id)', df)
    null = bmb.Model('y ~ category + grp + age_c + (1|subject_id)', df)

    kw = dict(draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
              idata_kwargs={'log_likelihood': True}, progressbar=False)
    f = full.fit(**kw)
    n = null.fit(**kw)

    for name, idata in (('full', f), ('null', n)):
        rh = az.rhat(idata).max().to_array().max().item()
        div = int(idata.sample_stats.diverging.sum())
        print(f'    {name}: max r-hat {rh:.3f}, divergences {div}')

    lf, ln = az.loo(f, pointwise=True), az.loo(n, pointwise=True)
    diff = lf.elpd_loo - ln.elpd_loo
    se = np.sqrt(lf.se ** 2 + ln.se ** 2)
    er = float(np.exp(diff))
    verdict = ('favours a difference' if er > 3
               else 'favours absence' if er < 1 / 3 else 'inconclusive')
    print(f'  {label}')
    print(f'    elpd full {lf.elpd_loo:.2f}, null {ln.elpd_loo:.2f}, diff {diff:+.2f} (se {se:.2f})')
    print(f'    evidence ratio = {er:.4g}   ({verdict})')
    return er


# ---------------------------------------------------------------- run
if __name__ == '__main__':
    print('1. preferred-category similarity, LH-intact vs left controls')
    rsa = load(RSA, 'liu_distinctiveness')
    er1 = evidence_ratio(cell(rsa, hemi='l', intact='left'),
                         'ROI x group, preferred-category similarity, LH-intact vs L controls')

    print()
    print('2. peak selectivity, RH-intact vs right controls')
    uni = load(UNIV, 'peak_z')
    er2 = evidence_ratio(cell(uni, hemi='r', intact='right'),
                         'category x group, peak selectivity, RH-intact vs R controls')

    print()
    print(f'RESULT  preferred-category similarity, LH-intact : {er1:.4g}')
    print(f'RESULT  peak selectivity,              RH-intact : {er2:.4g}')