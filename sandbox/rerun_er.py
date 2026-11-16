#!/usr/bin/env python3
"""
rerun_er.py - the sampled interaction evidence ratios (V3 Table 1, "BF" cells)
for each processing version, as a SLURM array: one model comparison per job.

The model, priors, sampler settings and data loading are copied from
whole_otc_bayes.ipynb (cell 1 load_harmonized; cell 12 patients; cell 13
controls): two bambi models differing only in the interaction term, subject
random intercept, 4 chains x 4000 draws (2000 tune), seed 42, PSIS-LOO; the
evidence ratio is exp(elpd_full - elpd_null).

Runs: P0 record x A0 (check against V3), P0 x A1, P1 x A1, P2 x A1;
      14 tests each (7 patient, 7 control) = 56 jobs.

  python rerun_er.py plan            lists the jobs, checks inputs and cohort sizes
  python rerun_er.py submit          writes and submits the array (20 at a time)
  python rerun_er.py run INDEX       one job (what the array calls)
  python rerun_er.py collect         merges the results; compares P0 x A0 with V3
Writes only into sandbox/rerun_er/.
"""
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
SB = REPO / 'sandbox'
OUT = SB / 'rerun_er'
HZ = SB / 'rerun_harmonize'
VERSIONS = {
    'P0_record':       (REPO / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv', REPO / 'D_liu' / 'rsa_v1_harmonized.csv'),
    'P1_nosmooth':     (HZ / 'new13' / 'univariate_v1_harmonized_sqrt.csv', HZ / 'new13' / 'rsa_v1_harmonized.csv'),
    'P2_nosmooth_c20': (HZ / 'new20' / 'univariate_v1_harmonized_sqrt.csv', HZ / 'new20' / 'rsa_v1_harmonized.csv'),
}
EXCLUDE = ['sub-017', 'sub-091', 'sub-095', 'sub-096', 'sub-027', 'sub-084']
COHORTS = {'A0_record': EXCLUDE, 'A1_no010': EXCLUDE + ['sub-010']}
RUNS = [('P0_record', 'A0_record'), ('P0_record', 'A1_no010'),
        ('P1_nosmooth', 'A1_no010'), ('P2_nosmooth_c20', 'A1_no010')]
EXCLUDE_SES = [('sub-108', 2)]
AGE_CAP = 23
ROIS = ['word_VWFA', 'face_FFA', 'house_PPA_strict', 'object_LOC']
MEASURES = [('peak_z', 'peak selectivity'), ('volume', 'selective voxel count'),
            ('liu_distinctiveness', 'preferred-category similarity')]
DRAWS, TUNE, CHAINS = 4000, 2000, 4
V3 = {  # V3 Table 1 / bayes_factors_writeup.md, cohort of record
    ('patients', 'peak selectivity'): 0.173, ('patients', 'selective voxel count'): 0.063,
    ('patients', 'preferred-category similarity'): 0.050,
    ('patients', 'word_VWFA'): 0.130, ('patients', 'face_FFA'): 42.7,
    ('patients', 'house_PPA_strict'): 0.013, ('patients', 'object_LOC'): 0.022,
    ('controls', 'peak selectivity'): 16.2, ('controls', 'selective voxel count'): 39.7,
    ('controls', 'preferred-category similarity'): 218,
    ('controls', 'word_VWFA'): 63.5, ('controls', 'face_FFA'): 92.1,
    ('controls', 'house_PPA_strict'): 44.8, ('controls', 'object_LOC'): 0.591}

TESTS = ([('patients', m, lab) for m, lab in MEASURES] + [('patients', 'fisher_r', r) for r in ROIS]
         + [('controls', m, lab) for m, lab in MEASURES] + [('controls', 'fisher_r', r) for r in ROIS])
JOBS = [(v, c, t) for v, c in RUNS for t in TESTS]


def _ages():
    """As the notebook's _merge_age: no subject_id column in sub_info.csv means
    no age lookup, so the cap passes everyone (EXCLUDE already removes 091/095/096)."""
    si = pd.read_csv(REPO / 'sub_info.csv')
    col = next((c for c in si.columns if c.lower() in ('age', 'age_at_scan', 'scan_age')), None)
    if col is None or 'subject_id' not in si.columns:
        return None
    return si.groupby('subject_id')[col].min().rename('age')


def load_harmonized(path, exclude):
    """As whole_otc_bayes.ipynb cell 1 (age cap applied): controls -> first
    session, OTC -> last, patients' intact hemisphere only."""
    d = pd.read_csv(path)
    d = d[d['group'].isin(['OTC', 'control'])]
    d = d[~d['subject_id'].isin(exclude)].copy()
    d['ses_num'] = pd.to_numeric(d['session'].astype(str).str.extract(r'(\d+)', expand=False),
                                 errors='coerce').astype('Int64')
    for bs, bn in EXCLUDE_SES:
        d = d[~((d['subject_id'] == bs) & (d['ses_num'] == bn))]
    a = _ages()
    if a is None:
        d['age'] = np.nan
    else:
        d = d.merge(a, left_on='subject_id', right_index=True, how='left')
    d = d[(d['age'] <= AGE_CAP) | d['age'].isna()]
    status = d['status'] if 'status' in d.columns else d['group']
    ctrl = d[status == 'control'].copy()
    ctrl = ctrl[ctrl['ses_num'] == ctrl.groupby('subject_id')['ses_num'].transform('min')]
    pt = d[d['group'] == 'OTC'].copy()
    pt = pt[pt['ses_num'] == pt.groupby('subject_id')['ses_num'].transform('max')]
    pt = pt[pt['hemi'] == pt['intact_hemi'].map({'left': 'l', 'right': 'r'})]
    return ctrl, pt


def test_frame(version, cohort, test):
    who, col, lab = test
    uni, rsa = VERSIONS[version]
    excl = COHORTS[cohort]
    if col in ('peak_z', 'volume'):
        c, p = load_harmonized(uni, excl)
    else:
        c, p = load_harmonized(rsa, excl)
    if col == 'liu_distinctiveness':
        c = c.drop_duplicates(['subject_id', 'category', 'hemi'])
        p = p.drop(columns=['pair', 'fisher_r'], errors='ignore').drop_duplicates()
    if col == 'fisher_r':
        d = (p if who == 'patients' else c)
        d = d[d['category'] == lab]
        return d, 'fisher_r', 'pair', ('intact_hemi' if who == 'patients' else 'hemi')
    d = (p if who == 'patients' else c)
    d = d[d['category'].isin(ROIS)]
    return d, col, 'category', ('intact_hemi' if who == 'patients' else 'hemi')


def evidence_ratio(d, y, f1, f2, sid='subject_id'):
    if os.environ.get('ER_STUB'):                      # plumbing tests only
        return float(np.exp(np.random.default_rng(len(d)).normal()))
    import bambi as bmb
    import arviz as az
    d = d[[y, f1, f2, sid]].dropna().copy()
    for c in (f1, f2):
        d[c] = d[c].astype('category')
    fits = {}
    for name, formula in (('full', f'{y} ~ {f1} * {f2} + (1|{sid})'),
                          ('null', f'{y} ~ {f1} + {f2} + (1|{sid})')):
        fits[name] = bmb.Model(formula, d).fit(draws=DRAWS, tune=TUNE, chains=CHAINS,
                                               idata_kwargs={'log_likelihood': True},
                                               progressbar=False, random_seed=42)
    e = az.compare(fits, ic='loo')['elpd_loo']
    return float(np.exp(e['full'] - e['null']))


def plan():
    missing = [str(p) for v in VERSIONS.values() for p in v if not p.exists()]
    print(f'{len(JOBS)} jobs ({len(RUNS)} version x cohort runs x {len(TESTS)} tests)')
    print('inputs:', 'all present' if not missing else 'MISSING ' + ', '.join(missing))
    if missing:
        return False
    for v, c in RUNS:
        cu, pu = load_harmonized(VERSIONS[v][0], COHORTS[c])
        print(f'  {v:16s} {c:10s} controls {cu.subject_id.nunique()}, LH-intact '
              f"{pu[pu.intact_hemi == 'left'].subject_id.nunique()}, RH-intact "
              f"{pu[pu.intact_hemi == 'right'].subject_id.nunique()}")
    return True


def submit():
    if not plan():
        sys.exit('STOP: inputs missing')
    (OUT / 'results').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    sh = OUT / 'er_array.sh'
    sh.write_text(f"""#!/bin/bash
#SBATCH --job-name=er
#SBATCH --array=0-{len(JOBS) - 1}%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output={OUT}/logs/er_%a.out
#SBATCH --error={OUT}/logs/er_%a.err
source /home/csimmon2/anaconda3/etc/profile.d/conda.sh && conda activate fmri
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTENSOR_FLAGS="compiledir=/tmp/pytensor_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
python {Path(__file__).resolve()} run $SLURM_ARRAY_TASK_ID
rm -rf /tmp/pytensor_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
""")
    r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


def run(i):
    v, c, t = JOBS[i]
    d, y, f1, f2 = test_frame(v, c, t)
    er = evidence_ratio(d, y, f1, f2)
    who, col, lab = t
    out = OUT / 'results' / f'{i:03d}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'version': v, 'cohort': c, 'who': who, 'test': lab, 'n_rows': len(d), 'ER': er}]).to_csv(out, index=False)
    print(f'{v} {c} {who} {lab}: ER = {er:.4g}  -> {out}')


def collect():
    files = sorted((OUT / 'results').glob('*.csv'))
    print(f'{len(files)} of {len(JOBS)} results present')
    if not files:
        return
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    R.to_csv(OUT / 'evidence_ratios.csv', index=False)
    T = R.pivot_table(index=['who', 'test'], columns=['version', 'cohort'], values='ER', aggfunc='first')
    with pd.option_context('display.float_format', '{:.3g}'.format, 'display.width', 200):
        print(T.to_string())
    a0 = R[(R.version == 'P0_record') & (R.cohort == 'A0_record')]
    if len(a0):
        print('\nP0 x A0 against V3 (sampling varies slightly between runs):')
        for _, r in a0.iterrows():
            v3 = V3.get((r.who, r.test))
            print(f'  {r.who:8s} {r.test:30s} here {r.ER:9.3g}   V3 {v3:9.3g}   ratio {r.ER / v3:6.2f}')
    print(f'saved: {OUT / "evidence_ratios.csv"}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'plan'
    {'plan': plan, 'submit': submit, 'collect': collect}.get(cmd, lambda: run(int(sys.argv[2])))()
