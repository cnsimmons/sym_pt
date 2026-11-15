#!/usr/bin/env python3
"""
rerun_harmonize.py - ComBat refit for the rerun, using the record's two ComBat
scripts (combat_05b univariate, combat_06 RSA) unchanged.

For each version it loads both scripts fresh, points their input and output at
that version's files, and sets the fit set to exactly the subjects in the
record's harmonized files (so the refit uses the same set as the record). The
scanner batch, covariates, winsorizing and sqrt are the scripts' own.

Versions, outputs in sandbox/rerun_harmonize/<version>/:
  record_check  the record's own extraction (D_liu/univariate_v1.csv, rsa_v1.csv).
                Must reproduce the record's harmonized files exactly: checks the refit.
  rec13         the record tree re-extracted (rerun_extract uni_rec13 / rsa_rec13).
                Must also reproduce them: checks the re-extraction.
  new13         new tree, word ROIs on -cope13
  new20         new tree, word ROIs on cope 20 (Words - Face)
Writes only under sandbox/rerun_harmonize/; stops before overwriting anything.

Usage (fmri env, on a compute node), after the rx_ extraction jobs have finished:
    python rerun_harmonize.py
"""
import contextlib
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
D_LIU = REPO / 'D_liu'
X = REPO / 'sandbox' / 'rerun_extract'
H = REPO / 'sandbox' / 'rerun_harmonize'
SCRIPTS = {'uni': REPO / 'F_harmonization' / 'combat_05b_harmonize_univariate_sqrt.py',
           'rsa': REPO / 'F_harmonization' / 'combat_06_harmonize_rsa.py'}
NAMES = {'uni': 'univariate_v1_harmonized_sqrt.csv', 'rsa': 'rsa_v1_harmonized.csv'}
RECORD = {k: D_LIU / n for k, n in NAMES.items()}
INPUTS = {'record_check': {'uni': D_LIU / 'univariate_v1.csv', 'rsa': D_LIU / 'rsa_v1.csv'},
          'rec13': {'uni': X / 'uni_rec13' / 'univariate_v1.csv', 'rsa': X / 'rsa_rec13' / 'rsa_v1.csv'},
          'new13': {'uni': X / 'uni_new13' / 'univariate_v1.csv', 'rsa': X / 'rsa_new13' / 'rsa_v1.csv'},
          'new20': {'uni': X / 'uni_new20' / 'univariate_v1.csv', 'rsa': X / 'rsa_new20' / 'rsa_v1.csv'}}
KEYS = ['subject_id', 'session', 'hemi', 'category', 'pair']


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(new, old):
    a, b = pd.read_csv(new), pd.read_csv(old)
    keys = [k for k in KEYS if k in a.columns and k in b.columns]
    m = a.merge(b, on=keys, how='outer', suffixes=('', '__rec'), indicator=True)
    only_new, only_rec = int((m._merge == 'left_only').sum()), int((m._merge == 'right_only').sum())
    m = m[m._merge == 'both']
    num = [c for c in b.columns if c not in keys and c in a.columns and pd.api.types.is_numeric_dtype(b[c])]
    worst = {}
    for c in num:
        u, v = pd.to_numeric(m[c], errors='coerce'), pd.to_numeric(m[c + '__rec'], errors='coerce')
        d = (u - v).abs()
        d[u.isna() & v.isna()] = 0
        d[u.isna() ^ v.isna()] = np.inf
        if d.max() > 1e-9:
            worst[c] = float(d.max())
    ok = not worst and only_new == 0 and only_rec == 0
    return ok, f'{len(m)} rows matched, {only_new} only here, {only_rec} only in the record; ' + \
        ('every value identical' if not worst else 'max differences ' +
         ', '.join(f'{c} {v:.3g}' for c, v in worst.items()))


def run(version, kind):
    inp, out = INPUTS[version][kind], H / version / NAMES[kind]
    if not inp.exists():
        return f'skipped: {inp} not found (extraction not finished?)'
    if out.exists():
        return f'skipped: {out} exists (nothing overwritten)'
    out.parent.mkdir(parents=True, exist_ok=True)
    m = load(SCRIPTS[kind], f'combat_{version}_{kind}')
    fit = set(pd.read_csv(RECORD[kind], usecols=['subject_id'])['subject_id'])
    have = set(pd.read_csv(inp, usecols=['subject_id'])['subject_id'])
    m.s.EXCLUDE = sorted(have - fit)                  # fit set = the record's harmonized subjects
    setattr(m, 'UNIVAR' if kind == 'uni' else 'RSA', inp)
    m.OUT = out
    with open(out.with_suffix('.log'), 'w') as log, contextlib.redirect_stdout(log), \
            contextlib.redirect_stderr(log):
        m.main()
    got = set(pd.read_csv(out, usecols=['subject_id'])['subject_id'])
    line = f'{len(got)} subjects (record fit set {len(fit)}; missing {sorted(fit - got) or "none"})'
    if version in ('record_check', 'rec13'):
        ok, msg = compare(out, RECORD[kind])
        line += f'\n      vs record: {"PASS" if ok else "DIFFERS"}: {msg}'
    return line


if __name__ == '__main__':
    for version in INPUTS:
        for kind in ('uni', 'rsa'):
            print(f'{version:12s} {kind}: {run(version, kind)}')
    print(f'\noutputs and logs in {H}')
