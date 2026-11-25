#!/usr/bin/env python3
"""
harmonize_lo_inf.py - ComBat for the inferior-only LO, run exactly as the paper's version (new20).

Takes new20's two extraction files (rerun_extract/uni_new20, rsa_new20), replaces the values of every
object_LOC row with the matching object_LOC_inf row from ventral_otc/uni_lo, rsa_lo (same subject,
session, hemisphere, pair; the label stays object_LOC, so the record ComBat scripts see the same 12
ROIs), then calls sandbox/rerun_harmonize.py's own run() as a new version 'lo_inf': same two record
ComBat scripts, same fit set, same scanner batch, covariates, winsorizing and sqrt, same rule of never
overwriting.

Writes only:
  sandbox/ventral_otc/harmonize_lo/univariate_v1.csv, rsa_v1.csv   swapped inputs
  sandbox/rerun_harmonize/lo_inf/                                  harmonized outputs, logs, checks.txt
  /user_data/csimmon2/lo_inf_harmonized_2026-11-18.tar.gz          the lo_inf folder, for scp
Stops before writing anything if the extraction's old-LO rows are not identical to new20's.

Usage (fmri env, compute node):
  python /user_data/csimmon2/git_repos/sym_pt/sandbox/ventral_otc/harmonize_lo_inf.py
"""
import importlib.util
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
SB = REPO / 'sandbox'
SWAP_DIR = SB / 'ventral_otc' / 'harmonize_lo'
LO = {'uni': SB / 'ventral_otc' / 'uni_lo' / 'uni_lo.csv', 'rsa': SB / 'ventral_otc' / 'rsa_lo' / 'rsa_lo.csv'}
SWAP_NAME = {'uni': 'univariate_v1.csv', 'rsa': 'rsa_v1.csv'}
KEYS = {'uni': ['subject_id', 'session', 'hemi'], 'rsa': ['subject_id', 'session', 'hemi', 'pair']}
ID_COLS = ['subject_id', 'code', 'session', 'group', 'status', 'surgery_side', 'intact_hemi', 'hemi',
           'hemi_label', 'category', 'pair']
TAR = Path('/user_data/csimmon2/lo_inf_harmonized_2026-11-18.tar.gz')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    """Element-wise equality with NaN == NaN, any dtype."""
    return (a == b) | (a.isna() & b.isna())


def build(kind, base_path):
    """Swapped file as text: every non-LO line is new20's own line, byte for byte; every object_LOC line
    is the matching object_LOC_inf line from the extraction with only its category field relabeled.
    (Rewriting through pandas could move floats by one unit in the last place.)"""
    base, lo, k = pd.read_csv(base_path), pd.read_csv(LO[kind]), KEYS[kind]
    vals = [c for c in base.columns if c not in ID_COLS]
    tgt = (base['category'] == 'object_LOC').to_numpy()
    old = lo[lo['category'] == 'object_LOC']
    inf = lo[lo['category'] == 'object_LOC_inf']
    # check 1: the extraction's old-LO rows are exactly new20's object_LOC rows
    a = base.loc[tgt, k + vals].merge(old[k + vals], on=k, how='outer', suffixes=('', '_x'), indicator=True)
    if (a['_merge'] != 'both').any() or len(a) != tgt.sum():
        sys.exit(f'STOP ({kind}): old-LO rows do not pair one-to-one with new20 object_LOC rows')
    bad = [c for c in vals if not same(a[c], a[c + '_x']).all()]
    if bad:
        sys.exit(f'STOP ({kind}): old-LO rows differ from new20 in {bad}')
    # row of the extraction file that replaces each object_LOC row of new20
    pos = base.loc[tgt, k].reset_index().merge(inf[k].reset_index(), on=k, how='left', validate='one_to_one',
                                                suffixes=('_b', '_lo'))
    if pos['index_lo'].isna().any():
        sys.exit(f'STOP ({kind}): {int(pos.index_lo.isna().sum())} object_LOC rows have no object_LOC_inf row')
    bl, ll = Path(base_path).read_text().splitlines(), Path(LO[kind]).read_text().splitlines()
    if len(bl) != len(base) + 1 or len(ll) != len(lo) + 1 or bl[0] != ll[0] or '"' in bl[0] + ll[0]:
        sys.exit(f'STOP ({kind}): unexpected CSV layout (line counts or headers differ)')
    ci, nf = bl[0].split(',').index('category'), len(bl[0].split(','))
    for b, j in zip(pos['index_b'], pos['index_lo'].astype(int)):
        f = ll[j + 1].split(',')
        if len(f) != nf or f[ci] != 'object_LOC_inf' or '"' in ll[j + 1]:
            sys.exit(f'STOP ({kind}): cannot relabel extraction line {j + 2}')
        f[ci] = 'object_LOC'
        bl[b + 1] = ','.join(f)
    text = '\n'.join(bl) + '\n'
    # verify what the ComBat script will read: non-LO rows identical to new20, LO rows identical to LO_inf
    from io import StringIO
    new = pd.read_csv(StringIO(text))
    if not all(same(new.loc[~tgt, c], base.loc[~tgt, c]).all() for c in base.columns):
        sys.exit(f'STOP ({kind}): a non-LO row changed during the swap')
    got = new.loc[tgt, k + vals].reset_index(drop=True)
    want = inf.loc[pos['index_lo'].astype(int), k + vals].reset_index(drop=True)
    if not all(same(got[c], want[c]).all() for c in k + vals):
        sys.exit(f'STOP ({kind}): swapped LO rows do not equal the object_LOC_inf rows')
    n_changed = int(sum((~same(new.loc[tgt, c], base.loc[tgt, c])).sum() for c in vals))
    return text, f'{kind}: {int(tgt.sum())} object_LOC rows swapped ({n_changed} values changed); ' \
                 f'{int((~tgt).sum())} other rows identical to new20'


def moved(kind, rh):
    """After ComBat: how far every non-LO value moved from new20's harmonized file."""
    a = pd.read_csv(rh.H / 'lo_inf' / rh.NAMES[kind])
    b = pd.read_csv(rh.H / 'new20' / rh.NAMES[kind])
    k = [c for c in rh.KEYS if c in a.columns and c in b.columns]
    m = a.merge(b, on=k, how='outer', suffixes=('', '_n20'), indicator=True)
    lines = [f'{kind}: {int((m._merge == "both").sum())} rows matched, {int((m._merge != "both").sum())} unmatched']
    m = m[(m._merge == 'both') & (m['category'] != 'object_LOC')]
    for c in [c for c in b.columns if c not in k and c in a.columns and pd.api.types.is_numeric_dtype(b[c])]:
        d = (pd.to_numeric(m[c], errors='coerce') - pd.to_numeric(m[c + '_n20'], errors='coerce')).abs()
        if d.max() > 0:
            lines.append(f'   non-LO {c:22s} max {d.max():.4g}  median {d.median():.3g}')
    return lines


if __name__ == '__main__':
    rh = load(SB / 'rerun_harmonize.py', 'rerun_harmonize')
    for p in [*LO.values(), *rh.INPUTS['new20'].values(), *rh.RECORD.values()]:
        if not p.exists():
            sys.exit(f'STOP: {p} not found')
    if any((SWAP_DIR / n).exists() for n in SWAP_NAME.values()):
        sys.exit(f'STOP: swapped inputs already exist in {SWAP_DIR}; nothing overwritten. To redo: rm -r {SWAP_DIR}')
    built = {kind: build(kind, rh.INPUTS['new20'][kind]) for kind in ('uni', 'rsa')}   # all checks before any write
    SWAP_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for kind, (text, msg) in built.items():
        (SWAP_DIR / SWAP_NAME[kind]).write_text(text)
        report.append(msg)
    rh.INPUTS['lo_inf'] = {kind: SWAP_DIR / SWAP_NAME[kind] for kind in ('uni', 'rsa')}
    for kind in ('uni', 'rsa'):
        report.append(f'lo_inf {kind}: {rh.run("lo_inf", kind)}')
    for kind in ('uni', 'rsa'):
        if (rh.H / 'lo_inf' / rh.NAMES[kind]).exists():
            report += moved(kind, rh)
    out = rh.H / 'lo_inf'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checks.txt').write_text('\n'.join(report) + '\n')
    with tarfile.open(TAR, 'w:gz') as t:
        t.add(out, arcname='lo_inf')
    print('\n'.join(report))
    print(f'\nwrote {out} and {TAR}')
