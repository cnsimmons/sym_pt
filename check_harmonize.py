#!/usr/bin/env python3
"""
check_harmonize.py - compare pipeline/ harmonization outputs with the committed files.

Prints, for univariate, RSA and the TFCE maps (left, right), the largest absolute
difference between pipeline/outputs/... and the committed file. 0, or something
near 1e-12, means 02_harmonize.py reproduces the committed harmonization.
"""
from pathlib import Path

import numpy as np
import pandas as pd

G = Path('/user_data/csimmon2/git_repos/sym_pt')
O = G / 'pipeline' / 'outputs'

for name, a, b in [
        ('univariate', G / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv', O / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv'),
        ('RSA',        G / 'D_liu' / 'rsa_v1_harmonized.csv',             O / 'D_liu' / 'rsa_v1_harmonized.csv')]:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        print(f'{name}: different layout - committed {x.shape}, pipeline {y.shape}')
        continue
    key = [c for c in ['subject_id', 'session', 'hemi', 'category', 'pair'] if c in x.columns]
    x = x.sort_values(key).reset_index(drop=True)
    y = y.sort_values(key).reset_index(drop=True)
    num = x.select_dtypes('number').columns
    d = np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))
    same_nan = bool((x[num].isna().to_numpy() == y[num].isna().to_numpy()).all())
    text_diff = [c for c in x.columns if c not in num and not (x[c].astype(str) == y[c].astype(str)).all()]
    print(f'{name:10s} {x["subject_id"].nunique()} subjects | max abs diff {d:.2e} | '
          f'same missing cells: {same_nan} | text columns that differ: {text_diff or "none"}')

for h in 'lr':
    a = np.load(G / 'F_harmonization' / 'combat_harmonized' / f'harmonized_{h}.npz', allow_pickle=True)
    b = np.load(O / 'F_harmonization' / 'combat_harmonized' / f'harmonized_{h}.npz', allow_pickle=True)
    same_subs = list(a['subs']) == list(b['subs'])
    d = (max(float(np.abs(a[f'X_{c}'] - b[f'X_{c}']).max()) for c in ['face', 'house', 'object', 'word'])
         if same_subs else float('nan'))
    print(f'TFCE {h}     {len(b["subs"])} subjects | same subjects: {same_subs} | max abs diff {d:.2e}')
