#!/usr/bin/env python3
"""
check_outputs.py - compare every pipeline/ output with its committed counterpart.

Run after any stage. Outputs the pipeline has not produced yet show as "not yet".
For tables, rows are matched on their identifying columns, so extra or missing rows
are reported separately from value differences ("max diff" = largest absolute
difference over all matched numbers; 0 means identical).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description='Compare pipeline outputs with the committed files.')
ap.add_argument('--ignore', nargs='*', default=[], metavar='SUB:SES',
                help='sessions expected to differ (new or reprocessed), e.g. sub-108:2 sub-092:2; '
                     'their rows are left out of both files before comparing')
IGNORE = {(a.split(':')[0], int(a.split(':')[1])) for a in ap.parse_args().ignore}

G   = Path('/user_data/csimmon2/git_repos/sym_pt')
P   = G / 'pipeline'
O   = P / 'outputs'
GR  = Path('/user_data/csimmon2/sym_pt/group_results')            # committed group-level outputs
GRP = Path('/user_data/csimmon2/sym_pt/group_results_pipeline')   # pipeline group-level outputs

TABLES = [  # stage, label, committed, pipeline
    ('extraction', 'univariate_v1',          G / 'D_liu/univariate_v1.csv',                    O / 'D_liu/univariate_v1.csv'),
    ('extraction', 'rsa_v1',                 G / 'D_liu/rsa_v1.csv',                           O / 'D_liu/rsa_v1.csv'),
    ('extraction', 'liu_exact_replication',  G / 'liu_exact_replication_v2.csv',               P / 'liu_exact_replication_v2.csv'),
    ('extraction', 'peak_coords_mni',        GR / 'peak_coords/peak_coords_mni.csv',           GRP / 'peak_coords/peak_coords_mni.csv'),
    ('extraction', 'selective_voxel_counts', GR / 'roi_betas/selective_voxel_counts.csv',      GRP / 'roi_betas/selective_voxel_counts.csv'),
    ('extraction', 'otc_rsm',                G / 'H_RosenkeRSM/otc_rsm.csv',                   O / 'H_RosenkeRSM/otc_rsm.csv'),
    ('extraction', 'otc_rsm_persubject',     G / 'H_RosenkeRSM/otc_rsm_persubject.csv',        O / 'H_RosenkeRSM/otc_rsm_persubject.csv'),
    ('harmonization', 'univariate_harmonized', G / 'D_liu/univariate_v1_harmonized_sqrt.csv',  O / 'D_liu/univariate_v1_harmonized_sqrt.csv'),
    ('harmonization', 'rsa_harmonized',      G / 'D_liu/rsa_v1_harmonized.csv',                O / 'D_liu/rsa_v1_harmonized.csv'),
    ('analyses', 'wta_percentages',          GR / 'wta_percentages.csv',                       GRP / 'wta_percentages.csv'),
    ('analyses', 'stats vs main file',       G / 'D_liu/stats_results_harmonized_corrected.csv',       O / 'D_liu/stats_results_harmonized_corrected.csv'),
    ('analyses', 'stats vs _ptpt4',          G / 'D_liu/stats_results_harmonized_corrected_ptpt4.csv', O / 'D_liu/stats_results_harmonized_corrected.csv'),
    ('analyses', 'tfce_clusters',            G / 'D_liu/tfce_clusters.csv',                    O / 'D_liu/tfce_clusters.csv'),
    ('analyses', 'tfce_clusters vs _harmonized_corrected', G / 'D_liu/tfce_clusters_harmonized_corrected.csv', O / 'D_liu/tfce_clusters.csv'),
    ('analyses', 'grid (Tables 2, 4)',       G / 'grid.csv',                                   O / 'grid.csv'),
    ('analyses', 'lmm',                      G / 'lmm.csv',                                    O / 'lmm.csv'),
]
ARRAYS = [('harmonization', f'TFCE maps {h}', G / f'F_harmonization/combat_harmonized/harmonized_{h}.npz',
           O / f'F_harmonization/combat_harmonized/harmonized_{h}.npz') for h in 'lr']
IMAGE_DIRS = [('analyses', 'TFCE results', GR / 'tfce_votc_harmonized', GRP / 'tfce_votc_harmonized')]

ID_NUMERIC = {'session', 'ses_num', 'n_ses', 'comparison'}            # numeric columns that identify a row
NOT_KEYS   = {'note', 'notes', 'description', 'path'}  # free text, never used to match rows


def _max(a):
    a = np.abs(np.asarray(a, dtype=float))
    a = a[~np.isnan(a)]
    return float(a.max()) if a.size else 0.0


def _drop_ignored(df):
    ses = pd.to_numeric(df['session'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    keep = [not (pd.notna(n) and (sid, int(n)) in IGNORE) for sid, n in zip(df['subject_id'], ses)]
    return df[keep]


def compare_table(a, b):
    x, y = pd.read_csv(a, low_memory=False), pd.read_csv(b, low_memory=False)
    note = ''
    if IGNORE and all({'subject_id', 'session'} <= set(d.columns) for d in (x, y)):
        n0 = len(x) + len(y)
        x, y = _drop_ignored(x), _drop_ignored(y)
        if n0 - len(x) - len(y):
            note += f' | {n0 - len(x) - len(y)} rows of ignored sessions left out'
    if set(x.columns) != set(y.columns):
        only_c, only_p = sorted(set(x) - set(y)), sorted(set(y) - set(x))
        note = f' | columns only committed {only_c}, only pipeline {only_p} (compared on shared columns)'
        shared = [c for c in x.columns if c in set(y.columns)]
        x, y = x[shared], y[shared]
    y = y[x.columns]
    keys = [c for c in x.columns
            if (not pd.api.types.is_numeric_dtype(x[c]) or c in ID_NUMERIC) and c not in NOT_KEYS]
    num = [c for c in x.columns if c not in keys and c not in NOT_KEYS
           and pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
    # identifying columns whose labels differ between the two files (e.g. renamed models)
    relabeled = [k for k in keys if set(x[k].astype(str)) != set(y[k].astype(str))]
    if relabeled:
        ex = relabeled[0]
        a_only = sorted(set(x[ex].astype(str)) - set(y[ex].astype(str)))[:1]
        b_only = sorted(set(y[ex].astype(str)) - set(x[ex].astype(str)))[:1]
        note += f' | labels differ in {relabeled}, e.g. {a_only} vs {b_only}'
        rest = [k for k in keys if k not in relabeled]
        if rest and not x.duplicated(rest).any() and not y.duplicated(rest).any():
            keys = rest                          # match on the columns that still agree
    if keys and not x.duplicated(keys).any() and not y.duplicated(keys).any():
        xm, ym = x.copy(), y.copy()
        for k in keys:
            xm[k], ym[k] = xm[k].astype(str), ym[k].astype(str)
        drop = [c for c in x.columns if c not in keys and c not in num]   # unmatched text columns
        m = xm.drop(columns=drop).merge(ym.drop(columns=drop), on=keys, how='outer',
                                        suffixes=('_c', '_p'), indicator=True)
        both = m[m['_merge'] == 'both']
        d = max([_max(both[f'{c}_c'].astype(float) - both[f'{c}_p'].astype(float)) for c in num] + [0.0])
        same_nan = all((both[f'{c}_c'].isna() == both[f'{c}_p'].isna()).all() for c in num)
        msg = f'{len(both)} rows matched | max diff {d:.2e} | same missing cells: {same_nan}'
        for side, tag in (('left_only', 'only committed'), ('right_only', 'only pipeline')):
            extra = m[m['_merge'] == side]
            if len(extra):
                who = sorted(extra['subject_id'].unique())[:6] if 'subject_id' in extra else ''
                msg += f' | {len(extra)} rows {tag} {who}'
        return msg + note
    if len(x) != len(y):
        return f'row counts differ: committed {len(x)}, pipeline {len(y)}' + note
    cols = list(x.columns)
    xs = x.sort_values(cols).reset_index(drop=True)
    ys = y.sort_values(cols).reset_index(drop=True)
    d = _max(xs[num].to_numpy(float) - ys[num].to_numpy(float)) if num else 0.0
    return f'{len(x)} rows, compared in sorted order | max diff {d:.2e}' + note


def compare_arrays(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    if list(x['subs']) != list(y['subs']):
        return f'subjects differ: committed {len(x["subs"])}, pipeline {len(y["subs"])}'
    d = max(_max(x[f'X_{c}'] - y[f'X_{c}']) for c in ['face', 'house', 'object', 'word'])
    return f'{len(y["subs"])} subjects | max diff {d:.2e}'


def compare_images(a, b):
    import nibabel as nib
    files = sorted(p for p in a.rglob('*.nii.gz'))
    missing, d = [], 0.0
    for f in files:
        g = b / f.relative_to(a)
        if not g.exists():
            missing.append(str(f.relative_to(a)))
            continue
        d = max(d, _max(nib.load(str(f)).get_fdata() - nib.load(str(g)).get_fdata()))
    msg = f'{len(files) - len(missing)} images matched | max diff {d:.2e}'
    if missing:
        msg += f' | {len(missing)} not produced, e.g. {missing[0]}'
    return msg


def run(kind, fn, a, b):
    if not a.exists():
        return 'no committed file to compare against'
    if not b.exists():
        return 'not yet'
    try:
        return fn(a, b)
    except Exception as e:                      # report, keep checking the rest
        return f'could not compare: {e}'


if IGNORE:
    print('ignoring sessions:', ', '.join(f'{a} ses-{b:02d}' for a, b in sorted(IGNORE)))
print(f'{"stage":14s} {"output":40s} result')
for stage, label, a, b in TABLES:
    print(f'{stage:14s} {label:40s} {run("table", compare_table, a, b)}')
for stage, label, a, b in ARRAYS:
    print(f'{stage:14s} {label:40s} {run("arrays", compare_arrays, a, b)}')
for stage, label, a, b in IMAGE_DIRS:
    print(f'{stage:14s} {label:40s} {run("images", compare_images, a, b)}')
