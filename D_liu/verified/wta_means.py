#!/usr/bin/env python3
"""Print per-category WTA group means matching 05_stats_harmony.py exactly.
Reuses its apply_exclusions + select_sessions(pt_rule='last') so the means
are consistent with the reported chi2/Delta. Read-only; writes nothing."""
import sys
from pathlib import Path
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt/D_liu/verified')
import importlib.util

# import the stats module to reuse its exact filtering functions
spec = importlib.util.spec_from_file_location(
    "sh", "/user_data/csimmon2/git_repos/sym_pt/D_liu/verified/05_stats_harmony.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)

import pandas as pd

wta = sh.apply_exclusions(pd.read_csv(sh.WTA_CSV))
wta = wta[(wta['region'] == 'otc') & (wta['denominator'] == 'selective')].copy()
wta = sh.select_sessions(wta, pt_rule='last')   # identical to line 351

CATS = ['face', 'house', 'object', 'word']

def means(sub, factor_col):
    levels = sorted(sub[factor_col].unique())
    out = {}
    for lev in levels:
        out[lev] = {c: sub[(sub[factor_col] == lev) & (sub['category'] == c)]['wta_pct'].mean()
                    for c in CATS}
        out[lev]['n'] = sub[sub[factor_col] == lev]['subject_id'].nunique()
    return levels, out

# M1: LH ctrl vs pt
m1 = wta[wta['hemi'] == 'l'].copy()
m1['grp'] = (m1['group'] == 'OTC').map({True: 'pt', False: 'ctrl'})
# M2: RH ctrl vs pt
m2 = wta[wta['hemi'] == 'r'].copy()
m2['grp'] = (m2['group'] == 'OTC').map({True: 'pt', False: 'ctrl'})
# M3: LH pt vs RH pt
m3 = wta[wta['group'] == 'OTC'].copy()
m3['intact'] = (m3['hemi'] == 'l').map({True: 'LH', False: 'RH'})

for name, sub, fac in [('M1 (LH ctrl vs pt)', m1, 'grp'),
                       ('M2 (RH ctrl vs pt)', m2, 'grp'),
                       ('M3 (LH-intact vs RH-intact pt)', m3, 'intact')]:
    levels, out = means(sub, fac)
    print(f"\n=== {name} ===")
    hdr = "cat      " + "".join(f"{lev} (n={out[lev]['n']})".ljust(16) for lev in levels)
    print(hdr)
    for c in CATS:
        print(f"{c:8s} " + "".join(f"{out[lev][c]:6.2f}%".ljust(16) for lev in levels))