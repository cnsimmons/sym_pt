#!/usr/bin/env python3
"""
combat_04_compare.py — with/without-harmonization comparison (step 5).

Reads FWE-corrected TFCE maps from the original (tfce_votc) and harmonized
(tfce_votc_harmonized) runs; tables per category x hemisphere x direction:
  min FWE p, # significant voxels (corrp>0.95), survives (FWE p<.05).

A cluster survives if max(1-p map) >= 0.95.  tstat1 = ctrl>pt, tstat2 = pt>ctrl.
Same naming/prefix in both runs ('{cat}_{hemi}_pt_vs_ctrl' / 'rand_tfce_corrp_tstat#').

Run:  python combat_04_compare.py
"""
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

GR   = Path(processed_dir) / 'group_results'
ORIG = GR / 'tfce_votc_fdr'             # <-- your ORIGINAL unharmonized TFCE.
HARM = GR / 'tfce_votc_harmonized'   #     If your published run is elsewhere
                                     #     (e.g. tfce_votc_fdr), change ORIG.

CATEGORIES = ['face', 'house', 'object', 'word']
HEMIS      = ['l', 'r']
DIRS       = {1: 'ctrl>pt', 2: 'pt>ctrl'}
THR        = 0.95                    # corrp >= .95  <=>  FWE p <= .05
KEY = {('object', 'l', 'ctrl>pt'), ('house', 'r', 'ctrl>pt'), ('word', 'r', 'pt>ctrl')}


def read_corrp(run_dir, cat, hemi, tstat):
    f = run_dir / f'{cat}_{hemi}_pt_vs_ctrl' / f'rand_tfce_corrp_tstat{tstat}.nii.gz'
    if not f.exists():
        return None
    mx = float(nib.load(str(f)).get_fdata().max())
    return {'fwe_p': round(1 - mx, 4), 'n_sig': int((nib.load(str(f)).get_fdata() > THR).sum()),
            'survives': mx >= THR}


rows = []
for cat in CATEGORIES:
    for hemi in HEMIS:
        for t, dirn in DIRS.items():
            o = read_corrp(ORIG, cat, hemi, t)
            h = read_corrp(HARM, cat, hemi, t)
            row = {'category': cat, 'hemi': hemi, 'direction': dirn}
            row.update({'orig_p':  o['fwe_p']    if o else None,
                        'orig_sig': o['survives'] if o else None,
                        'orig_nvox': o['n_sig']   if o else None,
                        'harm_p':  h['fwe_p']    if h else None,
                        'harm_sig': h['survives'] if h else None,
                        'harm_nvox': h['n_sig']   if h else None})
            rows.append(row)

df = pd.DataFrame(rows)

def verdict(r):
    if r.orig_sig is None or r.harm_sig is None: return 'missing'
    if r.orig_sig and r.harm_sig:               return 'held'
    if r.orig_sig and not r.harm_sig:           return 'LOST'
    if not r.orig_sig and r.harm_sig:           return 'gained'
    return 'n.s. both'
df['verdict'] = df.apply(verdict, axis=1)

df.to_csv(HARM / 'comparison_with_without.csv', index=False)
print(df.to_string(index=False))
print("\n=== KEY CLUSTERS ===")
for _, r in df.iterrows():
    if (r.category, r.hemi, r.direction) in KEY:
        print(f"  {r.category}_{r.hemi:1s} {r.direction:8s}: "
              f"orig p={r.orig_p} ({'sig' if r.orig_sig else 'n.s.'})  ->  "
              f"harm p={r.harm_p} ({'sig' if r.harm_sig else 'n.s.'})   [{r.verdict}]")
print(f"\nSaved: {HARM/'comparison_with_without.csv'}")