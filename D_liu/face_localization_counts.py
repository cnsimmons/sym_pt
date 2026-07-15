#!/usr/bin/env python3
"""Recompute per-patient face-localization voxel counts for the corrected
cohort, reusing 05_stats_harmony's exclusions + session rule so numbers match
the reported WTA analysis. Face-winning voxel count = voxel_count where
region=otc, denominator=selective, category=face. Read-only; writes nothing."""
import sys, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

spec = importlib.util.spec_from_file_location(
    "sh", "/user_data/csimmon2/git_repos/sym_pt/D_liu/verified/05_stats_harmony.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)

wta = sh.apply_exclusions(pd.read_csv(sh.WTA_CSV))
wta = wta[(wta['region'] == 'otc') & (wta['denominator'] == 'selective')
          & (wta['category'] == 'face')].copy()
wta = sh.select_sessions(wta, pt_rule='last')   # same rule as reported WTA

def counts(status, hemi, hl=None):
    d = wta[(wta['status'] == status) & (wta['hemi'] == hemi)]
    if hl is not None:
        d = d[d['hemi_label'] == hl]
    return d['voxel_count'].astype(float).values

# Patients: intact hemisphere only
pt_lh = counts('patient', 'l', 'intact')   # LH-intact patients
pt_rh = counts('patient', 'r', 'intact')   # RH-intact patients
pt_all = np.concatenate([pt_lh, pt_rh])

# Controls: matched hemisphere
ct_lh = counts('control', 'l')
ct_rh = counts('control', 'r')

print("=== Per-patient face-winning voxel counts (intact-hemi OTC) ===\n")
print(f"All patients (n={len(pt_all)}): "
      f"range {pt_all.min():.0f}-{pt_all.max():.0f}, median {np.median(pt_all):.1f}")
print()
print(f"LH-intact patients (n={len(pt_lh)}): mean {pt_lh.mean():.0f}, "
      f"median {np.median(pt_lh):.0f}, range {pt_lh.min():.0f}-{pt_lh.max():.0f}")
print(f"  vs controls LH (n={len(ct_lh)}): mean {ct_lh.mean():.0f}, median {np.median(ct_lh):.0f}")
print()
print(f"RH-intact patients (n={len(pt_rh)}): mean {pt_rh.mean():.0f}, "
      f"median {np.median(pt_rh):.0f}, range {pt_rh.min():.0f}-{pt_rh.max():.0f}")
print(f"  vs controls RH (n={len(ct_rh)}): mean {ct_rh.mean():.0f}, median {np.median(ct_rh):.0f}")
print()
print("Per-patient LH-intact counts (to check the 'two large' claim):")
print("  " + ", ".join(f"{v:.0f}" for v in sorted(pt_lh, reverse=True)))