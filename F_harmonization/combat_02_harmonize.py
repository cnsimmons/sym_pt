#!/usr/bin/env python3
"""
combat_02_harmonize.py — Run ComBat on the feature matrix (step 3).

One harmonization per hemisphere, with the 4 categories stacked as features.
Covariates match the lab precedent (Robert et al., PNAS): batch = scanner,
preserve group + age + sex. Hemisphere is handled by the per-hemisphere split,
so it is NOT a covariate (avoids collinearity with group).

Age = linear by default. Set SMOOTH_AGE=True for a ComBat-GAM smooth age term.

Outputs to F_harmonization/combat_harmonized/:
  harmonized_{hemi}.npz   per-category harmonized [n_subj x n_vox] + subjects + mask
  model_{hemi}            saved neuroHarmonize model

Run on the cluster:  python combat_02_harmonize.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from neuroHarmonize import harmonizationLearn, saveHarmonizationModel
except ImportError:
    sys.exit("neuroHarmonize not found. Install with:  pip install neuroHarmonize neuroCombat")

GIT_DIR = Path('/user_data/csimmon2/git_repos/sym_pt')
IN_DIR  = GIT_DIR / 'F_harmonization' / 'combat_inputs'
OUT_DIR = GIT_DIR / 'F_harmonization' / 'combat_harmonized'

CATEGORIES = ['face', 'house', 'object', 'word']
HEMIS      = ['l', 'r']
SMOOTH_AGE = False     # True -> ComBat-GAM smooth age term


def build_design(cov):
    """neuroHarmonize covars: 'SITE' column + numeric covariates to preserve.
    Matches lab: group + age + sex (hemisphere handled by the per-hemi split)."""
    design = pd.DataFrame({'SITE': cov['scanner'].values,
                           'age':  cov['age'].values})
    design = pd.concat([
        design,
        pd.get_dummies(cov['sex'],   prefix='sex',   drop_first=True).astype(int),
        pd.get_dummies(cov['group'], prefix='group', drop_first=True).astype(int),
    ], axis=1)
    return design


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    covars = pd.read_csv(IN_DIR / 'covars.csv').set_index('subject_id')
    smooth = ['age'] if SMOOTH_AGE else []
    print(f"Covariates preserved: group, age, sex  |  age term: "
          f"{'SMOOTH (ComBat-GAM)' if SMOOTH_AGE else 'linear'}")

    for hemi in HEMIS:
        npz  = np.load(IN_DIR / f'features_{hemi}.npz', allow_pickle=True)
        subs = list(npz['subs_face'])
        for c in CATEGORIES:
            assert list(npz[f'subs_{c}']) == subs, f'subject mismatch in {c}'

        cov    = covars.loc[subs].reset_index()
        design = build_design(cov)
        nvox   = npz['X_face'].shape[1]
        data   = np.hstack([npz[f'X_{c}'] for c in CATEGORIES])   # [n_subj x 4*nvox]

        print(f"\n[{hemi.upper()}H] n={len(subs)}  features={data.shape[1]:,}  "
              f"site={cov['scanner'].value_counts().to_dict()}  "
              f"group={cov['group'].value_counts().to_dict()}")

        model, adj = harmonizationLearn(data, design, smooth_terms=smooth)

        out = {'subs': np.array(subs), 'mask': npz['mask'],
               'affine': npz['affine'], 'shape': npz['shape']}
        for i, c in enumerate(CATEGORIES):
            out[f'X_{c}'] = adj[:, i*nvox:(i+1)*nvox].astype(np.float32)
        np.savez_compressed(OUT_DIR / f'harmonized_{hemi}.npz', **out)
        saveHarmonizationModel(model, str(OUT_DIR / f'model_{hemi}'))

        # diagnostics: site gap should shrink; group gap should hold
        site = cov['scanner'].values
        ctrl = (cov['group'] == 'control').values
        def gap(M, a, b): return float(np.abs(M[a].mean(0) - M[b].mean(0)).mean())
        print(f"  site  gap (Verio vs Prisma): {gap(data, site=='Verio', site=='Prisma'):.4f}"
              f"  ->  {gap(adj, site=='Verio', site=='Prisma'):.4f}   (should shrink)")
        print(f"  group gap (ctrl vs patient): {gap(data, ctrl, ~ctrl):.4f}"
              f"  ->  {gap(adj, ctrl, ~ctrl):.4f}   (should hold)")
        print(f"  -> harmonized_{hemi}.npz, model_{hemi}")

    print(f"\nDone. Harmonized features in {OUT_DIR}")


if __name__ == '__main__':
    main()