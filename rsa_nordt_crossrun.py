#!/usr/bin/env python3
"""
rsa_nordt_crossrun.py — Nordt et al. (2023, Nat Commun 14:8010) distinctiveness.

WHAT IS DIFFERENT FROM 04_multivariate_analyses.py
--------------------------------------------------
1. CROSS-RUN correlations. Patterns for run A are correlated against patterns
   for run B, so the noise in the two vectors is independent. The existing
   pipeline correlates patterns estimated from the same combined runs, so noise
   is shared and every correlation is inflated.

2. A WITHIN-CATEGORY TERM.
       Nordt:   distinctiveness = within-category  -  between-category
       current: distinctiveness = between-category only
   The within term is corr(category i in run A, category i in run B) — a
   per-subject reliability estimate. Subtracting it cancels subject-level noise.
   Without it, a noisy subject looks artificially MORE distinct and a subject
   with globally inflated correlations looks LESS distinct.

3. Raw Pearson r, not Fisher-z. Nordt's distinctiveness ranges -2 to +2.
   HIGHER = MORE distinct (opposite sign convention to liu_distinctiveness).
   A Fisher-z version is also written for comparability.

4. MVPs z-scored across voxels before correlating (their Methods). Note this is
   a no-op for Pearson r, which already centres and scales each vector; it is
   included for fidelity, not because it changes the number.

WHAT IS THE SAME
----------------
Peak-finding, sphere construction, ROI definitions, session handling — all
imported unchanged from 04_multivariate_analyses.py.

DATA SOURCE
-----------
run-NN/1stLevel.feat/reg_standard/stats/cope{15,16,17,18}.nii.gz
Verified same grid as HighLevel.gfeat: (176,256,256) @ 1mm, so the existing
sphere and DOWNSAMPLE_FAC apply without change.

Run pairs are averaged over all available pairings (1-2, 1-3, 2-3 for 3 runs).
Nordt used exactly 2 runs per session; --n-runs 2 reproduces that.

Output: D_liu/rsa_nordt_crossrun.csv
  one row per subject x session x hemi x ROI x category
  within_r / between_r / distinctiveness  (raw r, Nordt convention)
  within_z / between_z / distinctiveness_z  (Fisher-z)
  plus n_runs_used, n_run_pairs, n_rsa_voxels

Usage:
  python rsa_nordt_crossrun.py
  python rsa_nordt_crossrun.py --n-runs 2
"""
import argparse
import glob
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
SRC = GIT / 'D_liu' / 'verified' / '04_multivariate_analyses.py'
OUT = GIT / 'D_liu' / 'rsa_nordt_crossrun.csv'

sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('mv', str(SRC))
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

CATS = list(mv.RSA_COPES.keys())          # face, house, object, word
COPES = [mv.RSA_COPES[c] for c in CATS]   # 15, 16, 17, 18


def run_dirs(sid, session):
    base = mv.BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc'
    return sorted(glob.glob(str(base / 'run-*')))


def run_patterns(run_dir, sphere_2mm):
    """4 category patterns from one run's reg_standard copes. None if incomplete."""
    st = Path(run_dir) / '1stLevel.feat' / 'reg_standard' / 'stats'
    pats = []
    for cope in COPES:
        f = st / f'cope{cope}.nii.gz'
        if not f.exists():
            return None
        v = zoom(mv._load(f).get_fdata(), mv.DOWNSAMPLE_FAC, order=1)
        shp = tuple(min(a, b) for a, b in zip(v.shape, sphere_2mm.shape))
        pats.append(v[:shp[0], :shp[1], :shp[2]][
            sphere_2mm[:shp[0], :shp[1], :shp[2]]])
    n = min(len(p) for p in pats)
    return np.column_stack([p[:n] for p in pats])      # voxels x 4


def zscore_cols(M):
    """z-score each category pattern across voxels (Nordt Methods)."""
    mu = M.mean(0, keepdims=True)
    sd = M.std(0, ddof=0, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    return (M - mu) / sd


def cross_run_rsm(A, B):
    """4x4 matrix of corr(A_i, B_j). Rows = run A category, cols = run B."""
    A = zscore_cols(A)
    B = zscore_cols(B)
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    if A.shape[0] < 20:
        return None, 0
    n = A.shape[0]
    # columns already zero-mean unit-sd -> corr = dot / n
    R = (A.T @ B) / n
    return np.clip(R, -0.999, 0.999), n


def distinctiveness_from_rsm(R):
    """Nordt: within (diagonal) minus mean between (off-diagonal), symmetrised."""
    k = R.shape[0]
    Rs = (R + R.T) / 2.0
    within = np.diag(Rs).copy()
    between = np.array([
        np.mean([Rs[i, j] for j in range(k) if j != i]) for i in range(k)])
    return within, between


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-runs', type=int, default=0,
                    help='use only the first N runs per session (0 = all)')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    subs = mv.load_subjects()
    print(f'Subjects: {len(subs)}   categories: {CATS}   copes: {COPES}')
    print(f'ROIs: {list(mv.ROIS.keys())}')
    print(f'n-runs: {"all available" if args.n_runs == 0 else args.n_runs}\n')

    rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i+1}/{len(subs)}] {info["code"]} ({time.time()-t0:.0f}s)      ',
              end='\r', flush=True)
        is_ctrl = info['patient_status'] == 'control'
        for session in info['sessions']:
            rds = run_dirs(sid, session)
            if args.n_runs:
                rds = rds[:args.n_runs]
            if len(rds) < 2:
                continue
            for roi in mv.ROIS:
                hemis = mv.CONTROL_HEMIS if is_ctrl else [info['patient_hemi']]
                for hemi in hemis:
                    peak = mv.find_peak(sid, session, roi, hemi, info)
                    if peak is None:
                        continue
                    sph = mv.create_sphere(peak['peak_coord'], peak['affine'],
                                           peak['brain_shape'])
                    s2 = zoom(sph.astype(float), mv.DOWNSAMPLE_FAC, order=0) > 0.5

                    pats = {}
                    for rd in rds:
                        P = run_patterns(rd, s2)
                        if P is not None:
                            pats[Path(rd).name] = P
                    if len(pats) < 2:
                        continue

                    wl, bl, nv = [], [], []
                    for a, b in itertools.combinations(sorted(pats), 2):
                        R, n = cross_run_rsm(pats[a], pats[b])
                        if R is None:
                            continue
                        w, bt = distinctiveness_from_rsm(R)
                        wl.append(w); bl.append(bt); nv.append(n)
                    if not wl:
                        continue
                    W = np.mean(wl, axis=0)
                    B = np.mean(bl, axis=0)

                    hemi_label = (('left' if hemi == 'l' else 'right') if is_ctrl
                                  else ('intact' if hemi == info['patient_hemi']
                                        else 'lesioned'))
                    for ci, cat in enumerate(CATS):
                        rows.append({
                            'subject_id': sid,
                            'code': info['code'],
                            'session': session,
                            'group': 'control' if is_ctrl else info['group'],
                            'status': info['patient_status'],
                            'surgery_side': info['surgery_side'],
                            'intact_hemi': info['intact_hemi'],
                            'hemi': hemi,
                            'hemi_label': hemi_label,
                            'roi': roi,
                            'category': cat,
                            'n_runs_used': len(pats),
                            'n_run_pairs': len(wl),
                            'n_rsa_voxels': int(np.mean(nv)),
                            'peak_z': peak['peak_z'],
                            'within_r': float(W[ci]),
                            'between_r': float(B[ci]),
                            'distinctiveness': float(W[ci] - B[ci]),
                            'within_z': float(np.arctanh(W[ci])),
                            'between_z': float(np.arctanh(B[ci])),
                            'distinctiveness_z': float(np.arctanh(W[ci]) -
                                                       np.arctanh(B[ci])),
                        })
        mv._CACHE.clear()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\n\nSaved: {args.out}  ({len(df)} rows, '
          f'{df["subject_id"].nunique()} subjects)')

    print('\nRuns used per session:')
    print(df.drop_duplicates(['subject_id', 'session']).n_runs_used
          .value_counts().sort_index().to_string())
    print('\nVoxels entering RSA, by ROI (median):')
    print(df.groupby('roi').n_rsa_voxels.median().round(0).to_string())

    prim = ['object_LOC', 'house_PPA_strict', 'face_FFA', 'word_VWFA']
    d = df[df.roi.isin(prim)]
    d = d[d.roi.str.split('_').str[0] == d.category]     # preferred category only
    print('\nCONTROL means, preferred category only  '
          '(distinctiveness: HIGHER = MORE distinct)')
    print(d[d.group == 'control']
          .pivot_table(index='roi', columns='hemi',
                       values=['within_r', 'between_r', 'distinctiveness'])
          .round(3).to_string())


if __name__ == '__main__':
    main()
