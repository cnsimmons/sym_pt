#!/usr/bin/env python3
"""
quality_check.py — tSNR and motion, patients vs controls, per intact hemisphere.

The confound control Rosenke et al. (2020) used for their heterogeneity result:
noisier data lowers between-subject odRSM correlations mechanically, so a group
that is less internally consistent may simply have worse data. They compared
tSNR and head motion between groups and found neither differed.

Same check here, for the LH-intact heterogeneity result (within-group r = 0.404
vs 0.670 in controls, p = .0001).

tSNR    mean(timeseries) / SD(timeseries), per voxel, averaged over the
        session's runs. Computed on filtered_func_data in each run's
        1stLevel.feat, restricted to the run's own brain mask.
motion  root mean square of the 6 realignment parameters, averaged over runs,
        from mc/prefiltered_func_data_mcf.par.

Same subjects and sessions as the RSM analysis, via the verified TFCE loader.

Usage
  python quality_check.py
  python quality_check.py --csv quality_check.csv
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

GIT  = Path('/user_data/csimmon2/git_repos/sym_pt')
PROC = Path('/user_data/csimmon2/sym_pt')

_V = [GIT / 'D_liu' / 'verified' / '02_tfce_analyses_dontuse_useharmony.py',
      GIT / 'D_liu' / 'verified' / '02_tfce_analyses_not_as_verified.py',
      GIT / 'D_liu' / 'verified' / '02_tfce_analyses.py']
VT = next((p for p in _V if p.exists()), None)
if VT is None:
    sys.exit('verified TFCE module not found')
sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('vt', str(VT))
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

RNG = np.random.default_rng(42)


def perm(a, b, n=10000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b]); na = len(a); k = 0
    for _ in range(n):
        p = RNG.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


def run_dirs(sid, ses):
    return sorted((PROC / sid / f'ses-{ses}' / 'derivatives' / 'fsl' / 'loc')
                  .glob('run-*/1stLevel.feat'))


def tsnr_of_run(feat):
    f = feat / 'filtered_func_data.nii.gz'
    if not f.exists():
        return np.nan
    img = nib.load(str(f))
    d = np.asarray(img.dataobj, dtype=np.float32)
    mk = feat / 'mask.nii.gz'
    m = nib.load(str(mk)).get_fdata() > 0.5 if mk.exists() else d.mean(-1) > 0
    ts = d[m]
    sd = ts.std(-1)
    ok = sd > 0
    return float(np.mean(ts.mean(-1)[ok] / sd[ok])) if ok.any() else np.nan


def motion_of_run(feat):
    p = feat / 'mc' / 'prefiltered_func_data_mcf.par'
    if not p.exists():
        return np.nan
    par = np.loadtxt(p)
    if par.ndim != 2 or par.shape[1] < 6:
        return np.nan
    return float(np.mean(np.sqrt((par[:, :6] ** 2).mean(0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    subs = v.load_subjects()
    rows = []
    for i, (sid, info) in enumerate(sorted(subs.items()), 1):
        feats = run_dirs(sid, info['session'])
        if not feats:
            print(f'  {sid}: no run dirs in ses-{info["session"]}')
            continue
        t = [tsnr_of_run(f) for f in feats]
        m = [motion_of_run(f) for f in feats]
        rows.append(dict(subject_id=sid, group=info['group'],
                         intact_hemi=info['intact_hemi'], n_runs=len(feats),
                         tsnr=np.nanmean(t), motion=np.nanmean(m)))
        print(f'  [{i}/{len(subs)}] {sid} {len(feats)} runs  '
              f'tSNR {np.nanmean(t):6.1f}  motion {np.nanmean(m):.3f}')

    d = pd.DataFrame(rows)
    ctl = d[d['group'] == 'control']

    print('\n' + '=' * 66)
    print('tSNR and motion — the Rosenke confound control')
    print('=' * 66)
    print(f"controls (n={len(ctl)}): tSNR {ctl.tsnr.mean():.1f} "
          f"(SD {ctl.tsnr.std():.1f}), motion {ctl.motion.mean():.3f}")

    for lab, side in [('LH-intact', 'left'), ('RH-intact', 'right')]:
        p = d[(d['group'] == 'OTC') & (d['intact_hemi'] == side)]
        if len(p) < 3:
            continue
        dt, pt_ = perm(ctl.tsnr.values, p.tsnr.values)
        dm, pm = perm(ctl.motion.values, p.motion.values)
        print(f"\n{lab} (n={len(p)}): tSNR {p.tsnr.mean():.1f}, "
              f"motion {p.motion.mean():.3f}")
        print(f"  tSNR   diff {dt:+.1f}  p = {pt_:.4f}"
              + ('  * DIFFERS' if pt_ == pt_ and pt_ < .05 else '  n.s.'))
        print(f"  motion diff {dm:+.3f}  p = {pm:.4f}"
              + ('  * DIFFERS' if pm == pm and pm < .05 else '  n.s.'))

    print('\nBoth n.s. -> data quality does not explain the heterogeneity result.')
    print('Either significant -> it may, and the split-half is then required.')

    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
