#!/usr/bin/env python3
"""
roi_sufficiency.py — is the 7 mm ROI sphere a sufficient window on the
representational group difference?

WHY THIS AND NOT THE TFCE
  The searchlight TFCE and the ROI RSA disagree: rVWFA gives d = -0.92 while
  word_r is flat across OTC (+0.005). Two very different explanations, and the
  TFCE cannot tell them apart:

    (a) the searchlight is WRONG — a bug, or its sphere does not reproduce the
        number the ROI analysis produces
    (b) the searchlight is RIGHT and the ROI is reading an unrepresentative spot

  This script separates them, in two steps.

STEP 1 — VALIDATION. Does the searchlight reproduce the ROI?
  The native searchlight maps and the ROI peak coordinates are BOTH in ses-01
  anat space, so no registration is involved. For each subject x ROI, read the
  searchlight map at that subject's own ROI peak and correlate against
  liu_distinctiveness from the RSA table.

  r ~ 1.0   the searchlight is measuring the same quantity. Any ROI/searchlight
            disagreement is then about LOCATION, and step 2 is interpretable.
  r low     the searchlight is not reproducing the ROI. Stop; the group-level
            comparison means nothing until that is resolved.

  Perfect agreement is not expected: the RSA builds its sphere at 1 mm and
  downsamples, while the searchlight works at 2 mm throughout, and the RSA table
  may be harmonized. Both attenuate r without being errors.

STEP 2 — SUFFICIENCY. Is the ROI peak where the effect actually is?
  Per ROI, take the group difference in the searchlight value AT each subject's
  own ROI peak, then compare it against the distribution of group differences
  across all OTC voxels. Reported as a percentile.

    ~100th  the ROI sits at the maximum — well placed, and sufficient
    ~50th   the ROI is unremarkable — the effect is elsewhere, ROI insufficient
    ~0th    the effect at the ROI runs OPPOSITE to the rest of OTC

  A high percentile with a null TFCE is not a contradiction. It means the effect
  is real and strongest at a functionally-defined location that does not line up
  across subjects in MNI, which is precisely the case for using ROIs.

Usage
  python roi_sufficiency.py
  python roi_sufficiency.py --csv roi_sufficiency.csv
  python roi_sufficiency.py --rsa D_liu/rsa_v1.csv     # unharmonized, tighter r
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

GIT  = Path('/user_data/csimmon2/git_repos/sym_pt')
PROC = Path('/user_data/csimmon2/sym_pt')
SL   = PROC / 'group_results' / 'tfce_searchlight_distinct'
NAT  = SL / 'native_maps'

CAT2ROI = {'face': 'face_FFA', 'house': 'house_PPA_strict',
           'object': 'object_LOC', 'word': 'word_VWFA'}
SPHERE_MM = 7.0
N_CTRL = 38


def sphere_mean(img_data, affine, peak_mm, radius=SPHERE_MM):
    """Mean of img_data within `radius` mm of peak_mm. Ignores zeros (no data)."""
    inv = np.linalg.inv(affine)
    ijk = nib.affines.apply_affine(inv, np.asarray(peak_mm, float))
    r_vox = radius / np.abs(np.diag(affine)[:3])
    lo = np.maximum(np.floor(ijk - r_vox).astype(int), 0)
    hi = np.minimum(np.ceil(ijk + r_vox).astype(int) + 1,
                    np.array(img_data.shape))
    if (lo >= hi).any():
        return np.nan, 0
    sub = img_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    gi, gj, gk = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
                             np.arange(lo[2], hi[2]), indexing='ij')
    world = nib.affines.apply_affine(
        affine, np.stack([gi.ravel(), gj.ravel(), gk.ravel()], 1))
    d = np.linalg.norm(world - np.asarray(peak_mm, float), axis=1)
    vals = sub.ravel()[d <= radius]
    vals = vals[np.isfinite(vals) & (vals != 0)]
    return (float(vals.mean()), int(vals.size)) if vals.size else (np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rsa', default=str(GIT / 'D_liu' / 'rsa_v1_harmonized.csv'))
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    rsa = pd.read_csv(args.rsa)
    need = ['peak_x_native', 'peak_y_native', 'peak_z_native',
            'liu_distinctiveness', 'subject_id', 'session', 'hemi', 'category']
    miss = [c for c in need if c not in rsa.columns]
    if miss:
        raise SystemExit(f'{Path(args.rsa).name} missing columns: {miss}')
    rsa = rsa.drop_duplicates(['subject_id', 'session', 'hemi', 'category'])
    print(f'{Path(args.rsa).name}: {len(rsa)} unique subject x session x hemi x ROI rows')

    # ---- STEP 1: does the searchlight reproduce the ROI? --------------------
    rows = []
    for cat, roi in CAT2ROI.items():
        for hemi in ['l', 'r']:
            sub = rsa[(rsa['category'] == roi) & (rsa['hemi'] == hemi)]
            for _, r in sub.iterrows():
                f = NAT / f"{r['subject_id']}_{cat}_{hemi}_distinct.nii.gz"
                if not f.exists():
                    continue
                img = nib.load(str(f))
                sl, n = sphere_mean(img.get_fdata(), img.affine,
                                    [r['peak_x_native'], r['peak_y_native'],
                                     r['peak_z_native']])
                rows.append(dict(subject_id=r['subject_id'], cat=cat, hemi=hemi,
                                 roi_rsa=r['liu_distinctiveness'],
                                 sl_at_peak=sl, n_vox=n))
    v = pd.DataFrame(rows)
    ok = v.dropna(subset=['roi_rsa', 'sl_at_peak'])

    print('\n' + '=' * 74)
    print('STEP 1  VALIDATION — searchlight at the ROI peak vs the ROI RSA value')
    print('=' * 74)
    if len(ok) < 10:
        print(f'  only {len(ok)} matched cells — cannot validate. '
              'Check that stage-1 native maps exist.')
        return
    print(f"{'cell':10s} {'n':>4s} {'r':>7s} {'mean SL':>9s} {'mean RSA':>9s} {'bias':>8s}")
    for cat in CAT2ROI:
        for hemi in ['l', 'r']:
            s = ok[(ok['cat'] == cat) & (ok['hemi'] == hemi)]
            if len(s) < 5:
                print(f'{cat}_{hemi:1s}   {len(s):4d}   (too few)')
                continue
            r = np.corrcoef(s['sl_at_peak'], s['roi_rsa'])[0, 1]
            print(f"{cat}_{hemi:1s}   {len(s):4d} {r:+7.3f} "
                  f"{s['sl_at_peak'].mean():9.3f} {s['roi_rsa'].mean():9.3f} "
                  f"{s['sl_at_peak'].mean() - s['roi_rsa'].mean():+8.3f}")
    rall = np.corrcoef(ok['sl_at_peak'], ok['roi_rsa'])[0, 1]
    print(f"\n  POOLED r = {rall:+.3f}  (n={len(ok)})")
    print('  r near 1 means the searchlight measures the same quantity as the ROI,')
    print('  so any disagreement between them is about WHERE, not WHAT.')

    # ---- STEP 2: is the ROI peak where the effect is? -----------------------
    print('\n' + '=' * 74)
    print('STEP 2  SUFFICIENCY — group difference at the ROI vs across OTC')
    print('=' * 74)
    print('diff = pt - ctrl, similarity scale (+ = patients LESS distinct)\n')
    print(f"{'cell':10s} {'nC':>3s} {'nP':>3s} {'diff@ROI':>9s} "
          f"{'OTC med':>8s} {'OTC max':>8s} {'pctile':>7s}")

    st = rsa[['subject_id', 'group']].drop_duplicates().set_index('subject_id')
    out = []
    for cat in CAT2ROI:
        for hemi in ['l', 'r']:
            s = ok[(ok['cat'] == cat) & (ok['hemi'] == hemi)].copy()
            s['grp'] = s['subject_id'].map(
                lambda x: 'OTC' if str(st['group'].get(x, '')).upper().find('OTC') >= 0
                else 'control')
            c = s.loc[s['grp'] == 'control', 'sl_at_peak']
            p = s.loc[s['grp'] == 'OTC', 'sl_at_peak']
            if len(c) < 5 or len(p) < 3:
                print(f'{cat}_{hemi:1s}   {len(c):3d} {len(p):3d}   (too few)')
                continue
            d_roi = p.mean() - c.mean()

            cell = SL / f'{cat}_{hemi}_pt_vs_ctrl' / 'merged_distinct.nii.gz'
            if not cell.exists():
                print(f'{cat}_{hemi:1s}   merged_distinct missing')
                continue
            mg = nib.load(str(cell)); X = mg.get_fdata()
            ref = nib.Nifti1Image(X[..., 0], mg.affine)
            m = resample_to_img(nib.load(str(SL / f'votc_{hemi}_mask.nii.gz')),
                                ref, interpolation='nearest').get_fdata() > 0.5
            m &= (X != 0).all(-1)
            dmap = X[..., N_CTRL:].mean(-1) - X[..., :N_CTRL].mean(-1)
            dv = dmap[m]
            dv = dv[np.isfinite(dv)]
            # percentile by absolute effect: where does the ROI rank?
            pct = 100.0 * (np.abs(dv) < abs(d_roi)).mean()
            print(f"{cat}_{hemi:1s}   {len(c):3d} {len(p):3d} {d_roi:+9.3f} "
                  f"{np.median(dv):+8.3f} {dv[np.argmax(np.abs(dv))]:+8.3f} "
                  f"{pct:6.1f}%")
            out.append(dict(cat=cat, hemi=hemi, n_ctrl=len(c), n_pt=len(p),
                            diff_at_roi=float(d_roi),
                            otc_median=float(np.median(dv)),
                            otc_absmax=float(dv[np.argmax(np.abs(dv))]),
                            roi_percentile=float(pct),
                            validation_r=float(np.corrcoef(
                                s['sl_at_peak'], s['roi_rsa'])[0, 1])))

    print('\n  pctile = share of OTC voxels whose |group difference| is SMALLER')
    print('  than the ROI\'s. High = the ROI sits where the effect is; low = the')
    print('  ROI is an unremarkable location and a sphere there is insufficient.')

    if args.csv:
        pd.DataFrame(out).to_csv(args.csv, index=False)
        v.to_csv(str(args.csv).replace('.csv', '_persubject.csv'), index=False)
        print(f'\nwrote {args.csv} and *_persubject.csv')


if __name__ == '__main__':
    main()
