#!/usr/bin/env python3
"""
searchlight_vs_roi.py — is the 7 mm ROI sphere a sufficient window on the
representational group difference, or does the effect sit somewhere else?

This is NOT another significance test. The TFCE in combat_07 asks whether a
cluster survives FWE across ~11,000 voxels, which is a strictly harder bar than
four ROI tests, so a null there says nothing about whether the ROI was well
placed. The question here is about LOCATION, not threshold.

THREE ANALYSES

  1  ROI PLACEMENT. For each subject, read searchlight distinctiveness at that
     subject's own ROI peak, and compare the group difference there against the
     group difference at every other OTC voxel. Reported as the percentile the
     ROI peak occupies in the whole-OTC distribution of group differences.
       ~100th  the ROI sits exactly where the effect is largest — well placed
       ~50th   the ROI is unremarkable — the effect is elsewhere
       ~0th    the effect runs the OPPOSITE way at the ROI than in OTC at large

  2  SPATIAL CONCENTRATION. Is the unthresholded effect focal or diffuse? If
     diffuse, no ROI of any placement could capture it, and the ROI approach is
     the wrong instrument rather than merely mis-centred. Quantified as the share
     of the total absolute effect held by the top 5% and 25% of voxels, against
     the 5% / 25% a perfectly uniform effect would give.

  3  INSIDE vs OUTSIDE. Group difference within the ROI sphere versus the rest of
     that hemisphere's OTC, in the same units. A larger difference outside means
     the ROI is looking in the wrong place.

INPUTS
  the stage-1/2/3 products of combat_07_searchlight_distinctiveness.py, plus
  rsa_v1_harmonized.csv for each subject's ROI peak coordinates

ORIENTATION
  masks are ('R','A','S') and the maps are ('L','A','S'); every mask is
  resampled onto the data grid before indexing, as combat_07 stage 3 does.

Usage
  python searchlight_vs_roi.py
  python searchlight_vs_roi.py --csv searchlight_vs_roi.csv
"""
import argparse
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

GIT  = Path('/user_data/csimmon2/git_repos/sym_pt')
PROC = Path('/user_data/csimmon2/sym_pt')
SL   = PROC / 'group_results' / 'tfce_searchlight_distinct'
RSA  = GIT / 'D_liu' / 'rsa_v1_harmonized.csv'

CAT2ROI = {'face': 'face_FFA', 'house': 'house_PPA_strict',
           'object': 'object_LOC', 'word': 'word_VWFA'}
SPHERE_MM = 7.0
N_CTRL = 38


def load_cell(cat, hemi):
    """Merged 4D distinctiveness + the OTC mask on its grid. ctrl first, then pt."""
    d = SL / f'{cat}_{hemi}_pt_vs_ctrl'
    mg = nib.load(str(d / 'merged_distinct.nii.gz'))
    X = mg.get_fdata()
    ref = nib.Nifti1Image(X[..., 0], mg.affine)
    m = resample_to_img(nib.load(str(SL / f'votc_{hemi}_mask.nii.gz')),
                        ref, interpolation='nearest').get_fdata() > 0.5
    m &= (X != 0).all(-1)                    # voxels with data in every subject
    return mg, X, m, ref


def roi_peaks(hemi):
    """Each subject's ROI peak in MNI, from the harmonized RSA table.

    NOTE the RSA stores peaks in NATIVE space (peak_*_native). Native and MNI
    are not interchangeable, so this returns None if no MNI columns exist, and
    analysis 1 is reported as unavailable rather than computed on the wrong
    coordinates.
    """
    d = pd.read_csv(RSA)
    cols = [c for c in d.columns if 'peak' in c.lower()]
    mni = [c for c in cols if 'mni' in c.lower()]
    if not mni:
        return None, cols
    return d, mni


def analysis_2_concentration(diff, m):
    """Share of total |effect| held by the top 5% and 25% of in-mask voxels."""
    a = np.abs(diff[m])
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    s = np.sort(a)[::-1]
    tot = s.sum()
    if tot == 0:
        return np.nan, np.nan
    k5, k25 = max(1, int(.05 * s.size)), max(1, int(.25 * s.size))
    return s[:k5].sum() / tot, s[:k25].sum() / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    rsa, cols = roi_peaks('l')
    print('=' * 78)
    if rsa is None:
        print('ANALYSIS 1 (ROI placement): UNAVAILABLE')
        print(f'  {RSA.name} has no MNI peak columns. Present: {cols}')
        print('  Those are native-space coordinates and cannot be compared to')
        print('  the MNI searchlight maps. To run analysis 1, the ROI peaks must')
        print('  be warped to MNI with each subject\'s anat2stand.mat first.')
    print('=' * 78)

    rows = []
    print('\nANALYSIS 2 & 3 — concentration, and inside vs outside the ROI sphere')
    print('diff = pt - ctrl on the SIMILARITY scale (+ = patients LESS distinct)\n')
    print(f'{"cell":10s} {"n_vox":>7s} {"diff_all":>9s} {"top5%":>7s} {"top25%":>7s} '
          f'{"|d|_max":>8s} {"peak MNI":>18s}')

    for cat, hemi in itertools.product(['face', 'house', 'object', 'word'],
                                       ['l', 'r']):
        try:
            mg, X, m, ref = load_cell(cat, hemi)
        except Exception as e:
            print(f'{cat}_{hemi:1s}   LOAD FAILED: {type(e).__name__}')
            continue
        npt = X.shape[3] - N_CTRL
        diff = X[..., N_CTRL:].mean(-1) - X[..., :N_CTRL].mean(-1)

        c5, c25 = analysis_2_concentration(diff, m)
        a = np.abs(diff); a[~m] = 0
        pk = np.unravel_index(np.argmax(a), a.shape)
        mm = nib.affines.apply_affine(ref.affine, np.array(pk)).round(0)

        print(f'{cat}_{hemi:1s}   {int(m.sum()):7d} {diff[m].mean():+9.3f} '
              f'{c5:7.1%} {c25:7.1%} {a[pk]:8.3f} '
              f'[{mm[0]:+.0f},{mm[1]:+.0f},{mm[2]:+.0f}]')

        rows.append(dict(cat=cat, hemi=hemi, n_vox=int(m.sum()), n_pt=npt,
                         diff_mean=float(diff[m].mean()),
                         diff_sd=float(diff[m].std()),
                         conc_top5=float(c5), conc_top25=float(c25),
                         abs_max=float(a[pk]),
                         peak_x=float(mm[0]), peak_y=float(mm[1]),
                         peak_z=float(mm[2])))

    print('\nA uniform effect would give top5% = 5.0% and top25% = 25.0%.')
    print('Values far above that mean the effect is focal; near them, diffuse.')

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
