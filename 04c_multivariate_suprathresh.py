#!/usr/bin/env python3
"""
04c_multivariate_suprathresh.py — distinctiveness on ALL suprathreshold voxels.

Same pipeline as D_liu/verified/04_multivariate_analyses.py with one change:
the RSA voxel set is (searchmask & zstat > SEL_Z) instead of a 7mm sphere at
the peak. Purpose is to test whether object_LOC's representational null is a
sampling artifact of a fixed 168-voxel sphere covering ~1% of a large region
while the same sphere covers ~20-30% of the smaller regions.

The thresholding contrast per ROI is the SAME differential contrast used for
peak-finding and for the univariate selective-voxel count, so the voxel set
here matches the one the count measure operates on.

DISTINCTIVENESS ONLY. Geometry pairs are not written: each category's mask is
a different voxel set, so the six pairwise correlations would be computed on
non-overlapping voxels. That needs a separate decision (union mask, or the
preferred category's mask for all six pairs).

Writes nothing unless --csv is passed.

Usage:
  python 04c_multivariate_suprathresh.py            # print only
  python 04c_multivariate_suprathresh.py --csv      # write rsa_suprathresh.csv
"""
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
SRC  = REPO / 'D_liu' / 'verified' / '04_multivariate_analyses.py'
OUT  = REPO / 'D_liu' / 'rsa_suprathresh.csv'

SEL_Z   = 2.326      # same threshold as the univariate selective-voxel count
MIN_VOX = 30         # 2mm voxels; cells below this are dropped and reported

sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location('mv04', SRC)
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

ROIS_USED = ['face_FFA', 'house_PPA', 'object_LOC', 'word_VWFA']
if hasattr(mv, 'ROIS'):
    ROIS_USED = [r for r in ROIS_USED if r in mv.ROIS]
print('ROIs:', ROIS_USED)
print('RSA copes:', mv.RSA_COPES)
print('threshold contrasts:', {r: mv.ROIS[r] for r in ROIS_USED})


def suprathresh_mask(sid, session, roi, hemi, info):
    """searchmask & (zstat > SEL_Z), at 1mm native, same contrast as peak-finding."""
    anchor = info['anchor_ses']
    cope, neg = mv.ROIS[roi]
    mask, aff = mv._load_searchmask(sid, anchor, roi, hemi)
    if mask is None:
        return None, None, None
    z = mv._load_zstat(sid, session, anchor, cope, neg)
    if z is None:
        return None, None, None
    supra = mask & (z > SEL_Z)
    return supra, aff, z.shape


rows, skipped = [], []
subjects = mv.load_subjects()

for sid, info in subjects.items():
    hemis = [info['patient_hemi']] if info['patient_status'] == 'patient' else ['l', 'r']
    hemis = [h for h in hemis if h]
    for session in info['sessions'] if isinstance(info['sessions'], (list, tuple)) else [info['sessions']]:
        ses = f'{int(session):02d}' if not isinstance(session, str) else session
        for hemi in hemis:
            for roi in ROIS_USED:
                supra, aff, shape = suprathresh_mask(sid, ses, roi, hemi, info)
                if supra is None:
                    skipped.append((sid, ses, hemi, roi, 'no mask/zstat'))
                    continue
                n1mm = int(supra.sum())
                if n1mm == 0:
                    skipped.append((sid, ses, hemi, roi, 'zero suprathreshold'))
                    continue
                betas, valid = mv.extract_betas(sid, ses, supra, info)
                if betas is None:
                    skipped.append((sid, ses, hemi, roi, 'beta extraction failed'))
                    continue
                n2mm = betas.shape[0]
                if n2mm < MIN_VOX:
                    skipped.append((sid, ses, hemi, roi, f'{n2mm} vox < {MIN_VOX}'))
                    continue
                dist, pairs = mv.compute_rdm(betas, valid, roi)
                rows.append(dict(
                    subject_id=sid, session=ses, hemi=hemi, roi=roi,
                    group=info['group'], status=info['patient_status'],
                    intact_hemi=info['intact_hemi'],
                    surgery_side=info['surgery_side'],
                    n_vox_1mm=n1mm, n_vox_2mm=n2mm,
                    liu_distinctiveness=dist,
                ))

df = pd.DataFrame(rows)
print(f'\nrows: {len(df)}   skipped: {len(skipped)}')
if skipped:
    s = pd.DataFrame(skipped, columns=['sid', 'ses', 'hemi', 'roi', 'why'])
    print('\nskip reasons:')
    print(s.groupby(['roi', 'why']).size().to_string())
    print('\nskipped cells:')
    print(s.to_string(index=False))

if len(df):
    print('\n--- voxel counts by ROI x hemi (2mm) ---')
    print(df.groupby(['roi', 'hemi'])['n_vox_2mm']
            .agg(['count', 'median', 'min', 'max']).to_string())
    print('\n--- distinctiveness: mean (sd), n ---')
    g = (df.groupby(['status', 'hemi', 'roi'])['liu_distinctiveness']
           .agg(['count', 'mean', 'std']).round(3))
    print(g.to_string())

if '--csv' in sys.argv and len(df):
    df.to_csv(OUT, index=False)
    print('\nwrote', OUT)
else:
    print('\n(no --csv, nothing written)')
