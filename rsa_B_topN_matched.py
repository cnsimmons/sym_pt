#!/usr/bin/env python3
"""
rsa_B_topN_matched.py — RSA on the top-N most selective voxels per ROI,
                        with N matched across ROIs.

Motivation: the 7mm peak-centred sphere is ~168 voxels at 2mm regardless of ROI
size. That is ~1% of the suprathreshold object_LOC region but ~20-30% of
face_FFA / word_VWFA / house_PPA_strict. Matching N removes ROI-size as a
confound and selects by selectivity rather than by proximity to one peak.

Selection is done AT 2mm (the RSA resolution) so exactly N voxels enter every
correlation. Ranking uses the same ROI-defining contrast as find_peak
(ROIS[roi] -> cope, negate), restricted to the ROI searchmask.

CIRCULARITY CAVEAT — read before interpreting:
  Selection contrasts (copes 1/2/3/13) and RSA copes (15-18) come from the same
  runs. Choosing voxels that respond strongly to e.g. Face>Object and then
  measuring how face and object patterns correlate is not independent. A clean
  version needs split-half (select on odd runs, test on even), which halves the
  data. Treat this script as a convergence check on the sphere result, not as a
  replacement primary analysis.

Output: D_liu/rsa_topN_matched.csv
        same schema as rsa_v1.csv plus 'n_requested' and 'n_rsa_voxels'.

Usage:
  python rsa_B_topN_matched.py --report-only          # available voxels per ROI
  python rsa_B_topN_matched.py --n 150
  python rsa_B_topN_matched.py --n 100 150 200
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
SRC = GIT / 'D_liu' / 'verified' / '04_multivariate_analyses.py'
OUT = GIT / 'D_liu' / 'rsa_topN_matched.csv'

sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('mv', str(SRC))
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)


def selection_map_2mm(sid, session, roi, hemi, info):
    """Return (z_2mm, mask_2mm) for the ROI-defining contrast, at RSA resolution.
    Reuses mv._load_searchmask and mv._load_zstat unchanged."""
    anchor = info['anchor_ses']
    cope, neg = mv.ROIS[roi]
    mask, aff = mv._load_searchmask(sid, anchor, roi, hemi)
    if mask is None:
        return None, None
    z = mv._load_zstat(sid, session, anchor, cope, neg)
    if z is None:
        return None, None
    z2 = zoom(z, mv.DOWNSAMPLE_FAC, order=1)
    m2 = zoom(mask.astype(float), mv.DOWNSAMPLE_FAC, order=0) > 0.5
    # guard against off-by-one shape drift from independent zooms
    shp = tuple(min(a, b) for a, b in zip(z2.shape, m2.shape))
    z2 = z2[:shp[0], :shp[1], :shp[2]]
    m2 = m2[:shp[0], :shp[1], :shp[2]]
    return z2, m2


def load_rsa_volumes_2mm(sid, session, info):
    """Return (list_of_2mm_volumes, category_names) for the 4 RSA copes."""
    anchor = info['anchor_ses']
    feat = (mv.BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' /
            'loc' / 'HighLevel.gfeat')
    cn = 'cope1.nii.gz' if session == anchor else f'cope1_ses{anchor}.nii.gz'
    vols, valid = [], []
    for cat, cope in mv.RSA_COPES.items():
        cf = feat / f'cope{cope}.feat' / 'stats' / cn
        if not cf.exists():
            continue
        vols.append(zoom(mv._load(cf).get_fdata(), mv.DOWNSAMPLE_FAC, order=1))
        valid.append(cat)
    return vols, valid


def extract_topN(z2, m2, vols, n):
    """Top-n voxels of z2 within m2; return (n_used x n_cat) matrix or None."""
    shp = tuple(min(dim) for dim in zip(z2.shape, *[v.shape for v in vols]))
    z2c = z2[:shp[0], :shp[1], :shp[2]]
    m2c = m2[:shp[0], :shp[1], :shp[2]]
    vc = [v[:shp[0], :shp[1], :shp[2]] for v in vols]

    cand = m2c & np.isfinite(z2c)
    for v in vc:
        cand &= np.isfinite(v)
    n_avail = int(cand.sum())
    if n_avail == 0:
        return None, 0

    zf = np.where(cand, z2c, -np.inf).ravel()
    k = min(n, n_avail)
    top = np.argpartition(zf, -k)[-k:]

    M = np.column_stack([v.ravel()[top] for v in vc])
    keep = np.isfinite(M).all(1)
    return M[keep], n_avail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[150],
                    help='matched voxel count(s)')
    ap.add_argument('--report-only', action='store_true',
                    help='print available voxels per ROI and exit')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    subs = mv.load_subjects()
    print(f'Subjects: {len(subs)}   N values: {args.n}')
    print(f'ROIs: {list(mv.ROIS.keys())}\n')

    rows, avail = [], []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i+1}/{len(subs)}] {info["code"]} ({time.time()-t0:.0f}s)   ',
              end='\r', flush=True)
        is_ctrl = info['patient_status'] == 'control'
        for session in info['sessions']:
            vols, valid = load_rsa_volumes_2mm(sid, session, info)
            if len(vols) < 4:
                continue
            for roi in mv.ROIS:
                hemis = mv.CONTROL_HEMIS if is_ctrl else [info['patient_hemi']]
                for hemi in hemis:
                    z2, m2 = selection_map_2mm(sid, session, roi, hemi, info)
                    if z2 is None:
                        continue
                    peak = mv.find_peak(sid, session, roi, hemi, info)
                    if peak is None:
                        continue
                    for n_req in args.n:
                        M, n_av = extract_topN(z2, m2, vols, n_req)
                        avail.append(dict(category=roi, hemi=hemi,
                                          n_available=n_av))
                        if args.report_only or M is None or M.shape[0] < 20:
                            continue
                        dist, pairs = mv.compute_rdm(M, valid, roi)
                        if not pairs:
                            continue
                        hemi_label = (('left' if hemi == 'l' else 'right') if is_ctrl
                                      else ('intact' if hemi == info['patient_hemi']
                                            else 'lesioned'))
                        base = {
                            'subject_id': sid,
                            'code': info['code'],
                            'session': session,
                            'group': 'control' if is_ctrl else info['group'],
                            'status': info['patient_status'],
                            'surgery_side': info['surgery_side'],
                            'intact_hemi': info['intact_hemi'],
                            'hemi': hemi,
                            'hemi_label': hemi_label,
                            'category': roi,
                            'n_requested': n_req,
                            'n_available': n_av,
                            'n_rsa_voxels': int(M.shape[0]),
                            'peak_x_native': peak['peak_coord'][0],
                            'peak_y_native': peak['peak_coord'][1],
                            'peak_z_native': peak['peak_coord'][2],
                            'peak_z': peak['peak_z'],
                            'liu_distinctiveness': dist,
                        }
                        for pr, fz in pairs.items():
                            rows.append({**base, 'pair': pr, 'fisher_r': fz})
        mv._CACHE.clear()

    A = pd.DataFrame(avail).drop_duplicates()
    print('\n\nAvailable voxels at 2mm within searchmask, by ROI x hemi:')
    print(A.groupby(['category', 'hemi']).n_available
          .agg(['min', 'median', 'max']).round(0).to_string())

    if args.report_only:
        print('\n--report-only: no RSA computed. '
              'Pick --n at or below the smallest median above.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nSaved: {args.out}  ({len(df)} rows, '
          f'{df["subject_id"].nunique()} subjects)')

    summ = df.drop(columns=['pair', 'fisher_r']).drop_duplicates()
    print('\nVoxels actually used, by ROI x n_requested (median):')
    print(summ.pivot_table(index='category', columns='n_requested',
                           values='n_rsa_voxels', aggfunc='median')
          .round(0).to_string())
    print('\nDistinctiveness by ROI x n_requested (control mean):')
    ct = summ[summ.group == 'control']
    print(ct.pivot_table(index='category', columns='n_requested',
                         values='liu_distinctiveness', aggfunc='mean')
          .round(3).to_string())


if __name__ == '__main__':
    main()
