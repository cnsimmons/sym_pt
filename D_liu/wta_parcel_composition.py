"""1a composition diagnostic. Reuses 04_multivariate_analyses geometry voxels
(same 2mm sphere) and scores WTA winner per voxel from category zstats.
Read-only except its own output CSV. Session rule: pt=last, ctrl=first."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import zoom

V = '/user_data/csimmon2/git_repos/sym_pt/D_liu/verified/04_multivariate_analyses.py'
spec = importlib.util.spec_from_file_location('mv', V)
mv = importlib.util.module_from_spec(spec); spec.loader.exec_module(mv)

CATS  = ['face', 'house', 'object', 'word']
COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}   # zstat copes (from 03)
WTA_THRESHOLD = 2.326                                      # from 03
DF    = mv.DOWNSAMPLE_FAC
OUT   = Path('/user_data/csimmon2/git_repos/sym_pt/D_liu/wta_parcel_composition.csv')

def winner_in_sphere(sid, session, info, sphere_2mm):
    """Per-voxel argmax over 4 category zstats on the SAME 2mm sphere geometry used."""
    anchor = info['anchor_ses']
    stack = []
    for cat in CATS:
        z = mv._load_zstat(sid, session, anchor, COPES[cat], negate=False)
        if z is None:
            return None
        z2 = zoom(z, DF, order=1)                 # match extract_betas resampling
        if z2.shape != sphere_2mm.shape:
            return None
        stack.append(z2[sphere_2mm])
    Z = np.column_stack(stack)                    # voxels x 4
    finite = np.isfinite(Z).all(axis=1)
    Z = Z[finite]
    if len(Z) == 0:
        return None
    maxz = Z.max(axis=1)
    win  = Z.argmax(axis=1)                        # 0..3
    n = len(Z)
    n_sel = int((maxz >= WTA_THRESHOLD).sum())
    row = {'n_vox': n, 'n_selective': n_sel}
    for i, cat in enumerate(CATS):
        row[f'{cat}_raw'] = 100.0 * (win == i).sum() / n
        row[f'{cat}_sel'] = (100.0 * ((win == i) & (maxz >= WTA_THRESHOLD)).sum() / n_sel
                             if n_sel > 0 else np.nan)
    return row

def main():
    subs = mv.load_subjects()
    rows = []
    for sid, info in subs.items():
        is_ctrl = info['patient_status'] == 'control'
        session = info['sessions'][0] if is_ctrl else info['sessions'][-1]   # ctrl first / pt last
        for roi in ['face_FFA', 'house_PPA', 'object_LOC', 'word_VWFA']:
            hemis = mv.CONTROL_HEMIS if is_ctrl else [info['patient_hemi']]
            for hemi in hemis:
                peak = mv.find_peak(sid, session, roi, hemi, info)
                if peak is None:
                    continue
                s1 = mv.create_sphere(peak['peak_coord'], peak['affine'], peak['brain_shape'])
                s2 = zoom(s1.astype(float), DF, order=0) > 0.5
                comp = winner_in_sphere(sid, session, info, s2)
                if comp is None:
                    continue
                rows.append({'subject_id': sid, 'status': info['patient_status'],
                             'group': info['group'], 'roi': roi, 'hemi': hemi,
                             'session': session, **comp})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT}  ({len(df)} rows)')

    # 1a summary: FFA & VWFA composition, pt vs ctrl, selective-denominator
    print('\n=== 1a: mean selective-winner % by parcel x status (FFA/VWFA are the ¶4 parcels) ===')
    for roi in ['face_FFA', 'word_VWFA', 'object_LOC', 'house_PPA']:
        sub = df[df.roi == roi]
        g = sub.groupby('status')[[f'{c}_sel' for c in CATS]].mean().round(1)
        print(f'\n[{roi}]  (n ctrl={sub[sub.status=="control"].subject_id.nunique()}, '
              f'pt={sub[sub.status=="patient"].subject_id.nunique()})')
        print(g.to_string())


if __name__ == "__main__":
    main()
