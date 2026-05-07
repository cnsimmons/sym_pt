#!/usr/bin/env python3
"""Per-ROI volume (suprathreshold voxels) for ALL 4 category contrasts.
   Output: cross_contrast_volumes.csv — columns:
     subject_id, session, hemi, roi, contrast_cat, volume, n_searchmask
"""
import sys, time
import numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions, get_sub_info, _load_csv, is_patient

# Reuse v2 conventions
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from D_liu.liu_recreation_csv_v2 import (
    CONTRASTS, EXTRA_SKIP, PRE_SURGERY_SESSIONS, SEL_Z_THRESH,
    _load_searchmask, _load_zstat, load_subjects, _CACHE
)

BASE_DIR = Path(processed_dir)
OUT      = Path('/user_data/csimmon2/git_repos/sym_pt/cross_contrast_volumes.csv')

# Each ROI (key) is probed by ALL 4 category contrasts (values)
PROBE_CONTRASTS = {
    'face':   (1,  False),  # Face>Object
    'house':  (2,  False),  # House>Object
    'object': (3,  False),  # Object>Scramble
    'word':   (13, True),   # Word>Face
}

def main():
    subs = load_subjects()
    rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i+1}/{len(subs)}] {info["code"]} ({time.time()-t0:.0f}s)', end='\r')
        is_ctrl = info['patient_status'] == 'control'
        first_ses = info['sessions'][0]

        for session in info['sessions']:
            if session in PRE_SURGERY_SESSIONS.get(sid, set()):
                continue
            hemis = ['l','r'] if is_ctrl else [info['patient_hemi']]
            for hemi in hemis:
                for roi in CONTRASTS:
                    mask, _ = _load_searchmask(sid, first_ses, roi, hemi)
                    if mask is None: continue
                    n_mask = int(mask.sum())
                    for cat, (cope, neg) in PROBE_CONTRASTS.items():
                        z = _load_zstat(sid, session, first_ses, cope, neg)
                        if z is None:
                            rows.append({'subject_id':sid,'session':session,'hemi':hemi,
                                         'roi':roi,'contrast_cat':cat,'volume':np.nan,
                                         'n_searchmask':n_mask})
                            continue
                        vol = int((mask & (z > SEL_Z_THRESH)).sum())
                        rows.append({'subject_id':sid,'session':session,'hemi':hemi,
                                     'roi':roi,'contrast_cat':cat,'volume':vol,
                                     'n_searchmask':n_mask})
        _CACHE.clear()

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f'\nSaved: {OUT} ({len(df)} rows)')

if __name__ == '__main__':
    main()