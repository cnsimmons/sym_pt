#!/usr/bin/env python3
"""
1_univariate_analyses.py — Liu (2025) univariate cross-sectional + longitudinal extraction.

Univariate measures per subject × ROI × hemisphere:
  - peak        : max zstat within searchmask, 1mm anat space, no threshold
  - sum-sel     : Σ(z) over z>2.326 voxels in searchmask, normalized by
                  searchmask size, ×1000 (Ayzenberg 2023). Resolution-invariant.
  - vol         : count of suprathreshold (z>2.326) voxels in searchmask
  - magnitude   : mean z of those suprathreshold voxels (column: mean_act)

  → sum-sel jointly indexes magnitude AND spatial extent; vol and magnitude
    are its two separable components and are reported alongside it.

Multivariate measures (RSA: distinctiveness, between-category pairs) are NOT
computed here.

Contrast notes (divergences from Liu, documented for methods):
  - face/house: use >object rather than Liu's >house/>face
  - word: word>face via −cope13 (cope 13 is Face>Word)
  - EVC: cope 19 (Scramble_raw) — Liu methods don't specify EVC contrast

Output: D_liu/univariate_v1.csv (native-space coords)

Usage:
  python 1_univariate_analyses.py
"""
import sys
import time
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.stats import norm

# ── Corrected Import ─────────────────────────────────────────────────────────
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from params import (processed_dir, is_patient, should_skip,
                    get_sessions, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = Path(processed_dir)
OUTPUT_DIR  = Path('/user_data/csimmon2/git_repos/sym_pt/D_liu')
OUTPUT_NAME = 'univariate_v1.csv'

SEL_Z_THRESH = float(norm.ppf(0.99))            # ≈2.326, p<.01 one-tailed
SEL_RESCALE  = 1000.0

# GLM cope numbers (from FEAT design.con):
# 1=Face>Object, 2=House>Object, 3=Object>Scramble, 13=Face>Word, 19=Scramble_raw
CONTRASTS = {
    # category     → (cope, negate)
    'face_FFA':      (1,  False),
    'face_STS':      (1,  False),
    'house_PPA':     (2,  False),
    'house_TOS':     (2,  False),
    'object_pF':     (3,  False),
    'object_LOC':    (3,  False),
    'word_VWFA':     (13, True),   # −cope13 = Word>Face
    'word_STG':      (13, True),
    'word_pSTG_liu': (13, True),
    'word_IFG':      (13, True),
    'evc':           (19, False),
    'house_PPA_strict': (2, False),
}

# Control hemispheres: bilateral for ALL categories
CONTROL_HEMIS = {cat: ['l', 'r'] for cat in CONTRASTS}

# ── NIfTI cache ──────────────────────────────────────────────────────────────
_CACHE = {}
def _load(fp):
    k = str(fp)
    if k not in _CACHE:
        _CACHE[k] = nib.load(k)
    return _CACHE[k]

# ── Subject loader ───────────────────────────────────────────────────────────
def load_subjects():
    df = _load_csv()
    subjects = {}
    for sc in sorted(df['sub_clean'].unique()):
        sid = f'sub-{sc}'
        
        # 1. Logic handles standard skips and EXTRA_SKIP
        if should_skip(sid):
            continue
            
        alls = get_sessions(sc)
        if not alls or not (BASE_DIR / sid).exists():
            continue
            
        # 2. Anchor to FIRST chronological session
        first_ses = alls[0]
        info   = get_sub_info(sc, first_ses)
        pt     = is_patient(sc)
        intact = info.get('intact_hemi', '')
        code   = f"{info.get('group', '')}{sc}"
        
        subjects[sid] = {
            'code':           code,
            'sessions':       [f'{s:02d}' for s in alls],
            'anchor_ses':     f'{first_ses:02d}',
            'patient_hemi':   ('l' if intact == 'left' else 'r') if pt else None,
            'group':          info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi':    intact,
            'surgery_side':   ('right' if intact == 'left' else 'left') if pt else 'na',
        }
    return subjects

# ── File loaders ─────────────────────────────────────────────────────────────
def _load_searchmask(sid, first_ses, category, hemi):
    p = BASE_DIR / sid / f'ses-{first_ses}' / 'ROIs' / f'{hemi}_{category}_searchmask.nii.gz'
    if not p.exists():
        return None, None
    mi = _load(p)
    return mi.get_fdata() > 0, mi.affine

def _load_zstat(sid, session, first_ses, cope_num, negate=False):
    bm_file = BASE_DIR / sid / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    feat  = BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf    = feat / f'cope{cope_num}.feat' / 'stats' / zname
    if not zf.exists():
        return None

    z = _load(zf).get_fdata().copy()
    if negate:
        z = -z
    if bm is not None:
        z[~bm] = 0
    return z

# ── Analysis ─────────────────────────────────────────────────────────────────
def find_peak(sid, session, category, hemi, info):
    """Max zstat within searchmask (Liu 2025: no threshold, native space)."""
    first_ses   = info['anchor_ses']
    cope, neg   = CONTRASTS[category]
    mask, aff   = _load_searchmask(sid, first_ses, category, hemi)
    if mask is None:
        return None
    z = _load_zstat(sid, session, first_ses, cope, neg)
    if z is None:
        return None
    z_masked = np.where(mask, z, -np.inf)
    if not np.isfinite(z_masked).any():
        return None
    pidx = np.unravel_index(np.argmax(z_masked), z_masked.shape)
    return {
        'peak_z':     float(z[pidx]),
        'peak_coord': nib.affines.apply_affine(aff, np.array(pidx)),
    }

def compute_sum_selectivity(sid, session, category, hemi, info):
    """Ayzenberg sum-selectivity: Σ(z>thresh) / n_searchmask × 1000."""
    first_ses = info['anchor_ses']
    cope, neg = CONTRASTS[category]
    mask, _   = _load_searchmask(sid, first_ses, category, hemi)
    if mask is None:
        return {'sum_selec_norm': np.nan, 'mean_act': np.nan,
                'volume': 0, 'n_searchmask': 0}
    
    z = _load_zstat(sid, session, first_ses, cope, neg)
    n_searchmask = int(mask.sum())
    if z is None:
        return {'sum_selec_norm': np.nan, 'mean_act': np.nan,
                'volume': 0, 'n_searchmask': n_searchmask}
    
    supra  = mask & (z > SEL_Z_THRESH)
    volume = int(supra.sum())
    if volume == 0:
        return {'sum_selec_norm': 0.0 if n_searchmask > 0 else np.nan,
                'mean_act': np.nan, 'volume': 0, 'n_searchmask': n_searchmask}
    
    z_supra = z[supra]
    return {
        'sum_selec_norm': (float(z_supra.sum()) / n_searchmask) * SEL_RESCALE,
        'mean_act':       float(z_supra.mean()),
        'volume':         volume,
        'n_searchmask':   n_searchmask,
    }

# ── Pipeline ─────────────────────────────────────────────────────────────────
def process_session(sid, info, session):
    rows    = []
    is_ctrl = info['patient_status'] == 'control'

    for category in CONTRASTS:
        hemis = CONTROL_HEMIS[category] if is_ctrl else [info['patient_hemi']]
        for hemi in hemis:
            roi = find_peak(sid, session, category, hemi, info)
            if roi is None:
                continue
            sel = compute_sum_selectivity(sid, session, category, hemi, info)

            if is_ctrl:
                hemi_label = 'left' if hemi == 'l' else 'right'
            else:
                hemi_label = 'intact' if hemi == info['patient_hemi'] else 'lesioned'

            rows.append({
                'subject_id':      sid,
                'code':            info['code'],
                'session':         session,
                'group':           'control' if is_ctrl else info['group'],
                'status':          info['patient_status'],
                'surgery_side':    info['surgery_side'],
                'intact_hemi':     info['intact_hemi'],
                'hemi':            hemi,
                'hemi_label':      hemi_label,
                'category':        category,
                'peak_x_native':   roi['peak_coord'][0],
                'peak_y_native':   roi['peak_coord'][1],
                'peak_z_native':   roi['peak_coord'][2],
                'peak_z':          roi['peak_z'],
                'sum_selec_norm':  sel['sum_selec_norm'],
                'mean_act':        sel['mean_act'],        # Order explicitly restored
                'volume':          sel['volume'],          # Order explicitly restored
                'n_searchmask':    sel['n_searchmask'],
                'sel_threshold_z': SEL_Z_THRESH,
            })
    return rows

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / OUTPUT_NAME

    subs = load_subjects()
    n_pt   = sum(v['patient_status'] == 'patient' for v in subs.values())
    n_ctrl = sum(v['patient_status'] == 'control' for v in subs.values())
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Sum-selec: z>{SEL_Z_THRESH:.3f}; reporting sum_selec_norm, mean_act, volume')
    print(f'Categories: {list(CONTRASTS.keys())}\n')

    all_rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i + 1}/{len(subs)}] {info["code"]} ({time.time() - t0:.0f}s)', end='\r')
        for session in info['sessions']:
            all_rows.extend(process_session(sid, info, session))
        _CACHE.clear()

    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    print(f'\nSaved: {out} ({len(df)} rows, {df["subject_id"].nunique()} subjects)')

    print('\nROI × hemi coverage:')
    print(df.groupby(['category', 'hemi'])['subject_id'].nunique().unstack(fill_value=0))

if __name__ == '__main__':
    main()