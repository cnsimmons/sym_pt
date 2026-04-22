#!/usr/bin/env python3
"""
liu_exact_replication.py — standalone Liu (2025) cross-sectional replication.

Differences from 05/07:
  - 7mm sphere (Liu) instead of 6mm
  - NO z-threshold (peak = max voxel in searchmask, period)
  - NO top-10%/cluster step — just the single peak voxel → sphere
  - Liu's pairwise contrasts where available in your GLM:
      FFA, STS    → face > house     (not in your design — fallback: cope 1, face>object)
      PPA, TOS    → house > face     (not in your design — fallback: cope 2, house>object)
      pF, LOC     → object > scramble (cope 3)  ✓ exact match
      VWFA, STG   → word > face       (cope 13 negated)  ✓ exact match
      (FFA/STS/PPA/TOS fallback flagged in output column `contrast_exact`)

Output: ONE csv combining peak coords + distinctiveness + pairwise correlations.
  /user_data/csimmon2/sym_pt/group_results/liu_exact/liu_exact_replication.csv

Usage:
  python liu_exact_replication.py
  python liu_exact_replication.py --sub 021    # single subject for testing
"""

import os, sys, time, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR   = Path(processed_dir)
OUTPUT_DIR = Path('/user_data/csimmon2/git_repos/sym_pt')


SPHERE_RADIUS = 7   # Liu specifies 7mm

SUBJECTS_TO_SKIP = ['sub-017','control083', 'control085']
PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
    'sub-108': ['02'],  # skip ses-02 if ses-01 is the one to keep
}

# Liu's paired contrasts → your GLM cope numbers
# Your design.con:
#   1=Face>Object, 2=House>Object, 3=Object>Scramble, 13=Face>Word
#   No face>house, no house>face in your design
LOC_CONTRASTS = {
    'face_FFA':   {'cope': 1,  'negate': False, 'exact': False, 'liu': 'face>house', 'used': 'face>object'},
    'face_STS':   {'cope': 1,  'negate': False, 'exact': False, 'liu': 'face>house', 'used': 'face>object'},
    'house_PPA':  {'cope': 2,  'negate': False, 'exact': False, 'liu': 'house>face', 'used': 'house>object'},
    'house_TOS':  {'cope': 2,  'negate': False, 'exact': False, 'liu': 'house>face', 'used': 'house>object'},
    'object_pF':  {'cope': 3,  'negate': False, 'exact': True,  'liu': 'object>scramble', 'used': 'object>scramble'},
    'object_LOC': {'cope': 3,  'negate': False, 'exact': True,  'liu': 'object>scramble', 'used': 'object>scramble'},
    'word_VWFA':  {'cope': 13, 'negate': True,  'exact': True,  'liu': 'word>face', 'used': 'word>face (−cope13)'},
    'word_STG':   {'cope': 13, 'negate': True,  'exact': True,  'liu': 'word>face', 'used': 'word>face (−cope13)'},
    'evc': {'cope': 3, 'negate': False, 'exact': False, 'liu': 'EVC (anatomical)', 'used': 'object>scramble'},
}

# RSA betas (raw condition betas — same for all)
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}
RSA_CATEGORIES = ['face', 'house', 'object', 'word']

BILATERAL = {'object_pF', 'object_LOC', 'house_PPA', 'house_TOS'}
# word_VWFA, word_STG, face_FFA, face_STS → processed at:
#   controls: LH only (Liu convention for word; + standard for face)
#   Actually Liu processes face bilat in controls. Keep face bilat, word LH only for ctrls.
CONTROL_HEMIS = {
    'face_FFA':   ['l', 'r'],
    'face_STS':   ['l', 'r'],
    'house_PPA':  ['l', 'r'],
    'house_TOS':  ['l', 'r'],
    'object_pF':  ['l', 'r'],
    'object_LOC': ['l', 'r'],
    'evc': ['l', 'r'],
    'word_VWFA':  ['l'],      # typically LH
    'word_STG':   ['l'],      # typically LH
}

# ── NIfTI cache ──────────────────────────────────────────────────────────────
_CACHE = {}
def _load(fp):
    k = str(fp)
    if k not in _CACHE:
        _CACHE[k] = nib.load(k)
    return _CACHE[k]

# ── Load subjects ────────────────────────────────────────────────────────────
def load_subjects():
    df = _load_csv()
    subjects = {}
    for sub_clean in sorted(df['sub_clean'].unique()):
        if sub_clean in skip_subs: continue
        sid = f'sub-{sub_clean}'
        sessions = get_sessions(sub_clean)
        if not sessions or not (BASE_DIR / sid).exists(): continue
        info = get_sub_info(sub_clean, sessions[0])
        pt = is_patient(sub_clean)
        intact = info.get('intact_hemi', '')
        code = f"{info.get('group','')}{sub_clean}"
        if code in SUBJECTS_TO_SKIP: continue
        subjects[sid] = {
            'code':           code,
            'sessions':       [f'{s:02d}' for s in sessions],
            'patient_hemi':   ('l' if intact == 'left' else 'r') if pt else None,
            'group':          info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi':    intact,
            'surgery_side':   ('right' if intact == 'left' else 'left') if pt else 'na',
        }
    return subjects

# ── Core: find peak, build sphere, extract betas ────────────────────────────
def find_peak_voxel(subject_id, session, category, hemi, subs):
    """Return peak voxel ijk + affine + brain_shape, using Liu's NO-threshold rule."""
    info = subs[subject_id]
    first_ses = info['sessions'][0]

    # Searchmask
    mf = None
    for sd in ['ROIs', os.path.join('derivatives', 'rois')]:
        p = BASE_DIR / subject_id / f'ses-{first_ses}' / sd / f'{hemi}_{category}_searchmask.nii.gz'
        if p.exists():
            mf = p; break
    if mf is None: return None

    mi = _load(mf)
    mask = mi.get_fdata() > 0
    affine = mi.affine

    # Brain mask
    bm_file = BASE_DIR / subject_id / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    # Zstat
    ci = LOC_CONTRASTS[category]
    feat = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf = feat / f'cope{ci["cope"]}.feat' / 'stats' / zname
    if not zf.exists(): return None

    z = _load(zf).get_fdata().copy()
    if ci['negate']: z = -z
    if bm is not None: z[~bm] = 0

    # Mask to searchmask, take absolute max — NO threshold
    z_masked = np.where(mask, z, -np.inf)
    if not np.isfinite(z_masked).any(): return None
    pidx = np.unravel_index(np.argmax(z_masked), z_masked.shape)
    peak_val = float(z[pidx])

    return {
        'peak_ijk':    pidx,
        'peak_z':      peak_val,
        'peak_coord':  nib.affines.apply_affine(affine, np.array(pidx)),
        'affine':      affine,
        'brain_shape': z.shape,
    }

def create_sphere(peak_coord, affine, brain_shape, radius=SPHERE_RADIUS):
    grid = np.array(np.meshgrid(
        np.arange(brain_shape[0]), np.arange(brain_shape[1]),
        np.arange(brain_shape[2]), indexing='ij'
    )).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    mask = np.zeros(brain_shape, dtype=bool)
    for c in grid[dists <= radius]:
        mask[c[0], c[1], c[2]] = True
    return mask

def extract_betas(subject_id, session, sphere, subs):
    info = subs[subject_id]
    first_ses = info['sessions'][0]
    feat = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'
    patterns, valid = [], []
    for cat in RSA_CATEGORIES:
        cf = feat / f'cope{RSA_COPES[cat]}.feat' / 'stats' / cn
        if not cf.exists(): continue
        b = _load(cf).get_fdata()[sphere]
        b = b[np.isfinite(b)]
        if len(b): patterns.append(b); valid.append(cat)
    if len(patterns) < 4: return None, None
    m = min(len(b) for b in patterns)
    return np.column_stack([b[:m] for b in patterns]), valid

def compute_distinctiveness(beta_mat, valid, roi_cat):
    """Mean Fisher-z between preferred and non-preferred categories."""
    # Extract parent category from roi_cat
    parent = roi_cat.split('_')[0] if '_' in roi_cat else roi_cat
    if parent not in valid: return np.nan, {}
    corr = np.corrcoef(beta_mat.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))
    pidx = valid.index(parent)
    others = [i for i in range(len(valid)) if i != pidx]
    dist = float(np.mean([fisher[pidx, i] for i in others]))

    # Pairwise (all 6 pairs)
    pairs = {}
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            pairs[f'{valid[i]}-{valid[j]}'] = float(fisher[i, j])
    return dist, pairs

# ── Pipeline ─────────────────────────────────────────────────────────────────
def process_subject_session(sid, info, session):
    rows = []
    is_ctrl = info['patient_status'] == 'control'
    for category, ci in LOC_CONTRASTS.items():
        hemis = CONTROL_HEMIS[category] if is_ctrl else [info['patient_hemi']]
        for hemi in hemis:
            roi = find_peak_voxel(sid, session, category, hemi, {sid: info})
            if roi is None: continue
            sphere = create_sphere(roi['peak_coord'], roi['affine'], roi['brain_shape'])
            betas, valid = extract_betas(sid, session, sphere, {sid: info})
            if betas is None: continue
            dist, pairs = compute_distinctiveness(betas, valid, category)

            hl = ('intact' if (hemi == info['patient_hemi']) else 'lesioned') if not is_ctrl else hemi
            base = {
                'subject_id':   sid,
                'code':         info['code'],
                'session':      session,
                'group':        info['group'] if not is_ctrl else 'control',
                'status':       info['patient_status'],
                'surgery_side': info['surgery_side'],
                'intact_hemi':  info['intact_hemi'],
                'hemi':         hemi,
                'hemi_label':   'left' if hemi == 'l' else 'right' if is_ctrl else hl,
                'category':     category,
                'contrast_liu': ci['liu'],
                'contrast_used':ci['used'],
                'contrast_exact': ci['exact'],
                'peak_x_mni':   roi['peak_coord'][0],
                'peak_y_mni':   roi['peak_coord'][1],
                'peak_z_mni':   roi['peak_coord'][2],
                'peak_z':       roi['peak_z'],
                'n_sphere_voxels': int(sphere.sum()),
                'liu_distinctiveness': dist,
            }
            # One row per pair, plus the summary
            for pair, fz in pairs.items():
                rows.append({**base, 'pair': pair, 'fisher_r': fz})
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, help='Single subject (e.g., 021) for testing')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subs = load_subjects()
    if args.sub:
        sid = f'sub-{args.sub.replace("sub-","")}'
        subs = {sid: subs[sid]} if sid in subs else {}

    n_pt = sum(1 for v in subs.values() if v['patient_status']=='patient')
    n_ctrl = sum(1 for v in subs.values() if v['patient_status']=='control')
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Sphere: {SPHERE_RADIUS}mm, NO z-threshold, peak = max voxel in searchmask')

    all_rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i+1}/{len(subs)}] {info["code"]} ({time.time()-t0:.0f}s)', end='\r')
        for session in info['sessions']:
            if sid in PRE_SURGERY_SESSIONS and session in PRE_SURGERY_SESSIONS[sid]:
                continue
            all_rows.extend(process_subject_session(sid, info, session))

    df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / 'liu_exact_replication.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved: {out} ({len(df)} rows, {df["subject_id"].nunique()} subjects)')

    # Summary
    print(f'\nROIs × hemi coverage:')
    print(df.groupby(['category','hemi'])['subject_id'].nunique().unstack(fill_value=0))
    print(f'\nContrasts used:')
    print(df[['category','contrast_liu','contrast_used','contrast_exact']].drop_duplicates().to_string(index=False))

if __name__ == '__main__':
    main()
