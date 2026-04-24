#!/usr/bin/env python3
"""
liu_recreation_csv.py — Liu (2025) cross-sectional + longitudinal extraction.

Pipeline:
  - Peak-finding: max zstat within searchmask, 1mm anat space, no threshold
  - RSA sphere: 7mm at 1mm anat → downsampled to 2mm (native functional
    resolution) for correlation. ~175 independent voxels per sphere.
  - Sum-selectivity (Ayzenberg 2023): Σ(z) over z>2.326 voxels in searchmask,
    normalized by searchmask size, rescaled ×1000. At 1mm (resolution-invariant).

Contrast notes (divergences from Liu, documented for methods):
  - face/house: use >object rather than Liu's >house/>face
  - word: word>face via −cope13 (cope 13 is Face>Word)
  - EVC: cope 19 (Scramble_raw) — Liu methods don't specify EVC contrast

Output: liu_exact_replication_v2.csv (native-space coords)

Usage:
  python liu_recreation_csv.py                 # all subjects
  python liu_recreation_csv.py --sub 021       # single subject
"""
import os
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.ndimage import zoom

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = Path(processed_dir)
OUTPUT_DIR  = Path('/user_data/csimmon2/git_repos/sym_pt')
OUTPUT_NAME = 'liu_exact_replication_v2.csv'

SPHERE_RADIUS  = 7                              # mm
FUNC_VOXEL_MM  = 2.0                            # native functional resolution
ANAT_VOXEL_MM  = 1.0
DOWNSAMPLE_FAC = ANAT_VOXEL_MM / FUNC_VOXEL_MM  # 0.5

SEL_Z_THRESH = float(norm.ppf(0.99))            # ≈2.326, p<.01 one-tailed
SEL_RESCALE  = 1000.0

EXTRA_SKIP = {'sub-017', 'control083', 'control085'}  # beyond sym_pt_params.skip_subs
PRE_SURGERY_SESSIONS = {
    'sub-021': {'01'}, 'sub-045': {'01'}, 'sub-047': {'01'}, 'sub-049': {'01'},
    'sub-070': {'01'}, 'sub-073': {'01'}, 'sub-081': {'01'}, 'sub-086': {'01'},
    'sub-108': {'02'},
}

# GLM cope numbers (from FEAT design.con):
# 1=Face>Object, 2=House>Object, 3=Object>Scramble, 13=Face>Word, 19=Scramble_raw
# Raw betas: 15=Face, 16=House, 17=Object, 18=Word
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
    'evc':           (19, False),
}
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

# Control hemispheres: words are LH-preferred in controls; others bilateral
# Controls: bilateral for ALL categories
CONTROL_HEMIS = {cat: ['l', 'r'] for cat in CONTRASTS}
#CONTROL_HEMIS = {cat: (['l'] if cat.startswith('word') else ['l', 'r'])
#                 for cat in CONTRASTS}

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
        if sc in skip_subs:
            continue
        sid = f'sub-{sc}'
        sessions = get_sessions(sc)
        if not sessions or not (BASE_DIR / sid).exists():
            continue
        info   = get_sub_info(sc, sessions[0])
        pt     = is_patient(sc)
        intact = info.get('intact_hemi', '')
        code   = f"{info.get('group', '')}{sc}"
        if code in EXTRA_SKIP or sid in EXTRA_SKIP:
            continue
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
    first_ses   = info['sessions'][0]
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
        'peak_ijk':    pidx,
        'peak_z':      float(z[pidx]),
        'peak_coord':  nib.affines.apply_affine(aff, np.array(pidx)),
        'affine':      aff,
        'brain_shape': z.shape,
    }

def create_sphere(peak_coord, affine, brain_shape, radius=SPHERE_RADIUS):
    """Binary sphere at 1mm anat resolution."""
    grid  = np.array(np.meshgrid(
        np.arange(brain_shape[0]), np.arange(brain_shape[1]),
        np.arange(brain_shape[2]), indexing='ij'
    )).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    mask  = np.zeros(brain_shape, dtype=bool)
    for c in grid[dists <= radius]:
        mask[c[0], c[1], c[2]] = True
    return mask

def extract_betas(sid, session, sphere_1mm, info):
    """RSA beta extraction, downsampled to 2mm functional resolution."""
    first_ses = info['sessions'][0]
    feat      = BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn        = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'
    sphere_2mm = zoom(sphere_1mm.astype(float), DOWNSAMPLE_FAC, order=0) > 0.5

    patterns, valid = [], []
    for cat, cope in RSA_COPES.items():
        cf = feat / f'cope{cope}.feat' / 'stats' / cn
        if not cf.exists():
            continue
        vol_2mm = zoom(_load(cf).get_fdata(), DOWNSAMPLE_FAC, order=1)
        b = vol_2mm[sphere_2mm]
        b = b[np.isfinite(b)]
        if len(b):
            patterns.append(b)
            valid.append(cat)
    if len(patterns) < 4:
        return None, None
    m = min(len(b) for b in patterns)
    return np.column_stack([b[:m] for b in patterns]), valid

def compute_distinctiveness(beta_mat, valid, roi_cat):
    """Mean Fisher-z correlation between preferred and non-preferred categories."""
    parent = 'word' if roi_cat == 'word_pSTG_liu' else roi_cat.split('_')[0]
    if parent not in valid:
        return np.nan, {}
    corr   = np.corrcoef(beta_mat.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))
    pidx   = valid.index(parent)
    others = [i for i in range(len(valid)) if i != pidx]
    dist   = float(np.mean([fisher[pidx, i] for i in others]))
    pairs  = {f'{valid[i]}-{valid[j]}': float(fisher[i, j])
              for i in range(len(valid)) for j in range(i + 1, len(valid))}
    return dist, pairs

def compute_sum_selectivity(sid, session, category, hemi, info):
    """Ayzenberg sum-selectivity: Σ(z>thresh) / n_searchmask × 1000."""
    first_ses = info['sessions'][0]
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
            sphere_1mm   = create_sphere(roi['peak_coord'], roi['affine'], roi['brain_shape'])
            betas, valid = extract_betas(sid, session, sphere_1mm, info)
            if betas is None:
                continue
            dist, pairs  = compute_distinctiveness(betas, valid, category)
            sel          = compute_sum_selectivity(sid, session, category, hemi, info)
            n_sphere_2mm = int((zoom(sphere_1mm.astype(float), DOWNSAMPLE_FAC, order=0) > 0.5).sum())

            if is_ctrl:
                hemi_label = 'left' if hemi == 'l' else 'right'
            else:
                hemi_label = 'intact' if hemi == info['patient_hemi'] else 'lesioned'

            base = {
                'subject_id':          sid,
                'code':                info['code'],
                'session':             session,
                'group':               'control' if is_ctrl else info['group'],
                'status':              info['patient_status'],
                'surgery_side':        info['surgery_side'],
                'intact_hemi':         info['intact_hemi'],
                'hemi':                hemi,
                'hemi_label':          hemi_label,
                'category':            category,
                'peak_x_native':       roi['peak_coord'][0],
                'peak_y_native':       roi['peak_coord'][1],
                'peak_z_native':       roi['peak_coord'][2],
                'peak_z':              roi['peak_z'],
                'n_sphere_voxels':     n_sphere_2mm,
                'liu_distinctiveness': dist,
                'sum_selec_norm':      sel['sum_selec_norm'],
                'mean_act':            sel['mean_act'],
                'volume':              sel['volume'],
                'n_searchmask':        sel['n_searchmask'],
                'sel_threshold_z':     SEL_Z_THRESH,
            }
            if pairs:
                for pair, fz in pairs.items():
                    rows.append({**base, 'pair': pair, 'fisher_r': fz})
            else:
                # ROIs with no RSA-preferred category (e.g., EVC):
                # emit a single row so peak/sum-selec aren't lost.
                rows.append({**base, 'pair': None, 'fisher_r': np.nan})
    return rows

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, help='Single subject (e.g., 021)')
    parser.add_argument('--category', type=str,
                        help='Run single category (e.g., evc). Merges into existing CSV.')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / OUTPUT_NAME

    # Restrict categories if --category flag used
    global CONTRASTS, CONTROL_HEMIS
    if args.category:
        if args.category not in CONTRASTS:
            print(f'ERROR: unknown category {args.category!r}. Valid: {list(CONTRASTS)}')
            sys.exit(1)
        CONTRASTS     = {args.category: CONTRASTS[args.category]}
        CONTROL_HEMIS = {args.category: CONTROL_HEMIS[args.category]}

    subs = load_subjects()
    if args.sub:
        sid  = f'sub-{args.sub.replace("sub-", "")}'
        subs = {sid: subs[sid]} if sid in subs else {}

    n_pt   = sum(v['patient_status'] == 'patient' for v in subs.values())
    n_ctrl = sum(v['patient_status'] == 'control' for v in subs.values())
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Sphere: {SPHERE_RADIUS}mm @ 1mm anat → 2mm RSA; sum-selec z>{SEL_Z_THRESH:.3f}')
    print(f'Categories: {list(CONTRASTS.keys())}\n')

    all_rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i + 1}/{len(subs)}] {info["code"]} ({time.time() - t0:.0f}s)', end='\r')
        for session in info['sessions']:
            if session in PRE_SURGERY_SESSIONS.get(sid, set()):
                continue
            all_rows.extend(process_session(sid, info, session))

    df_new = pd.DataFrame(all_rows)

    # Merge mode: if --category used and CSV exists, replace that category's rows
    if args.category and out.exists():
        df_old = pd.read_csv(out)
        n_before = len(df_old)
        df_old = df_old[df_old['category'] != args.category]
        df = pd.concat([df_old, df_new], ignore_index=True)
        print(f'\nMerge: dropped {n_before - len(df_old)} old {args.category} rows, '
              f'added {len(df_new)} new rows')
    else:
        df = df_new

    df.to_csv(out, index=False)
    print(f'\nSaved: {out} ({len(df)} rows, {df["subject_id"].nunique()} subjects)')

    # Diagnostics
    print('\nROI × hemi coverage:')
    print(df.groupby(['category', 'hemi'])['subject_id'].nunique().unstack(fill_value=0))

    summary = df.drop(columns=['pair', 'fisher_r']).drop_duplicates()
    print('\nSphere voxel counts (2mm, expect ~175):')
    print(summary.groupby('category')['n_sphere_voxels'].agg(['mean', 'std']).round(1))

if __name__ == '__main__':
    main()