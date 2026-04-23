#!/usr/bin/env python3
"""
liu_exact_replication_v2.py — Liu (2025) cross-sectional replication.

Pipeline:
  - Peak-finding: max zstat in searchmask, no threshold, 1mm space (fine for localization)
  - RSA sphere: 7mm sphere around peak, but betas downsampled to 2mm (native functional
    resolution) before correlation. FSL HighLevel copes are upsampled to 1mm anatomical
    space; correlating ~1400 interpolated voxels inflates r. Downsampling to 2mm gives
    ~175 independent voxels per sphere, matching Liu's BrainVoyager pipeline.
  - Sum-selectivity: Ayzenberg (2023) formula over full searchmask at 1mm (unaffected
    by resolution since it sums z-values, not correlations).
  - word_pSTG_liu: H-O posterior STG only (index 16), matching Liu's specification.
  - EVC: peak from raw scrambled cope.

Output: liu_exact_replication_v2.csv

Usage:
  python liu_exact_replication_v2.py
  python liu_exact_replication_v2.py --sub 021
  python liu_exact_replication_v2.py --include-pstg-liu
"""

import os, sys, time, argparse
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

SPHERE_RADIUS  = 7                         # Liu's 7mm sphere
FUNC_VOXEL_MM  = 2.0                       # Native functional resolution
ANAT_VOXEL_MM  = 1.0                       # Anatomical (cope) resolution
DOWNSAMPLE_FAC = ANAT_VOXEL_MM / FUNC_VOXEL_MM  # 0.5

SEL_Z_THRESH = float(norm.ppf(0.99))       # ≈2.326, p<.01 one-tailed
SEL_RESCALE  = 1000.0                      # Ayzenberg rescale factor

SUBJECTS_TO_SKIP = ['sub-017', 'control083', 'control085']
PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
    'sub-108': ['02'],
}

# GLM cope numbers
# 1=Face>Object, 2=House>Object, 3=Object>Scramble, 13=Face>Word
LOC_CONTRASTS = {
    'face_FFA':      {'cope': 1,  'negate': False, 'exact': False, 'liu': 'face>house',       'used': 'face>object'},
    'face_STS':      {'cope': 1,  'negate': False, 'exact': False, 'liu': 'face>house',       'used': 'face>object'},
    'house_PPA':     {'cope': 2,  'negate': False, 'exact': False, 'liu': 'house>face',       'used': 'house>object'},
    'house_TOS':     {'cope': 2,  'negate': False, 'exact': False, 'liu': 'house>face',       'used': 'house>object'},
    'object_pF':     {'cope': 3,  'negate': False, 'exact': True,  'liu': 'object>scramble',  'used': 'object>scramble'},
    'object_LOC':    {'cope': 3,  'negate': False, 'exact': True,  'liu': 'object>scramble',  'used': 'object>scramble'},
    'word_VWFA':     {'cope': 13, 'negate': True,  'exact': True,  'liu': 'word>face',        'used': 'word>face (−cope13)'},
    'word_STG':      {'cope': 13, 'negate': True,  'exact': True,  'liu': 'word>face',        'used': 'word>face (−cope13)'},
    'word_pSTG_liu': {'cope': 13, 'negate': True,  'exact': True,  'liu': 'word>face (pSTG)', 'used': 'word>face (−cope13)'},
    # EVC: raw scrambled — VERIFY cope 19 against design.con
    'evc':           {'cope': 19, 'negate': False, 'exact': False, 'liu': 'EVC (scrambled)',   'used': 'scrambled (raw)'},
}

RSA_COPES      = {'face': 15, 'house': 16, 'object': 17, 'word': 18}
RSA_CATEGORIES = ['face', 'house', 'object', 'word']

CONTROL_HEMIS = {
    'face_FFA':      ['l', 'r'],
    'face_STS':      ['l', 'r'],
    'house_PPA':     ['l', 'r'],
    'house_TOS':     ['l', 'r'],
    'object_pF':     ['l', 'r'],
    'object_LOC':    ['l', 'r'],
    'evc':           ['l', 'r'],
    'word_VWFA':     ['l'],
    'word_STG':      ['l'],
    'word_pSTG_liu': ['l'],
}

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

# ── Searchmask loader ────────────────────────────────────────────────────────
def _load_searchmask(subject_id, first_ses, category, hemi):
    """Return (mask_bool, affine) or (None, None)."""
    for sd in ['ROIs', os.path.join('derivatives', 'rois')]:
        p = BASE_DIR / subject_id / f'ses-{first_ses}' / sd / f'{hemi}_{category}_searchmask.nii.gz'
        if p.exists():
            mi = _load(p)
            return mi.get_fdata() > 0, mi.affine
    return None, None

def _load_zstat(subject_id, session, first_ses, cope_num, negate=False):
    """Return (zstat_array, brain_mask) or (None, None)."""
    bm_file = BASE_DIR / subject_id / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    feat = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf = feat / f'cope{cope_num}.feat' / 'stats' / zname
    if not zf.exists(): return None, None

    z = _load(zf).get_fdata().copy()
    if negate: z = -z
    if bm is not None: z[~bm] = 0
    return z, bm

# ── Peak-finding (1mm — fine for localization) ───────────────────────────────
def find_peak_voxel(subject_id, session, category, hemi, subs):
    """Max zstat within searchmask, no threshold (Liu 2025)."""
    info = subs[subject_id]
    first_ses = info['sessions'][0]
    ci = LOC_CONTRASTS[category]

    mask, affine = _load_searchmask(subject_id, first_ses, category, hemi)
    if mask is None: return None

    z, _ = _load_zstat(subject_id, session, first_ses, ci['cope'], ci['negate'])
    if z is None: return None

    z_masked = np.where(mask, z, -np.inf)
    if not np.isfinite(z_masked).any(): return None
    pidx = np.unravel_index(np.argmax(z_masked), z_masked.shape)

    return {
        'peak_ijk':    pidx,
        'peak_z':      float(z[pidx]),
        'peak_coord':  nib.affines.apply_affine(affine, np.array(pidx)),
        'affine':      affine,
        'brain_shape': z.shape,
    }

# ── Sphere (built at 1mm) ───────────────────────────────────────────────────
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

# ── Beta extraction at 2mm functional resolution ─────────────────────────────
def extract_betas(subject_id, session, sphere_1mm, subs):
    """Extract RSA betas within sphere, downsampled to 2mm functional resolution.

    FSL HighLevel copes are in 1mm anatomical space (upsampled from 2mm functional).
    Correlating ~1400 interpolated voxels inflates inter-category r. Downsampling
    both the cope volume and sphere mask to 2mm before extraction gives ~175
    independent voxels, matching Liu's BrainVoyager pipeline.
    """
    info = subs[subject_id]
    first_ses = info['sessions'][0]
    feat = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'

    # Downsample sphere mask once (nearest-neighbor to preserve binary mask)
    sphere_2mm = zoom(sphere_1mm.astype(float), DOWNSAMPLE_FAC, order=0) > 0.5

    patterns, valid = [], []
    for cat in RSA_CATEGORIES:
        cf = feat / f'cope{RSA_COPES[cat]}.feat' / 'stats' / cn
        if not cf.exists(): continue
        vol_1mm = _load(cf).get_fdata()
        # Downsample cope to 2mm (trilinear interpolation)
        vol_2mm = zoom(vol_1mm, DOWNSAMPLE_FAC, order=1)
        b = vol_2mm[sphere_2mm]
        b = b[np.isfinite(b)]
        if len(b): patterns.append(b); valid.append(cat)

    if len(patterns) < 4: return None, None
    m = min(len(b) for b in patterns)
    return np.column_stack([b[:m] for b in patterns]), valid

# ── Distinctiveness ──────────────────────────────────────────────────────────
def compute_distinctiveness(beta_mat, valid, roi_cat):
    parent = roi_cat.split('_')[0] if '_' in roi_cat else roi_cat
    if roi_cat == 'word_pSTG_liu': parent = 'word'
    if parent not in valid: return np.nan, {}
    corr = np.corrcoef(beta_mat.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))
    pidx = valid.index(parent)
    others = [i for i in range(len(valid)) if i != pidx]
    dist = float(np.mean([fisher[pidx, i] for i in others]))
    pairs = {}
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            pairs[f'{valid[i]}-{valid[j]}'] = float(fisher[i, j])
    return dist, pairs

# ── Sum-selectivity (1mm — unaffected by resolution) ─────────────────────────
def compute_sum_selectivity(subject_id, session, category, hemi, subs,
                            z_thresh=SEL_Z_THRESH):
    """Ayzenberg (2023) sum-selectivity over searchmask."""
    info = subs[subject_id]
    first_ses = info['sessions'][0]
    ci = LOC_CONTRASTS[category]

    mask, _ = _load_searchmask(subject_id, first_ses, category, hemi)
    if mask is None:
        return {'sum_selec_norm': np.nan, 'mean_act': np.nan,
                'volume': 0, 'n_searchmask': 0}

    z, _ = _load_zstat(subject_id, session, first_ses, ci['cope'], ci['negate'])
    if z is None:
        return {'sum_selec_norm': np.nan, 'mean_act': np.nan,
                'volume': 0, 'n_searchmask': int(mask.sum())}

    n_searchmask = int(mask.sum())
    supra = mask & (z > z_thresh)
    volume = int(supra.sum())

    if volume == 0 or n_searchmask == 0:
        return {'sum_selec_norm': 0.0 if n_searchmask > 0 else np.nan,
                'mean_act': np.nan, 'volume': volume, 'n_searchmask': n_searchmask}

    z_supra = z[supra]
    return {
        'sum_selec_norm': (float(z_supra.sum()) / n_searchmask) * SEL_RESCALE,
        'mean_act':       float(z_supra.mean()),
        'volume':         volume,
        'n_searchmask':   n_searchmask,
    }

# ── Pipeline ─────────────────────────────────────────────────────────────────
def process_subject_session(sid, info, session, categories):
    rows = []
    is_ctrl = info['patient_status'] == 'control'
    for category in categories:
        if category not in LOC_CONTRASTS: continue
        ci = LOC_CONTRASTS[category]
        hemis = CONTROL_HEMIS[category] if is_ctrl else [info['patient_hemi']]
        for hemi in hemis:
            roi = find_peak_voxel(sid, session, category, hemi, {sid: info})
            if roi is None: continue
            sphere_1mm = create_sphere(roi['peak_coord'], roi['affine'], roi['brain_shape'])
            betas, valid = extract_betas(sid, session, sphere_1mm, {sid: info})
            if betas is None: continue
            dist, pairs = compute_distinctiveness(betas, valid, category)

            sel = compute_sum_selectivity(sid, session, category, hemi, {sid: info})

            hl = ('intact' if (hemi == info['patient_hemi']) else 'lesioned') if not is_ctrl else hemi

            # Report sphere voxels at 2mm for transparency
            n_sphere_2mm = int((zoom(sphere_1mm.astype(float), DOWNSAMPLE_FAC, order=0) > 0.5).sum())

            base = {
                'subject_id':          sid,
                'code':                info['code'],
                'session':             session,
                'group':               info['group'] if not is_ctrl else 'control',
                'status':              info['patient_status'],
                'surgery_side':        info['surgery_side'],
                'intact_hemi':         info['intact_hemi'],
                'hemi':                hemi,
                'hemi_label':          'left' if hemi == 'l' else 'right' if is_ctrl else hl,
                'category':            category,
                'contrast_liu':        ci['liu'],
                'contrast_used':       ci['used'],
                'contrast_exact':      ci['exact'],
                'peak_x_mni':          roi['peak_coord'][0],
                'peak_y_mni':          roi['peak_coord'][1],
                'peak_z_mni':          roi['peak_coord'][2],
                'peak_z':              roi['peak_z'],
                'n_sphere_voxels':     n_sphere_2mm,
                'liu_distinctiveness': dist,
                'sum_selec_norm':      sel['sum_selec_norm'],
                'mean_act':            sel['mean_act'],
                'volume':              sel['volume'],
                'n_searchmask':        sel['n_searchmask'],
                'sel_threshold_z':     SEL_Z_THRESH,
            }
            for pair, fz in pairs.items():
                rows.append({**base, 'pair': pair, 'fisher_r': fz})
    return rows

def check_pstg_masks_exist(subjects):
    for sid, info in list(subjects.items())[:5]:
        first_ses = info['sessions'][0]
        m, _ = _load_searchmask(sid, first_ses, 'word_pSTG_liu', 'l')
        if m is not None:
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, help='Single subject (e.g., 021)')
    parser.add_argument('--include-pstg-liu', action='store_true',
                        help='Include word_pSTG_liu ROI (requires masks to exist)')
    parser.add_argument('--skip-check', action='store_true',
                        help='Skip pSTG mask existence check')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subs = load_subjects()
    if args.sub:
        sid = f'sub-{args.sub.replace("sub-","")}'
        subs = {sid: subs[sid]} if sid in subs else {}

    categories = [c for c in LOC_CONTRASTS.keys() if c != 'word_pSTG_liu']
    if args.include_pstg_liu:
        if not args.skip_check and not check_pstg_masks_exist(subs):
            print('ERROR: --include-pstg-liu requested but no pSTG_liu masks found.')
            print('       Run: python add_pstg_liu_searchmasks.py first.')
            sys.exit(1)
        categories.append('word_pSTG_liu')

    n_pt = sum(1 for v in subs.values() if v['patient_status']=='patient')
    n_ctrl = sum(1 for v in subs.values() if v['patient_status']=='control')
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Sphere: {SPHERE_RADIUS}mm; RSA betas downsampled to {FUNC_VOXEL_MM}mm')
    print(f'Sum-selec: z>{SEL_Z_THRESH:.3f} (p<.01 1-tail) over searchmask at 1mm')
    print(f'Categories: {categories}')
    print()

    all_rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items())):
        print(f'[{i+1}/{len(subs)}] {info["code"]} ({time.time()-t0:.0f}s)', end='\r')
        for session in info['sessions']:
            if sid in PRE_SURGERY_SESSIONS and session in PRE_SURGERY_SESSIONS[sid]:
                continue
            all_rows.extend(process_subject_session(sid, info, session, categories))

    df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / OUTPUT_NAME
    df.to_csv(out, index=False)
    print(f'\nSaved: {out} ({len(df)} rows, {df["subject_id"].nunique()} subjects)')

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print(f'\nROI × hemi coverage:')
    print(df.groupby(['category','hemi'])['subject_id'].nunique().unstack(fill_value=0))

    # Sphere size check (should be ~175 at 2mm)
    summary_df = df.drop(columns=['pair','fisher_r']).drop_duplicates()
    print(f'\nSphere voxel counts (2mm):')
    print(summary_df.groupby('category')['n_sphere_voxels'].agg(['mean','std']).round(1))

    # Sum-selectivity sanity check
    print(f'\nSum-selectivity (first session, controls):')
    summary_df['ses_int'] = pd.to_numeric(summary_df['session'], errors='coerce')
    first_ses = summary_df[summary_df['status']=='control'].groupby('subject_id')['ses_int'].min()
    fs_df = summary_df.merge(first_ses.rename('fs'), on='subject_id')
    fs_df = fs_df[fs_df['ses_int']==fs_df['fs']]
    agg = (fs_df.groupby(['category','hemi'])['sum_selec_norm']
                 .agg(['mean','std','count']).round(2))
    print(agg)

    # STG comparison
    if 'word_pSTG_liu' in df['category'].values:
        print(f'\nSTG comparison (controls, LH):')
        for cat in ['word_STG', 'word_pSTG_liu']:
            sub_df = fs_df[(fs_df['category']==cat) & (fs_df['hemi']=='l')]
            if len(sub_df):
                print(f'  {cat:20s}: n_searchmask M={sub_df["n_searchmask"].mean():.0f}, '
                      f'sum_selec_norm M={sub_df["sum_selec_norm"].mean():.2f}')

if __name__ == '__main__':
    main()