#!/usr/bin/env python3
"""
3_multivariate_analyses.py — Liu (2025) RSA extraction (cross-sectional + longitudinal).

Per subject x ROI x hemisphere x session, builds the 4-category representational
geometry from beta patterns in a peak-centered sphere:
  - liu_distinctiveness : mean Fisher-z correlation, preferred category vs the
                          other three (higher = LESS distinct)
  - pair / fisher_r     : the 6 pairwise between-category Fisher-z correlations
                          (one row per pair; the 4x4 RDM off-diagonal)

Sphere: 7mm at 1mm anat, peak-centered, downsampled to 2mm native functional
resolution before correlation (~175 independent voxels). All native space.

No statistics, no figures, no session/cohort filtering — the stats script
filters (patient last session, control first session, exclusions). Univariate
measures (peak-z, sum-selectivity, volume) live in the univariate script.

ROIs: face_FFA, house_PPA, object_LOC, word_VWFA.
RSA betas: copes 15=Face, 16=House, 17=Object, 18=Word.

Output: D_liu/rsa_v1.csv (native-space coords)

Usage:
  python 3_multivariate_analyses.py
"""
import sys
import time
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from params import (processed_dir, is_patient, should_skip,
                    get_sessions, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = Path(processed_dir)
OUTPUT_DIR  = Path('/user_data/csimmon2/git_repos/sym_pt/D_liu')
OUTPUT_NAME = 'rsa_v1.csv'

SPHERE_RADIUS  = 7                              # mm
FUNC_VOXEL_MM  = 2.0                            # native functional resolution
ANAT_VOXEL_MM  = 1.0
DOWNSAMPLE_FAC = ANAT_VOXEL_MM / FUNC_VOXEL_MM  # 0.5

# Peak-finding contrast per ROI (from FEAT design.con):
#   1=Face>Object, 2=House>Object, 3=Object>Scramble, 13=Face>Word, 19=Scramble_raw
# Stats uses the 4 primary ROIs only; the rest are extracted for completeness
# (Marlene may want them). evc self-excludes (no RSA-preferred category).
ROIS = {
    # roi              → (peak_cope, negate)
    'face_FFA':         (1,  False),
    'face_STS':         (1,  False),
    'house_PPA':        (2,  False),
    'house_TOS':        (2,  False),
    'object_pF':        (3,  False),
    'object_LOC':       (3,  False),
    'word_VWFA':        (13, True),   # −cope13 = Word>Face
    'word_STG':         (13, True),
    'word_pSTG_liu':    (13, True),
    'word_IFG':         (13, True),
    'evc':              (19, False),
    'house_PPA_strict': (2,  False),
}
# Raw category betas for the RDM:
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

CONTROL_HEMIS = ['l', 'r']

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
        if should_skip(sid):
            continue
        sessions = get_sessions(sc)
        if not sessions or not (BASE_DIR / sid).exists():
            continue
        info   = get_sub_info(sc, sessions[0])
        pt     = is_patient(sc)
        intact = info.get('intact_hemi', '')
        if info.get('group', 'unknown') == 'nonOTC':
            continue
        subjects[sid] = {
            'code':           f"{info.get('group', '')}{sc}",
            'sessions':       [f'{s:02d}' for s in sessions],
            'anchor_ses':     f'{sessions[0]:02d}',
            'patient_hemi':   ('l' if intact == 'left' else 'r') if pt else None,
            'group':          info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi':    intact,
            'surgery_side':   ('right' if intact == 'left' else 'left') if pt else 'na',
        }
    return subjects

# ── File loaders ─────────────────────────────────────────────────────────────
def _load_searchmask(sid, anchor_ses, roi, hemi):
    p = BASE_DIR / sid / f'ses-{anchor_ses}' / 'ROIs' / f'{hemi}_{roi}_searchmask.nii.gz'
    if not p.exists():
        return None, None
    mi = _load(p)
    return mi.get_fdata() > 0, mi.affine

def _load_zstat(sid, session, anchor_ses, cope_num, negate=False):
    bm_file = BASE_DIR / sid / f'ses-{anchor_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    feat  = BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1.nii.gz' if session == anchor_ses else f'zstat1_ses{anchor_ses}.nii.gz'
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
def find_peak(sid, session, roi, hemi, info):
    """Max zstat within searchmask (Liu 2025: no threshold, native 1mm space)."""
    anchor    = info['anchor_ses']
    cope, neg = ROIS[roi]
    mask, aff = _load_searchmask(sid, anchor, roi, hemi)
    if mask is None:
        return None
    z = _load_zstat(sid, session, anchor, cope, neg)
    if z is None:
        return None
    z_masked = np.where(mask, z, -np.inf)
    if not np.isfinite(z_masked).any():
        return None
    pidx = np.unravel_index(np.argmax(z_masked), z_masked.shape)
    return {
        'peak_z':      float(z[pidx]),
        'peak_coord':  nib.affines.apply_affine(aff, np.array(pidx)),
        'affine':      aff,
        'brain_shape': z.shape,
    }

def create_sphere(peak_coord, affine, brain_shape, radius=SPHERE_RADIUS):
    """Binary sphere at 1mm anat resolution, centered on the native peak."""
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
    anchor     = info['anchor_ses']
    feat       = BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn         = 'cope1.nii.gz' if session == anchor else f'cope1_ses{anchor}.nii.gz'
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

def compute_rdm(beta_mat, valid, roi):
    """Returns (distinctiveness, {pair: fisher_z}).
    distinctiveness = mean Fisher-z between preferred category and the other three."""
    parent = roi.split('_')[0]
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

# ── Pipeline ─────────────────────────────────────────────────────────────────
def process_session(sid, info, session):
    rows    = []
    is_ctrl = info['patient_status'] == 'control'

    for roi in ROIS:
        hemis = CONTROL_HEMIS if is_ctrl else [info['patient_hemi']]
        for hemi in hemis:
            peak = find_peak(sid, session, roi, hemi, info)
            if peak is None:
                continue
            sphere_1mm   = create_sphere(peak['peak_coord'], peak['affine'], peak['brain_shape'])
            betas, valid = extract_betas(sid, session, sphere_1mm, info)
            if betas is None:
                continue
            dist, pairs  = compute_rdm(betas, valid, roi)
            if not pairs:
                continue
            n_sphere_2mm = int((zoom(sphere_1mm.astype(float), DOWNSAMPLE_FAC, order=0) > 0.5).sum())

            hemi_label = ('left' if hemi == 'l' else 'right') if is_ctrl \
                         else ('intact' if hemi == info['patient_hemi'] else 'lesioned')

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
                'category':            roi,
                'peak_x_native':       peak['peak_coord'][0],
                'peak_y_native':       peak['peak_coord'][1],
                'peak_z_native':       peak['peak_coord'][2],
                'peak_z':              peak['peak_z'],
                'n_sphere_voxels':     n_sphere_2mm,
                'liu_distinctiveness': dist,
            }
            for pair, fz in pairs.items():
                rows.append({**base, 'pair': pair, 'fisher_r': fz})
    return rows

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / OUTPUT_NAME

    subs = load_subjects()
    n_pt   = sum(v['patient_status'] == 'patient' for v in subs.values())
    n_ctrl = sum(v['patient_status'] == 'control' for v in subs.values())
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Sphere: {SPHERE_RADIUS}mm @ 1mm anat → 2mm RSA')
    print(f'ROIs: {list(ROIS.keys())}\n')

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

    print('\nROI × hemi coverage (subjects):')
    print(df.groupby(['category', 'hemi'])['subject_id'].nunique().unstack(fill_value=0))
    summary = df.drop(columns=['pair', 'fisher_r']).drop_duplicates()
    print('\nSphere voxel counts (2mm, expect ~175):')
    print(summary.groupby('category')['n_sphere_voxels'].agg(['mean', 'std']).round(1))

if __name__ == '__main__':
    main()