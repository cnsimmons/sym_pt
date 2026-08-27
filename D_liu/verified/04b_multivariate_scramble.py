#!/usr/bin/env python3
"""
04b_multivariate_scramble.py — RSA extraction with SCRAMBLE as a fifth condition.

Companion to 04_multivariate_analyses.py, NOT a replacement. Writes a separate
output file so nothing downstream breaks.

WHY
  object_LOC distinctiveness sits at r = 0.81 in controls — the least distinct
  of the four ROIs — and does not move in patients. That is not a variance
  artifact (control SD 0.291, patient SD 0.440, range +0.58 to +1.77). Object is
  a generic shape category, so its beta pattern correlates with faces, houses
  and words as visual forms. Scrambled objects are the low-level control that
  separates object-specific structure from generic form.

WHAT CHANGES
  RSA_COPES gains scramble = cope 19 (raw beta vs fixation, same as the other
  four). The RDM becomes 5 x 5, so 10 pairs instead of 6.

WHAT DOES NOT CHANGE
  `liu_distinctiveness` is still the mean Fisher-z between the ROI's preferred
  category and the other three CATEGORIES — scramble is excluded from that mean.
  The headline metric stays comparable to rsa_v1 and to Liu (2025).

  `dist_incl_scramble` is added alongside it as the 4-pair version, for
  comparison only. Do not substitute it for the primary metric without deciding
  that deliberately.

NOTE ON SELECTION
  Spheres are centred on the peak of the ROI's differential peak-finding
  contrast, which for object_LOC is cope 3 (Object > Scramble). Voxel selection
  therefore favours object over scramble in that ROI specifically. The RSA
  itself uses raw single-condition betas, so this is a selection bias on which
  voxels enter, not a circular contrast. State it in Methods.

VALIDATION
  The six original category pairs are computed from the same betas and the same
  spheres as rsa_v1, so they should match rsa_v1.csv EXACTLY (both are
  pre-harmonization). Check after running:

    python -c "
    import pandas as pd
    k=['subject_id','session','hemi','category','pair']
    a=pd.read_csv('D_liu/rsa_v1.csv')[k+['fisher_r','liu_distinctiveness']]
    b=pd.read_csv('D_liu/rsa_v2_scramble.csv')[k+['fisher_r','liu_distinctiveness']]
    m=a.merge(b,on=k,suffixes=('_1','_2'))
    print(len(m),'shared rows')
    print('max|d fisher_r|      =',(m.fisher_r_1-m.fisher_r_2).abs().max())
    print('max|d distinctiveness|=',(m.liu_distinctiveness_1-m.liu_distinctiveness_2).abs().max())
    "

  Both maxima should be ~0. Anything larger means the spheres or betas moved and
  the new file is not a clean superset.

Output: D_liu/rsa_v2_scramble.csv

Usage:
  python 04b_multivariate_scramble.py
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
OUTPUT_NAME = 'rsa_v2_scramble.csv'

SPHERE_RADIUS  = 7
FUNC_VOXEL_MM  = 2.0
ANAT_VOXEL_MM  = 1.0
DOWNSAMPLE_FAC = ANAT_VOXEL_MM / FUNC_VOXEL_MM

# Peak-finding contrast per ROI (unchanged from 04_multivariate_analyses.py):
#   1=Face>Object, 2=House>Object, 3=Object>Scramble, 13=Face>Word, 19=Scramble_raw
ROIS = {
    'face_FFA':         (1,  False),
    'face_STS':         (1,  False),
    'house_PPA':        (2,  False),
    'house_TOS':        (2,  False),
    'object_pF':        (3,  False),
    'object_LOC':       (3,  False),
    'word_VWFA':        (13, True),
    'word_STG':         (13, True),
    'word_pSTG_liu':    (13, True),
    'word_IFG':         (13, True),
    'evc':              (19, False),
    'house_PPA_strict': (2,  False),
}

# Raw category betas for the RDM. SCRAMBLE ADDED (cope 19).
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18, 'scramble': 19}

# The four real categories. Distinctiveness averages over these only.
CATEGORIES = ['face', 'house', 'object', 'word']

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
    """RSA beta extraction, downsampled to 2mm functional resolution.

    Now expects 5 conditions. Requires all 5 to be present, so a session
    missing the scramble cope is skipped rather than silently reverting to 4.
    """
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
    if len(patterns) < len(RSA_COPES):
        return None, None
    m = min(len(b) for b in patterns)
    return np.column_stack([b[:m] for b in patterns]), valid

def compute_rdm(beta_mat, valid, roi):
    """Returns (distinctiveness, dist_incl_scramble, {pair: fisher_z}).

    distinctiveness      mean Fisher-z, preferred category vs the other three
                         CATEGORIES. Scramble EXCLUDED, so this matches rsa_v1.
    dist_incl_scramble   same but with scramble in the mean (4 pairs). Reported
                         for comparison only.
    pairs                all 10 pairwise Fisher-z values from the 5x5 RDM.
    """
    parent = roi.split('_')[0]
    corr   = np.corrcoef(beta_mat.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))

    pairs = {f'{valid[i]}-{valid[j]}': float(fisher[i, j])
             for i in range(len(valid)) for j in range(i + 1, len(valid))}

    if parent not in valid:
        return np.nan, np.nan, pairs

    pidx = valid.index(parent)
    cat_idx = [i for i, v in enumerate(valid) if v in CATEGORIES and i != pidx]
    all_idx = [i for i in range(len(valid)) if i != pidx]

    dist     = float(np.mean([fisher[pidx, i] for i in cat_idx])) if cat_idx else np.nan
    dist_scr = float(np.mean([fisher[pidx, i] for i in all_idx])) if all_idx else np.nan
    return dist, dist_scr, pairs

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
            dist, dist_scr, pairs = compute_rdm(betas, valid, roi)
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
                'dist_incl_scramble':  dist_scr,
            }
            for pair, fz in pairs.items():
                rows.append({**base,
                             'pair': pair,
                             'fisher_r': fz,
                             'has_scramble': 'scramble' in pair})
    return rows

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / OUTPUT_NAME

    subs = load_subjects()
    n_pt   = sum(v['patient_status'] == 'patient' for v in subs.values())
    n_ctrl = sum(v['patient_status'] == 'control' for v in subs.values())
    print(f'Patients: {n_pt}, Controls: {n_ctrl}, Total: {len(subs)}')
    print(f'Sphere: {SPHERE_RADIUS}mm @ 1mm anat -> 2mm RSA')
    print(f'RDM conditions: {list(RSA_COPES.keys())}  (10 pairs)')
    print('liu_distinctiveness EXCLUDES scramble; dist_incl_scramble includes it')
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

    print('\nPairs present:')
    print(df['pair'].value_counts().to_string())

    print('\nROI x hemi coverage (subjects):')
    print(df.groupby(['category', 'hemi'])['subject_id'].nunique().unstack(fill_value=0))

    summary = df.drop(columns=['pair', 'fisher_r', 'has_scramble']).drop_duplicates()
    print('\nSphere voxel counts (2mm, expect ~175):')
    print(summary.groupby('category')['n_sphere_voxels'].agg(['mean', 'std']).round(1))

    print('\ndistinctiveness, 3-category vs 4-condition (controls, primary ROIs):')
    c = summary[(summary['status'] == 'control') &
                (summary['category'].isin(['object_LOC', 'house_PPA_strict',
                                           'face_FFA', 'word_VWFA']))]
    print(c.groupby('category')[['liu_distinctiveness', 'dist_incl_scramble']]
           .mean().round(3).to_string())


if __name__ == '__main__':
    main()