#!/usr/bin/env python3
"""
add_ifg.py — Add word_IFG to the Liu pipeline (temporary, standalone).

Replicates the relevant logic from:
  01_create_searchmasks.py     — IFG searchmask creation (HO indices [4, 5])
  liu_recreation_csv_v2.py     — IFG peak / sphere / distinctiveness / sum-selec

Output:
  liu_exact_replication_v2_IFG_addon.csv  (one row per subj × ses × hemi × pair)

Idempotent and re-runnable:
  - Searchmasks: skip if present unless --force-masks
  - CSV: re-runs only missing subject-sessions unless --force-csv. New subjects
    auto-detected via sym_pt_params.
  - Use --sub <ID> for a single subject (e.g. after re-running their HighLevel)

After running: add ONE line to notebook cell 4 (after `le = pd.read_csv(LIU_CSV)`):
    addon = LIU_CSV.parent / 'liu_exact_replication_v2_IFG_addon.csv'
    if addon.exists():
        le = pd.concat([le, pd.read_csv(addon)], ignore_index=True)

Usage:
  python add_ifg.py                # process all subjects
  python add_ifg.py --sub 021      # single subject
  python add_ifg.py --force-masks  # rebuild searchmasks even if present
  python add_ifg.py --force-csv    # rebuild all CSV rows from scratch
"""
import os
import sys
import argparse
import subprocess
import shutil
import time
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

CATEGORY      = 'word_IFG'
HO_INDICES    = [4, 5]                # IFG pars triangularis + opercularis
HO_NAMES      = ['IFG pars triangularis', 'IFG pars opercularis']
COPE          = 13                    # Face>Word (we negate for Word>Face)
NEGATE        = True

FSLDIR        = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
PROB_ATLAS    = f'{FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz'
PROB_THRESH   = 25                    # %, matches existing pipeline

BASE_DIR      = Path(processed_dir)
OUTPUT_DIR    = Path('/user_data/csimmon2/git_repos/sym_pt')
OUTPUT_NAME   = 'liu_exact_replication_v2_IFG_addon.csv'

SPHERE_RADIUS  = 7
FUNC_VOXEL_MM  = 2.0
ANAT_VOXEL_MM  = 1.0
DOWNSAMPLE_FAC = ANAT_VOXEL_MM / FUNC_VOXEL_MM
SEL_Z_THRESH   = float(norm.ppf(0.99))
SEL_RESCALE    = 1000.0

EXTRA_SKIP = {'sub-017', 'control083', 'control085'}
PRE_SURGERY_SESSIONS = {
    'sub-021': {'01'}, 'sub-045': {'01'}, 'sub-047': {'01'}, 'sub-049': {'01'},
    'sub-070': {'01'}, 'sub-073': {'01'}, 'sub-081': {'01'}, 'sub-086': {'01'},
    'sub-108': {'02'},
}
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

_CACHE = {}
def _load(fp):
    k = str(fp)
    if k not in _CACHE:
        _CACHE[k] = nib.load(k)
    return _CACHE[k]


# ── Searchmask creation ──────────────────────────────────────────────────────

def extract_hemisphere_mask(atlas_data, indices, hemisphere, threshold=PROB_THRESH):
    combined = np.zeros(atlas_data.shape[:3], dtype=float)
    for idx in indices:
        combined = np.maximum(combined, atlas_data[:, :, :, idx])
    binary = combined > threshold
    midpoint = atlas_data.shape[0] // 2
    out = np.zeros_like(binary)
    if hemisphere == 'l':
        out[midpoint:, :, :] = binary[midpoint:, :, :]
    else:
        out[:midpoint, :, :] = binary[:midpoint, :, :]
    return out


def warp_mask(mni_path, ref_brain, mni2anat, output_path):
    subprocess.run(
        ['flirt', '-in', mni_path, '-ref', ref_brain,
         '-out', output_path, '-applyxfm', '-init', mni2anat,
         '-interp', 'nearestneighbour'],
        check=True, capture_output=True)
    subprocess.run(['fslmaths', output_path, '-bin', output_path],
                   check=True, capture_output=True)


def create_ifg_searchmasks(sub_clean, atlas_img, atlas_data, force=False):
    """Generate l/r word_IFG searchmasks for one subject's first session."""
    sessions = get_sessions(sub_clean)
    if not sessions:
        return 0
    first_ses = f'{sessions[0]:02d}'
    anat_dir  = BASE_DIR / f'sub-{sub_clean}' / f'ses-{first_ses}' / 'anat'
    roi_dir   = BASE_DIR / f'sub-{sub_clean}' / f'ses-{first_ses}' / 'ROIs'
    ref_brain = anat_dir / 'T1w_brain.nii.gz'
    mni2anat  = anat_dir / 'mni2anat.mat'

    if not ref_brain.exists() or not mni2anat.exists():
        print(f'  sub-{sub_clean} ses-{first_ses}: missing anat — SKIP')
        return 0
    roi_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(f'/tmp/ifg_sub-{sub_clean}')
    tmp_dir.mkdir(exist_ok=True)
    n_made = 0
    for hemi in ['l', 'r']:
        out = roi_dir / f'{hemi}_{CATEGORY}_searchmask.nii.gz'
        if out.exists() and not force:
            continue
        hmask = extract_hemisphere_mask(atlas_data, HO_INDICES, hemi)
        if hmask.sum() == 0:
            print(f'  sub-{sub_clean} {hemi}_{CATEGORY}: empty mask at {PROB_THRESH}%')
            continue
        tmp = tmp_dir / f'{hemi}_mni.nii.gz'
        nib.save(nib.Nifti1Image(hmask.astype(np.float32), atlas_img.affine), tmp)
        warp_mask(str(tmp), str(ref_brain), str(mni2anat), str(out))
        nvox = int((nib.load(out).get_fdata() > 0).sum())
        print(f'  sub-{sub_clean} {hemi}_{CATEGORY}: {nvox:>6,} voxels')
        n_made += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return n_made


# ── Extraction (mirrors liu_recreation_csv_v2.py) ────────────────────────────

def _load_searchmask(sid, first_ses, hemi):
    p = BASE_DIR / sid / f'ses-{first_ses}' / 'ROIs' / f'{hemi}_{CATEGORY}_searchmask.nii.gz'
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


def find_peak(sid, session, hemi, info):
    first_ses = info['sessions'][0]
    mask, aff = _load_searchmask(sid, first_ses, hemi)
    if mask is None:
        return None
    z = _load_zstat(sid, session, first_ses, COPE, NEGATE)
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
    grid  = np.array(np.meshgrid(
        np.arange(brain_shape[0]), np.arange(brain_shape[1]),
        np.arange(brain_shape[2]), indexing='ij')).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    mask  = np.zeros(brain_shape, dtype=bool)
    for c in grid[dists <= radius]:
        mask[c[0], c[1], c[2]] = True
    return mask


def extract_betas(sid, session, sphere_1mm, info):
    first_ses = info['sessions'][0]
    feat = BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn   = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'
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
        return None, None, sphere_2mm
    m = min(len(b) for b in patterns)
    return np.column_stack([b[:m] for b in patterns]), valid, sphere_2mm


def compute_distinctiveness(beta_mat, valid):
    """Preferred = word (parent of word_IFG)."""
    if 'word' not in valid:
        return np.nan, {}
    corr   = np.corrcoef(beta_mat.T)
    fisher = np.arctanh(np.clip(corr, -0.999, 0.999))
    pidx   = valid.index('word')
    others = [i for i in range(len(valid)) if i != pidx]
    dist   = float(np.mean([fisher[pidx, i] for i in others]))
    pairs  = {f'{valid[i]}-{valid[j]}': float(fisher[i, j])
              for i in range(len(valid)) for j in range(i + 1, len(valid))}
    return dist, pairs


def compute_sum_selectivity(sid, session, hemi, info):
    first_ses = info['sessions'][0]
    mask, _ = _load_searchmask(sid, first_ses, hemi)
    if mask is None:
        return {'sum_selec_norm': np.nan, 'mean_act': np.nan, 'volume': 0, 'n_searchmask': 0}
    z = _load_zstat(sid, session, first_ses, COPE, NEGATE)
    n_searchmask = int(mask.sum())
    if z is None:
        return {'sum_selec_norm': np.nan, 'mean_act': np.nan, 'volume': 0, 'n_searchmask': n_searchmask}
    supra = mask & (z > SEL_Z_THRESH)
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


# ── Subject loader (mirrors liu_recreation_csv_v2.py) ────────────────────────

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


# ── Per-session extraction ───────────────────────────────────────────────────

def process_session(sid, info, session):
    rows = []
    is_ctrl = info['patient_status'] == 'control'
    hemis = ['l', 'r'] if is_ctrl else [info['patient_hemi']]
    for hemi in hemis:
        roi = find_peak(sid, session, hemi, info)
        if roi is None:
            continue
        sphere_1mm   = create_sphere(roi['peak_coord'], roi['affine'], roi['brain_shape'])
        betas, valid, sphere_2mm = extract_betas(sid, session, sphere_1mm, info)
        if betas is None:
            continue
        dist, pairs = compute_distinctiveness(betas, valid)
        sel         = compute_sum_selectivity(sid, session, hemi, info)
        n_sphere_2mm = int(sphere_2mm.sum())

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
            'category':            CATEGORY,
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
            rows.append({**base, 'pair': None, 'fisher_r': np.nan})
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sub', type=str, help='Single subject (e.g. 021)')
    p.add_argument('--force-masks', action='store_true', help='Rebuild searchmasks')
    p.add_argument('--force-csv',   action='store_true', help='Rebuild CSV from scratch')
    args = p.parse_args()

    print('=' * 70)
    print(f'add_ifg.py — generating word_IFG ({HO_NAMES})')
    print('=' * 70)

    subs = load_subjects()
    if args.sub:
        sid_target = f'sub-{args.sub.replace("sub-", "")}'
        subs = {sid_target: subs[sid_target]} if sid_target in subs else {}
        if not subs:
            print(f'ERROR: {sid_target} not in loaded subjects')
            return
    print(f'Subjects to process: {len(subs)}')

    # ── Step 1: searchmasks ──
    print('\n[1/2] Searchmasks')
    print(f'Loading atlas: {PROB_ATLAS}')
    atlas_img  = nib.load(PROB_ATLAS)
    atlas_data = atlas_img.get_fdata()
    n_masks = 0
    for sid in sorted(subs):
        sub_clean = sid.replace('sub-', '')
        n_masks += create_ifg_searchmasks(sub_clean, atlas_img, atlas_data,
                                          force=args.force_masks)
    print(f'  Created/refreshed {n_masks} masks')

    # ── Step 2: extract IFG rows ──
    print('\n[2/2] Extraction')
    out_path = OUTPUT_DIR / OUTPUT_NAME

    # Load existing addon (if any) to preserve already-processed subjects
    if out_path.exists() and not args.force_csv:
        existing = pd.read_csv(out_path)
        done = set(zip(existing['subject_id'], existing['session']))
        print(f'  Existing addon: {len(existing)} rows, {len(done)} subj-sessions')
    else:
        existing = pd.DataFrame()
        done = set()

    new_rows = []
    t0 = time.time()
    for i, (sid, info) in enumerate(sorted(subs.items()), 1):
        for session in info['sessions']:
            if sid in PRE_SURGERY_SESSIONS and session in PRE_SURGERY_SESSIONS[sid]:
                continue
            if (sid, session) in done and not args.force_csv:
                continue
            session_rows = process_session(sid, info, session)
            new_rows.extend(session_rows)
        print(f'  [{i}/{len(subs)}] {info["code"]:8s} ({time.time()-t0:.0f}s)', end='\r')
    print()

    # ── Merge & save ──
    new_df = pd.DataFrame(new_rows)
    if not args.force_csv and len(existing):
        # Drop any rows from existing where (subject_id, session) overlaps with new (overwrite)
        if len(new_df):
            new_keys = set(zip(new_df['subject_id'], new_df['session']))
            existing = existing[~existing.apply(
                lambda r: (r['subject_id'], r['session']) in new_keys, axis=1)]
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    if len(df):
        df.to_csv(out_path, index=False)
        print(f'\nSaved: {out_path}')
        print(f'  Total rows: {len(df)}')
        print(f'  Subjects:   {df["subject_id"].nunique()}')
        print(f'  Sessions:   {len(df.groupby(["subject_id","session"]))}')
        summary = df.drop(columns=['pair', 'fisher_r']).drop_duplicates()
        print(f'\nROI x hemi coverage:')
        print(summary.groupby('hemi')['subject_id'].nunique())
    else:
        print('\nNo new rows to write.')


if __name__ == '__main__':
    main()