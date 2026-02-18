###############################################################################
# GEOMETRY PRESERVATION + MDS SHIFT NOTEBOOK
#
# Design:
#   Localization: differential COPEs (1, 2, 3, 4) — dynamic per session
#   Measurement:  raw betas (COPEs 15-18) — no circularity
#   Sphere: 6mm around each session's peak
#   Metrics:
#     1. Spatial Relocation (centroid distance T1→T2)
#     2. Geometry Preservation (RDM correlation T1→T2)
#     3. MDS Shift (Procrustes-aligned embedding distance per category)
###############################################################################

# ── CELL 1: Setup ────────────────────────────────────────────────────────────

import os, sys, time
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import label, center_of_mass
from scipy.stats import pearsonr, mannwhitneyu, ttest_ind, spearmanr
from scipy.spatial.distance import squareform
from scipy.linalg import orthogonal_procrustes
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, is_patient,
                           get_sessions, get_sub_info, _load_csv)

BASE_DIR = Path(processed_dir)
HOME_OUTPUT = Path('/home/csimmon2/geometry_results')
HOME_OUTPUT.mkdir(parents=True, exist_ok=True)

df = _load_csv()

# Exclusions
SCANNER_SESSION_DROPS = {'sub-004': ['06'], 'sub-008': ['02'], 'sub-018': ['03']}
SCANNER_SUBJECT_DROPS = ['sub-008', 'sub-018']
SUBJECTS_TO_SKIP = ['OTC108']
PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
}

# Localization contrasts (for finding peaks)
LOC_COPES = {'face': 1, 'house': 2, 'object': 3, 'word': 4}

# Measurement contrasts (raw betas — no circularity)
RSA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

CATEGORIES = ['face', 'house', 'object', 'word']
BILATERAL_CATEGORIES = ['object', 'house']

print(f'Setup complete. Output: {HOME_OUTPUT}')


# ── CELL 2: Load Subjects ───────────────────────────────────────────────────

def load_subjects(patient_only=None):
    subjects = {}
    for sub_clean in sorted(df['sub_clean'].unique()):
        if sub_clean in skip_subs:
            continue
        sid = f'sub-{sub_clean}'
        if sid in SCANNER_SUBJECT_DROPS:
            continue
        sessions = get_sessions(sub_clean)
        if not sessions:
            continue
        if sid in SCANNER_SESSION_DROPS:
            sessions = [s for s in sessions
                       if f'{s:02d}' not in SCANNER_SESSION_DROPS[sid]]
        if not sessions:
            continue
        info = get_sub_info(sub_clean, sessions[0])
        pt = is_patient(sub_clean)
        if patient_only is True and not pt:
            continue
        if patient_only is False and pt:
            continue
        if not (BASE_DIR / sid).exists():
            continue
        intact = info.get('intact_hemi', '')
        subjects[sid] = {
            'code': f"{info.get('group','')}{sub_clean}",
            'sessions': [f'{s:02d}' for s in sessions],
            'hemi': ('l' if intact == 'left' else 'r') if pt else None,
            'group': info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi': intact,
            'surgery_side': ('right' if intact == 'left' else 'left') if pt else 'na',
        }
    return subjects

ALL_PATIENTS = load_subjects(patient_only=True)
ALL_CONTROLS = load_subjects(patient_only=False)
SUBS = {**ALL_PATIENTS, **ALL_CONTROLS}

print(f'Patients: {len(ALL_PATIENTS)}, Controls: {len(ALL_CONTROLS)}, Total: {len(SUBS)}')


# ── CELL 3: ROI Extraction (Dynamic per session) ────────────────────────────

_CACHE = {}

def _load(fp):
    k = str(fp)
    if k not in _CACHE:
        _CACHE[k] = nib.load(k)
    return _CACHE[k]

def clear_cache():
    global _CACHE
    n = len(_CACHE)
    _CACHE = {}
    print(f'Cleared {n} cached files')


def extract_roi(subject_id, session, category, hemi,
                threshold_z=1.96, top_pct=0.10, min_voxels=50):
    """Dynamic ROI: localize using LOC_COPES per session."""
    info = SUBS[subject_id]
    first_ses = info['sessions'][0]
    cope_num = LOC_COPES[category]

    bm_file = BASE_DIR / subject_id / f'ses-{first_ses}' / 'anat' / 'T1w_brain_mask.nii.gz'
    bm = _load(bm_file).get_fdata() > 0 if bm_file.exists() else None

    mf = None
    for sd in ['ROIs', os.path.join('derivatives', 'rois')]:
        p = BASE_DIR / subject_id / f'ses-{first_ses}' / sd / f'{hemi}_{category}_searchmask.nii.gz'
        if p.exists():
            mf = p
            break
    if mf is None:
        return None

    mi = _load(mf)
    mask = mi.get_fdata() > 0
    affine = mi.affine

    feat = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zn = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
    zf = feat / f'cope{cope_num}.feat' / 'stats' / zn
    if not zf.exists():
        return None

    z = _load(zf).get_fdata().copy()
    if bm is not None:
        z[~bm] = 0

    supra = (z > threshold_z) & mask
    ns = supra.sum()
    if ns < min_voxels:
        return None

    top_n = max(min_voxels, int(ns * top_pct))
    top_n = min(top_n, ns)
    vals = z[supra]
    thresh = np.sort(vals)[-top_n]
    top = (z >= thresh) & supra

    labeled, nc = label(top)
    if nc == 0:
        return None

    sizes = [(labeled == i).sum() for i in range(1, nc + 1)]
    li = np.argmax(sizes) + 1
    roi = (labeled == li)
    peak_idx = np.unravel_index(np.argmax(z * roi), z.shape)

    return {
        'n_voxels': sizes[li - 1],
        'peak_z': z[peak_idx],
        'peak_coord': nib.affines.apply_affine(affine, np.array(peak_idx)),
        'centroid': nib.affines.apply_affine(affine, np.array(center_of_mass(roi))),
        'roi_mask': roi,
        'affine': affine,
        'brain_shape': z.shape,
    }


print('ROI extraction defined.')


# ── CELL 4: Sphere + Beta Extraction ────────────────────────────────────────

def create_sphere(peak_coord, affine, brain_shape, radius=6):
    grid = np.array(np.meshgrid(
        np.arange(brain_shape[0]),
        np.arange(brain_shape[1]),
        np.arange(brain_shape[2]),
        indexing='ij'
    )).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, grid)
    dists = np.linalg.norm(world - peak_coord, axis=1)
    mask = np.zeros(brain_shape, dtype=bool)
    within = grid[dists <= radius]
    for c in within:
        mask[c[0], c[1], c[2]] = True
    return mask


def extract_sphere_betas(subject_id, session, sphere_mask):
    """Extract raw beta patterns (COPEs 15-18) — independent from localization."""
    info = SUBS[subject_id]
    first_ses = info['sessions'][0]
    feat = BASE_DIR / subject_id / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    cn = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'

    patterns = []
    valid_cats = []
    for cat in CATEGORIES:
        cf = feat / f'cope{RSA_COPES[cat]}.feat' / 'stats' / cn
        if not cf.exists():
            continue
        betas = _load(cf).get_fdata()[sphere_mask]
        betas = betas[np.isfinite(betas)]
        if len(betas) > 0:
            patterns.append(betas)
            valid_cats.append(cat)

    if len(patterns) < 4:
        return None, None

    min_v = min(len(b) for b in patterns)
    patterns = [b[:min_v] for b in patterns]
    return np.column_stack(patterns), valid_cats


def compute_rdm(beta_matrix, fisher_transform=True):
    corr = np.corrcoef(beta_matrix.T)
    rdm = 1 - corr
    if fisher_transform:
        fisher = np.arctanh(np.clip(corr, -0.999, 0.999))
        return rdm, fisher
    return rdm, corr


print('Sphere + beta functions defined.')


# ── CELL 5: Compute All Metrics ─────────────────────────────────────────────

def mds_2d(rdm):
    """Classical MDS to 2D."""
    n = rdm.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (rdm ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    return coords


def compute_all_metrics(radius=6):
    """
    For each subject with 2+ post-surgery sessions, compute:
      1. Spatial relocation (centroid distance)
      2. Geometry preservation (RDM correlation T1 vs T2)
      3. MDS shift per category (Procrustes-aligned)
    
    Uses dynamic ROIs: localize per session, measure with raw betas.
    """
    print('COMPUTING GEOMETRY + MDS METRICS')
    print('=' * 70)
    print(f'Localization: COPEs {LOC_COPES}')
    print(f'Measurement:  COPEs {RSA_COPES}')
    print(f'Sphere radius: {radius}mm')
    t0 = time.time()

    spatial_rows = []
    geometry_rows = []
    mds_rows = []

    for sub_idx, (sid, info) in enumerate(sorted(SUBS.items())):
        code = info['code']
        print(f'  [{sub_idx+1}/{len(SUBS)}] {code} ({time.time()-t0:.0f}s)', end='\r')

        # Get post-surgery sessions
        post_sessions = [s for s in info['sessions']
                        if not (sid in PRE_SURGERY_SESSIONS
                                and s in PRE_SURGERY_SESSIONS[sid])]
        if len(post_sessions) < 2:
            continue

        s1, s2 = post_sessions[0], post_sessions[-1]

        # Determine hemispheres to analyze
        if info['patient_status'] == 'patient':
            hemis = [info['hemi']]
        else:
            hemis = ['l', 'r']

        for hemi in hemis:
            for category in CATEGORIES:
                # === Step 1: Dynamic ROI localization per session ===
                roi_t1 = extract_roi(sid, s1, category, hemi)
                roi_t2 = extract_roi(sid, s2, category, hemi)

                if roi_t1 is None or roi_t2 is None:
                    continue

                affine = roi_t1['affine']
                brain_shape = roi_t1['brain_shape']

                # === Step 2: Spatial relocation ===
                relocation_mm = np.linalg.norm(roi_t1['centroid'] - roi_t2['centroid'])

                # === Step 3: Build dynamic spheres ===
                sphere_t1 = create_sphere(roi_t1['centroid'], affine, brain_shape, radius)
                sphere_t2 = create_sphere(roi_t2['centroid'], affine, brain_shape, radius)

                # === Step 4: Extract raw betas at each timepoint ===
                betas_t1, cats_t1 = extract_sphere_betas(sid, s1, sphere_t1)
                betas_t2, cats_t2 = extract_sphere_betas(sid, s2, sphere_t2)

                if betas_t1 is None or betas_t2 is None:
                    continue
                if cats_t1 != cats_t2:
                    continue

                # === Step 5: Compute RDMs ===
                rdm_t1, fisher_t1 = compute_rdm(betas_t1)
                rdm_t2, fisher_t2 = compute_rdm(betas_t2)

                # === Step 6: Geometry preservation ===
                triu = np.triu_indices(4, k=1)
                r_geom, _ = pearsonr(rdm_t1[triu], rdm_t2[triu])

                # === Step 7: MDS shift ===
                try:
                    coords_t1 = mds_2d(rdm_t1)
                    coords_t2 = mds_2d(rdm_t2)
                    R, _ = orthogonal_procrustes(coords_t1, coords_t2)
                    coords_t1_aligned = coords_t1 @ R

                    mds_shifts = {}
                    for i, cat in enumerate(cats_t1):
                        mds_shifts[cat] = np.linalg.norm(
                            coords_t1_aligned[i] - coords_t2[i])
                except Exception:
                    mds_shifts = None

                # === Build row metadata ===
                if info['patient_status'] == 'patient':
                    hemi_label = 'intact'
                    surgery_side = info['surgery_side']
                else:
                    hemi_label = 'left' if hemi == 'l' else 'right'
                    surgery_side = 'na'

                cat_type = 'bilateral' if category in BILATERAL_CATEGORIES else 'unilateral'

                # Classify reorganized vs typical for unilateral
                if info['patient_status'] == 'patient' and cat_type == 'unilateral':
                    if surgery_side == 'left':
                        roi_status = 'reorganized' if category == 'word' else 'typical'
                    else:
                        roi_status = 'reorganized' if category == 'face' else 'typical'
                else:
                    roi_status = 'control' if info['patient_status'] == 'control' else 'bilateral'

                base_row = {
                    'subject': code, 'subject_id': sid,
                    'group': info['group'] if info['patient_status'] == 'patient' else 'control',
                    'status': info['patient_status'],
                    'surgery_side': surgery_side,
                    'hemi': hemi, 'hemi_label': hemi_label,
                    'category': category, 'cat_type': cat_type,
                    'roi_status': roi_status,
                    'session_1': s1, 'session_2': s2,
                }

                # Store spatial relocation
                spatial_rows.append({
                    **base_row,
                    'relocation_mm': relocation_mm,
                })

                # Store geometry preservation
                geometry_rows.append({
                    **base_row,
                    'geometry_preservation': r_geom,
                })

                # Store MDS shifts
                if mds_shifts is not None:
                    for measured_cat, shift_val in mds_shifts.items():
                        mds_rows.append({
                            **base_row,
                            'measured_category': measured_cat,
                            'measured_cat_type': ('bilateral' if measured_cat
                                                  in BILATERAL_CATEGORIES else 'unilateral'),
                            'mds_shift': shift_val,
                        })

    print(f'\n  Done: {time.time()-t0:.0f}s')

    spatial_df = pd.DataFrame(spatial_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    mds_df = pd.DataFrame(mds_rows)

    print(f'  Spatial: {len(spatial_df)} ROIs')
    print(f'  Geometry: {len(geometry_df)} ROIs')
    print(f'  MDS: {len(mds_df)} measurements')

    return spatial_df, geometry_df, mds_df


spatial_df, geometry_df, mds_df = compute_all_metrics(radius=6)
clear_cache()


# ── CELL 6: Results — Geometry Preservation ─────────────────────────────────

def report_geometry(geometry_df):
    print('\nGEOMETRY PRESERVATION (RDM correlation T1 vs T2)')
    print('=' * 70)
    print('Higher = more stable representational structure\n')

    # By group and category type
    print('BY GROUP × CATEGORY TYPE:')
    print(f"{'Group':<20} {'Bilateral':<15} {'Unilateral':<15} {'Diff'}")
    print('-' * 60)

    for grp in ['OTC', 'nonOTC', 'control']:
        gd = geometry_df[geometry_df['group'] == grp]
        bil = gd[gd['cat_type'] == 'bilateral']['geometry_preservation']
        uni = gd[gd['cat_type'] == 'unilateral']['geometry_preservation']
        if len(bil) > 0 and len(uni) > 0:
            print(f"{grp:<20} {bil.mean():<15.3f} {uni.mean():<15.3f} "
                  f"{bil.mean()-uni.mean():+.3f}")

    # OTC by surgery side
    otc = geometry_df[geometry_df['group'] == 'OTC']
    if len(otc) > 0:
        print(f'\nOTC BY SURGERY SIDE:')
        print(f"{'Side':<20} {'Bilateral':<15} {'Unilateral':<15} {'Diff'}")
        print('-' * 60)
        for side in ['left', 'right']:
            sd = otc[otc['surgery_side'] == side]
            bil = sd[sd['cat_type'] == 'bilateral']['geometry_preservation']
            uni = sd[sd['cat_type'] == 'unilateral']['geometry_preservation']
            if len(bil) > 0 and len(uni) > 0:
                print(f"OTC-{side:<16} {bil.mean():<15.3f} {uni.mean():<15.3f} "
                      f"{bil.mean()-uni.mean():+.3f}")

    # Reorganized vs typical
    uni_otc = geometry_df[(geometry_df['group'] == 'OTC') &
                          (geometry_df['cat_type'] == 'unilateral')]
    if len(uni_otc) > 0:
        print(f'\nREORGANIZED vs TYPICAL (unilateral ROIs):')
        for status in ['reorganized', 'typical']:
            sd = uni_otc[uni_otc['roi_status'] == status]
            if len(sd) > 0:
                print(f"  {status}: {sd['geometry_preservation'].mean():.3f} "
                      f"(n={len(sd)})")

    # Stats
    print(f'\nSTATISTICS:')
    otc_data = geometry_df[geometry_df['group'] == 'OTC']
    ctrl_data = geometry_df[geometry_df['group'] == 'control']

    for ct in ['bilateral', 'unilateral']:
        otc_vals = otc_data[otc_data['cat_type'] == ct]['geometry_preservation']
        ctrl_vals = ctrl_data[ctrl_data['cat_type'] == ct]['geometry_preservation']
        if len(otc_vals) > 1 and len(ctrl_vals) > 1:
            u, p = mannwhitneyu(otc_vals, ctrl_vals, alternative='less')
            print(f"  {ct}: OTC ({otc_vals.mean():.3f}) vs Control ({ctrl_vals.mean():.3f})"
                  f" — U={u}, p={p:.4f}")

    # OTC bilateral vs unilateral
    otc_bil = otc_data[otc_data['cat_type'] == 'bilateral']['geometry_preservation']
    otc_uni = otc_data[otc_data['cat_type'] == 'unilateral']['geometry_preservation']
    if len(otc_bil) > 1 and len(otc_uni) > 1:
        u, p = mannwhitneyu(otc_bil, otc_uni)
        print(f"  OTC bil vs uni: U={u}, p={p:.4f} "
              f"(bil={otc_bil.mean():.3f}, uni={otc_uni.mean():.3f})")


report_geometry(geometry_df)


# ── CELL 7: Results — MDS Shift ─────────────────────────────────────────────

def report_mds(mds_df):
    print('\nMDS EMBEDDING SHIFT (Procrustes-aligned)')
    print('=' * 70)
    print('Higher = more change in representational geometry\n')

    # By group and measured category type
    print('BY GROUP × MEASURED CATEGORY TYPE:')
    print(f"{'Group':<20} {'Bilateral':<15} {'Unilateral':<15} {'Diff'}")
    print('-' * 60)

    for grp in ['OTC', 'nonOTC', 'control']:
        gd = mds_df[mds_df['group'] == grp]
        bil = gd[gd['measured_cat_type'] == 'bilateral']['mds_shift']
        uni = gd[gd['measured_cat_type'] == 'unilateral']['mds_shift']
        if len(bil) > 0 and len(uni) > 0:
            print(f"{grp:<20} {bil.mean():<15.3f} {uni.mean():<15.3f} "
                  f"{bil.mean()-uni.mean():+.3f}")

    # OTC by surgery side
    otc = mds_df[mds_df['group'] == 'OTC']
    if len(otc) > 0:
        print(f'\nOTC BY SURGERY SIDE:')
        print(f"{'Side':<20} {'Bilateral':<15} {'Unilateral':<15} {'Diff'}")
        print('-' * 60)
        for side in ['left', 'right']:
            sd = otc[otc['surgery_side'] == side]
            bil = sd[sd['measured_cat_type'] == 'bilateral']['mds_shift']
            uni = sd[sd['measured_cat_type'] == 'unilateral']['mds_shift']
            if len(bil) > 0 and len(uni) > 0:
                print(f"OTC-{side:<16} {bil.mean():<15.3f} {uni.mean():<15.3f} "
                      f"{bil.mean()-uni.mean():+.3f}")

    # Stats
    print(f'\nSTATISTICS:')
    otc_data = mds_df[mds_df['group'] == 'OTC']
    ctrl_data = mds_df[mds_df['group'] == 'control']

    for ct in ['bilateral', 'unilateral']:
        otc_vals = otc_data[otc_data['measured_cat_type'] == ct]['mds_shift']
        ctrl_vals = ctrl_data[ctrl_data['measured_cat_type'] == ct]['mds_shift']
        if len(otc_vals) > 1 and len(ctrl_vals) > 1:
            u, p = mannwhitneyu(otc_vals, ctrl_vals, alternative='greater')
            print(f"  {ct}: OTC ({otc_vals.mean():.3f}) vs Control ({ctrl_vals.mean():.3f})"
                  f" — U={u}, p={p:.4f}")

    # Permutation test: OTC bilateral vs unilateral
    otc_bil = otc_data[otc_data['measured_cat_type'] == 'bilateral']['mds_shift']
    otc_uni = otc_data[otc_data['measured_cat_type'] == 'unilateral']['mds_shift']
    if len(otc_bil) > 1 and len(otc_uni) > 1:
        observed = otc_bil.mean() - otc_uni.mean()
        combined = np.concatenate([otc_bil.values, otc_uni.values])
        n_bil = len(otc_bil)
        rng = np.random.default_rng(42)
        n_perm = 10000
        perm_diffs = []
        for _ in range(n_perm):
            shuf = rng.permutation(combined)
            perm_diffs.append(shuf[:n_bil].mean() - shuf[n_bil:].mean())
        perm_diffs = np.array(perm_diffs)
        p_perm = np.mean(perm_diffs >= observed)
        print(f"\n  Permutation test (OTC bil > uni):")
        print(f"  Observed diff: {observed:.3f}, p={p_perm:.4f}")
        print(f"  95% null CI: [{np.percentile(perm_diffs, 2.5):.3f}, "
              f"{np.percentile(perm_diffs, 97.5):.3f}]")


report_mds(mds_df)


# ── CELL 8: Results — Spatial Relocation ────────────────────────────────────

def report_spatial(spatial_df):
    print('\nSPATIAL RELOCATION (centroid distance T1→T2, mm)')
    print('=' * 70)

    print(f"{'Group':<20} {'Bilateral':<15} {'Unilateral':<15} {'Diff'}")
    print('-' * 60)

    for grp in ['OTC', 'nonOTC', 'control']:
        gd = spatial_df[spatial_df['group'] == grp]
        bil = gd[gd['cat_type'] == 'bilateral']['relocation_mm']
        uni = gd[gd['cat_type'] == 'unilateral']['relocation_mm']
        if len(bil) > 0 and len(uni) > 0:
            print(f"{grp:<20} {bil.mean():<15.1f} {uni.mean():<15.1f} "
                  f"{bil.mean()-uni.mean():+.1f}")

    otc = spatial_df[spatial_df['group'] == 'OTC']
    if len(otc) > 0:
        print(f'\nOTC BY SURGERY SIDE:')
        for side in ['left', 'right']:
            sd = otc[otc['surgery_side'] == side]
            bil = sd[sd['cat_type'] == 'bilateral']['relocation_mm']
            uni = sd[sd['cat_type'] == 'unilateral']['relocation_mm']
            if len(bil) > 0 and len(uni) > 0:
                print(f"  OTC-{side}: bil={bil.mean():.1f}mm, uni={uni.mean():.1f}mm, "
                      f"diff={bil.mean()-uni.mean():+.1f}mm")


report_spatial(spatial_df)


# ── CELL 9: Double Dissociation Test ────────────────────────────────────────

def double_dissociation(spatial_df, geometry_df, mds_df):
    """Test: unilateral relocates MORE but bilateral changes representation MORE."""
    print('\nDOUBLE DISSOCIATION TEST')
    print('=' * 70)

    otc_spatial = spatial_df[spatial_df['group'] == 'OTC']
    otc_geom = geometry_df[geometry_df['group'] == 'OTC']
    otc_mds = mds_df[mds_df['group'] == 'OTC']

    # Spatial: unilateral > bilateral?
    sp_bil = otc_spatial[otc_spatial['cat_type'] == 'bilateral']['relocation_mm']
    sp_uni = otc_spatial[otc_spatial['cat_type'] == 'unilateral']['relocation_mm']
    if len(sp_bil) > 1 and len(sp_uni) > 1:
        u, p = mannwhitneyu(sp_uni, sp_bil, alternative='greater')
        print(f'Spatial relocation (uni > bil): '
              f'uni={sp_uni.mean():.1f}mm, bil={sp_bil.mean():.1f}mm, p={p:.4f}')

    # Geometry: bilateral < unilateral (less preserved)?
    gp_bil = otc_geom[otc_geom['cat_type'] == 'bilateral']['geometry_preservation']
    gp_uni = otc_geom[otc_geom['cat_type'] == 'unilateral']['geometry_preservation']
    if len(gp_bil) > 1 and len(gp_uni) > 1:
        u, p = mannwhitneyu(gp_bil, gp_uni, alternative='less')
        print(f'Geometry preservation (bil < uni): '
              f'bil={gp_bil.mean():.3f}, uni={gp_uni.mean():.3f}, p={p:.4f}')

    # MDS: bilateral > unilateral (more shift)?
    ms_bil = otc_mds[otc_mds['measured_cat_type'] == 'bilateral']['mds_shift']
    ms_uni = otc_mds[otc_mds['measured_cat_type'] == 'unilateral']['mds_shift']
    if len(ms_bil) > 1 and len(ms_uni) > 1:
        u, p = mannwhitneyu(ms_bil, ms_uni, alternative='greater')
        print(f'MDS shift (bil > uni): '
              f'bil={ms_bil.mean():.3f}, uni={ms_uni.mean():.3f}, p={p:.4f}')

    # Correlation: relocation vs geometry
    merged = otc_spatial.merge(otc_geom,
                               on=['subject_id', 'hemi', 'category'],
                               suffixes=('_sp', '_gm'))
    if len(merged) > 3:
        r, p = spearmanr(merged['relocation_mm'], merged['geometry_preservation'])
        print(f'\nCorrelation: relocation vs geometry: rho={r:.3f}, p={p:.4f}')
        print(f'  (Negative = categories that move more also change representation more)')


double_dissociation(spatial_df, geometry_df, mds_df)


# ── CELL 10: Save Results ──────────────────────────────────────────────────

spatial_df.to_csv(HOME_OUTPUT / 'spatial_relocation.csv', index=False)
geometry_df.to_csv(HOME_OUTPUT / 'geometry_preservation.csv', index=False)
mds_df.to_csv(HOME_OUTPUT / 'mds_shift.csv', index=False)
print(f'\nSaved to {HOME_OUTPUT}')