#!/usr/bin/env python3
"""
sensitivity_analysis.py — Liu (2025) patient-overlap sensitivity analysis.

Reruns the cross-sectional tests with the 4 Liu-overlap patients excluded,
and outputs a single side-by-side CSV comparing primary (n=22) vs
sensitivity (n=18) statistics.

Liu overlap patients (excluded in sensitivity sample):
  sub-004 (UD, LH-intact)
  sub-021 (TC, RH-intact)
  sub-044 (SN, RH-intact)
  sub-099 (KN, RH-intact)

→ Sensitivity sample: 10 LH-intact + 8 RH-intact = 18 OTC patients.

Tests
-----
  1. Distance to 2D control centroid (peak_x_mni, peak_y_mni)
  2. Sum-selectivity (log10 sum_selec_norm)
  3. Liu distinctiveness (Fisher-z, raw)
  5. WTA composition within surviving TFCE clusters
     (requires Test 4 output in TFCE_DIR_PRIMARY and TFCE_DIR_SENS)

All perm tests: 10,000 iterations, seed 42, two-sided.
FDR: Benjamini-Hochberg within measure (20 tests per measure = 10 ROIs × 2 hemis).
Raw p-values are also reported.

Usage
-----
  python sensitivity_analysis.py                # all available tests
  python sensitivity_analysis.py --skip-tfce    # Tests 1, 2, 3 only
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import nibabel as nib
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

# ── Configuration ────────────────────────────────────────────────────────────
LIU_OVERLAP_SUBS = ['sub-004', 'sub-021', 'sub-044', 'sub-099']
N_PERM = 10000
SEED = 42

ALL_ROIS = ['face_FFA', 'face_STS', 'house_PPA', 'house_TOS',
            'object_LOC', 'object_pF', 'word_VWFA', 'word_STG',
            'word_pSTG_liu', 'word_IFG']

# Test 5 (TFCE WTA) constants — match voxel_allegiance_xs_liu cells 32–34
CAT_COPES   = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
CATEGORIES  = ['face', 'house', 'object', 'word']
WTA_THRESH  = 2.33    # selectivity threshold for WTA tally
FWE_THRESH  = 0.95    # 1 − p_FWE; voxels > .95 survive at p<.05 FWE

PEAK_CSV         = Path(processed_dir) / 'group_results' / 'peak_coords' / 'peak_coords_mni.csv'
TFCE_DIR_PRIMARY = Path(processed_dir) / 'group_results' / 'tfce_votc'
TFCE_DIR_SENS    = Path(processed_dir) / 'group_results' / 'tfce_votc_excl_liu'
OUT_DIR          = Path(processed_dir) / 'group_results' / 'sensitivity_liu_overlap'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading + subject filtering ─────────────────────────────────────────
def load_peak_data():
    """Load peak_coords_mni.csv and de-duplicate to one row per (sub, cat, hemi).

    Per-pair rows (face-house, face-object, …) duplicate non-pair-dependent
    columns (sum_selec_norm, peak_x_mni, liu_distinctiveness, etc.), so
    drop_duplicates is safe for the perm-test columns we use.
    """
    df = pd.read_csv(PEAK_CSV)
    return df.drop_duplicates(subset=['subject_id', 'category', 'hemi']).copy()


def build_subject_lists(df, exclude_liu):
    """Return (ctrl_subs, pt_lh_subs, pt_rh_subs) for the chosen sample."""
    ctrl_subs = sorted(df[df['group'] == 'control']['subject_id'].unique().tolist())
    pt_df     = df[df['group'] == 'OTC']
    pt_subs   = sorted(pt_df['subject_id'].unique().tolist())
    if exclude_liu:
        pt_subs = [s for s in pt_subs if s not in LIU_OVERLAP_SUBS]
    pt_lh = sorted(s for s in pt_subs
                   if pt_df[pt_df['subject_id'] == s]['intact_hemi'].iloc[0] == 'left')
    pt_rh = sorted(s for s in pt_subs
                   if pt_df[pt_df['subject_id'] == s]['intact_hemi'].iloc[0] == 'right')
    return ctrl_subs, pt_lh, pt_rh


# ── Permutation test ────────────────────────────────────────────────────────
def perm_test(ctrl_vals, pt_vals, n_perm=N_PERM, seed=SEED):
    """Two-sided permutation test of mean difference (pt − ctrl)."""
    ctrl = np.asarray(ctrl_vals, float)
    pt   = np.asarray(pt_vals,   float)
    ctrl = ctrl[~np.isnan(ctrl)]
    pt   = pt[~np.isnan(pt)]
    if len(ctrl) < 3 or len(pt) < 2:
        return np.nan, np.nan, len(ctrl), len(pt)
    obs  = pt.mean() - ctrl.mean()
    pool = np.concatenate([ctrl, pt])
    n_pt = len(pt)
    rng  = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pool)
        null[i] = perm[:n_pt].mean() - perm[n_pt:].mean()
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, p, len(ctrl), len(pt)


# ── Test 1: 2D distance to control centroid ─────────────────────────────────
def run_distance_test(df, ctrl_subs, pt_lh, pt_rh, label):
    """Distance from each subject's peak (x,y in MNI) to the control centroid."""
    rows = []
    for intact, hemi_label in [('l', 'LH'), ('r', 'RH')]:
        pt_subs_h = pt_lh if intact == 'l' else pt_rh
        for roi in ALL_ROIS:
            c = df[df['subject_id'].isin(ctrl_subs) &
                   (df['category'] == roi) & (df['hemi'] == intact)] \
                [['peak_x_mni', 'peak_y_mni']].dropna().values
            if len(c) < 3:
                continue
            centroid = c.mean(0)
            ct_d = np.linalg.norm(c - centroid, axis=1)
            pt_d = []
            for sub in pt_subs_h:
                r = df[(df['subject_id'] == sub) & (df['category'] == roi) &
                       (df['hemi'] == intact)] \
                    [['peak_x_mni', 'peak_y_mni']].dropna().values
                if len(r):
                    pt_d.append(np.linalg.norm(r[0] - centroid))
            pt_d = np.array(pt_d)
            obs, p, n_c, n_p = perm_test(ct_d, pt_d)
            rows.append({
                'measure':   'distance_2d',
                'sample':    label,
                'hemi':      intact,
                'roi':       roi,
                'n_ctrl':    n_c,
                'n_pt':      n_p,
                'ctrl_mean': ct_d.mean(),
                'pt_mean':   pt_d.mean() if len(pt_d) else np.nan,
                'delta':     obs,
                'p_raw':     p,
            })
    return pd.DataFrame(rows)


# ── Tests 2, 3: generic per-ROI perm test on a column ───────────────────────
def run_column_test(df, ctrl_subs, pt_lh, pt_rh, label,
                    measure_name, column, transform=None):
    """Per-ROI × hemi perm test on a column in peak_coords_mni.csv."""
    rows = []
    for intact, hemi_label in [('l', 'LH'), ('r', 'RH')]:
        pt_subs_h = pt_lh if intact == 'l' else pt_rh
        for roi in ALL_ROIS:
            ct = df[df['subject_id'].isin(ctrl_subs) &
                    (df['category'] == roi) & (df['hemi'] == intact)] \
                 [column].dropna().values
            pt = df[df['subject_id'].isin(pt_subs_h) &
                    (df['category'] == roi) & (df['hemi'] == intact)] \
                 [column].dropna().values
            if transform == 'log10':
                ct = np.log10(ct[ct > 0])
                pt = np.log10(pt[pt > 0])
            obs, p, n_c, n_p = perm_test(ct, pt)
            rows.append({
                'measure':   measure_name,
                'sample':    label,
                'hemi':      intact,
                'roi':       roi,
                'n_ctrl':    n_c,
                'n_pt':      n_p,
                'ctrl_mean': ct.mean() if len(ct) else np.nan,
                'pt_mean':   pt.mean() if len(pt) else np.nan,
                'delta':     obs,
                'p_raw':     p,
            })
    return pd.DataFrame(rows)


# ── BH-FDR within measure ───────────────────────────────────────────────────
def apply_bh_fdr_within_measure(df):
    """BH-FDR across the 20 tests within each measure × sample."""
    out_parts = []
    for (measure, sample), g in df.groupby(['measure', 'sample'], sort=False):
        g = g.copy()
        valid = g['p_raw'].notna()
        g['p_bh_fdr'] = np.nan
        if valid.sum() > 0:
            _, p_adj, _, _ = multipletests(g.loc[valid, 'p_raw'].values, method='fdr_bh')
            g.loc[valid, 'p_bh_fdr'] = p_adj
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


# ── Side-by-side reshape: primary vs sensitivity ────────────────────────────
def pivot_side_by_side(long_df):
    """Reshape long → wide: one row per (measure, roi, hemi), with
    primary_* and sensitivity_* columns side by side."""
    cols = ['n_pt', 'ctrl_mean', 'pt_mean', 'delta', 'p_raw', 'p_bh_fdr']
    primary = long_df[long_df['sample'] == 'primary'].copy()
    sens    = long_df[long_df['sample'] == 'sensitivity'].copy()
    primary = primary.set_index(['measure', 'hemi', 'roi'])
    sens    = sens.set_index(['measure', 'hemi', 'roi'])

    wide = pd.DataFrame(index=primary.index)
    wide['n_ctrl'] = primary['n_ctrl']
    for c in cols:
        wide[f'primary_{c}']     = primary[c]
        wide[f'sensitivity_{c}'] = sens[c]
    return wide.reset_index()


# ── Pretty-print summary tables ─────────────────────────────────────────────
def print_perm_tables(wide):
    for measure in ['distance_2d', 'sum_sel', 'distinctiveness']:
        sub = wide[wide['measure'] == measure]
        if not len(sub):
            continue
        print(f'\n=== {measure} ===')
        print(f'{"Hemi":>4s} {"ROI":18s} {"n_pri":>5s} {"n_sen":>5s} '
              f'{"Δ_pri":>7s} {"p_pri":>7s} {"q_pri":>7s} '
              f'{"Δ_sen":>7s} {"p_sen":>7s} {"q_sen":>7s}')
        print('-' * 95)
        for _, r in sub.iterrows():
            def fmt(v, w=7, prec=3):
                return f'{v:>{w}.{prec}f}' if pd.notna(v) else ' ' * w
            print(f'{r["hemi"]:>4s} {r["roi"]:18s} '
                  f'{int(r["primary_n_pt"]):>5d} {int(r["sensitivity_n_pt"]):>5d} '
                  f'{fmt(r["primary_delta"])} {fmt(r["primary_p_raw"], 7, 4)} '
                  f'{fmt(r["primary_p_bh_fdr"], 7, 4)} '
                  f'{fmt(r["sensitivity_delta"])} {fmt(r["sensitivity_p_raw"], 7, 4)} '
                  f'{fmt(r["sensitivity_p_bh_fdr"], 7, 4)}')


# ── Test 5: WTA composition within surviving TFCE clusters ──────────────────
def build_control_wta(df, ctrl_subs, sample_label):
    """Build a 3D WTA map from the control mean zstat (cat-vs-all, copes 6–9).

    Per voxel: argmax across the 4 categories. Selectivity gate at WTA_THRESH.
    Note: controls are not excluded in the sensitivity sample (only patients are
    excluded), so this is identical for primary and sensitivity. We rebuild
    anyway for transparency.
    """
    print(f'    [{sample_label}] Building control WTA map (n={len(ctrl_subs)} controls)...')
    ctrl_mean = {}
    expected_shape = None
    for cat, cope in CAT_COPES.items():
        vols = []
        for sid in ctrl_subs:
            ses = df[df['subject_id'] == sid]['session'].iloc[0]
            ses_str = f'{int(ses):02d}'
            zp = (Path(processed_dir) / sid / f'ses-{ses_str}' / 'derivatives' / 'fsl'
                  / 'loc' / 'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats'
                  / 'zstat1_mni.nii.gz')
            if not zp.exists():
                continue
            v = nib.load(zp).get_fdata()
            if expected_shape is None:
                expected_shape = v.shape
            if v.shape != expected_shape:
                continue
            vols.append(v)
        if not vols:
            return None, None
        ctrl_mean[cat] = np.mean(np.stack(vols), axis=0)
    stack  = np.stack([ctrl_mean[c] for c in CATEGORIES])
    wta    = np.argmax(stack, axis=0)
    peak_z = np.max(stack, axis=0)
    return wta, peak_z


def tally_wta_in_clusters(tfce_dir, wta, peak_z, sample_label):
    """For each surviving TFCE cluster, count control-WTA categories within."""
    if not tfce_dir.exists():
        print(f'    [{sample_label}] TFCE dir missing: {tfce_dir}')
        return pd.DataFrame()
    rows = []
    # Hemi-baseline (% of selective voxels in each hemi assigned to each cat)
    hemi_mask = {'l': np.zeros_like(peak_z, dtype=bool),
                 'r': np.zeros_like(peak_z, dtype=bool)}
    mid = peak_z.shape[0] // 2
    hemi_mask['r'][:mid] = True
    hemi_mask['l'][mid:] = True
    selective = peak_z >= WTA_THRESH
    base = {}
    for h in ('l', 'r'):
        valid = selective & hemi_mask[h]
        if not valid.any():
            base[h] = {c: 0.0 for c in CATEGORIES}
        else:
            base[h] = {c: 100.0 * (wta[valid] == i).mean()
                       for i, c in enumerate(CATEGORIES)}

    for cat in CATEGORIES:
        for h in ('l', 'r'):
            test_dir = tfce_dir / f'{cat}_{h}_pt_vs_ctrl'
            for tstat, contrast in [(1, 'ctrl>pt'), (2, 'pt>ctrl')]:
                f = test_dir / f'rand_tfce_corrp_tstat{tstat}.nii.gz'
                if not f.exists():
                    continue
                vol = nib.load(str(f)).get_fdata()
                cluster_mask = vol > FWE_THRESH
                n_cluster    = int(cluster_mask.sum())
                if n_cluster == 0:
                    rows.append({
                        'sample': sample_label, 'tfce_cat': cat, 'tfce_hemi': h,
                        'tfce_contrast': contrast, 'n_cluster_vox': 0,
                        'n_selective': 0, 'pct_selective': 0,
                        **{f'{c}_pct': np.nan for c in CATEGORIES},
                        **{f'{c}_baseline_pct': base[h][c] for c in CATEGORIES},
                        **{f'{c}_enrichment': np.nan for c in CATEGORIES},
                    })
                    continue
                voxel_cats   = wta[cluster_mask]
                voxel_z      = peak_z[cluster_mask]
                valid        = voxel_z >= WTA_THRESH
                cats_valid   = voxel_cats[valid]
                total_valid  = len(cats_valid)
                counts       = Counter(cats_valid.tolist())
                pcts = {c: (100.0 * counts.get(i, 0) / total_valid if total_valid else 0.0)
                        for i, c in enumerate(CATEGORIES)}
                row = {
                    'sample':         sample_label,
                    'tfce_cat':       cat,
                    'tfce_hemi':      h,
                    'tfce_contrast':  contrast,
                    'n_cluster_vox':  n_cluster,
                    'n_selective':    total_valid,
                    'pct_selective':  round(100.0 * total_valid / n_cluster, 1),
                }
                for c in CATEGORIES:
                    row[f'{c}_pct']           = round(pcts[c], 1)
                    row[f'{c}_baseline_pct']  = round(base[h][c], 1)
                    row[f'{c}_enrichment']    = (round(pcts[c] / base[h][c], 2)
                                                 if base[h][c] > 0 else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def print_wta_table(wta_df):
    if not len(wta_df):
        print('  (no TFCE clusters to report)')
        return
    for sample in ('primary', 'sensitivity'):
        sub = wta_df[wta_df['sample'] == sample]
        if not len(sub):
            continue
        print(f'\n--- {sample} ---')
        print(f'{"cat":>6s} {"hemi":>4s} {"contrast":>10s} '
              f'{"n_vox":>6s} {"n_sel":>6s} '
              f'{"face%":>6s} {"hous%":>6s} {"obj%":>6s} {"word%":>6s}  '
              f'enrichment (>1.5 starred)')
        for _, r in sub.iterrows():
            stars = []
            for c in CATEGORIES:
                e = r[f'{c}_enrichment']
                if pd.notna(e) and e > 1.5:
                    stars.append(f'{c}={e:.2f}*')
            star_s = ' '.join(stars)
            print(f'{r["tfce_cat"]:>6s} {r["tfce_hemi"]:>4s} '
                  f'{r["tfce_contrast"]:>10s} '
                  f'{int(r["n_cluster_vox"]):>6d} {int(r["n_selective"]):>6d} '
                  f'{r["face_pct"]:>6.1f} {r["house_pct"]:>6.1f} '
                  f'{r["object_pct"]:>6.1f} {r["word_pct"]:>6.1f}  {star_s}')


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-tfce', action='store_true',
                        help='Skip Test 5 (WTA composition of TFCE clusters)')
    args = parser.parse_args()

    print('=' * 78)
    print('Liu (2025) patient-overlap sensitivity analysis')
    print('=' * 78)
    print(f'Excluded in sensitivity sample: {LIU_OVERLAP_SUBS}')

    df = load_peak_data()

    # --- Build subject lists for both samples ----------------------------------
    ctrl_subs, pt_lh_pri, pt_rh_pri = build_subject_lists(df, exclude_liu=False)
    _,         pt_lh_sen, pt_rh_sen = build_subject_lists(df, exclude_liu=True)
    print(f'\nPrimary    sample: {len(ctrl_subs)} ctrl, '
          f'{len(pt_lh_pri)} LH-intact, {len(pt_rh_pri)} RH-intact OTC')
    print(f'Sensitivity sample: {len(ctrl_subs)} ctrl, '
          f'{len(pt_lh_sen)} LH-intact, {len(pt_rh_sen)} RH-intact OTC')

    # --- Tests 1, 2, 3 ---------------------------------------------------------
    print('\n[1/3] Running distance, sum-sel, distinctiveness perm tests...')
    long_rows = []
    for sample_label, pt_lh, pt_rh in [
        ('primary',     pt_lh_pri, pt_rh_pri),
        ('sensitivity', pt_lh_sen, pt_rh_sen),
    ]:
        long_rows.append(
            run_distance_test(df, ctrl_subs, pt_lh, pt_rh, sample_label))
        long_rows.append(
            run_column_test(df, ctrl_subs, pt_lh, pt_rh, sample_label,
                            'sum_sel', 'sum_selec_norm', transform='log10'))
        long_rows.append(
            run_column_test(df, ctrl_subs, pt_lh, pt_rh, sample_label,
                            'distinctiveness', 'liu_distinctiveness', transform=None))
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df = apply_bh_fdr_within_measure(long_df)
    wide    = pivot_side_by_side(long_df)
    print_perm_tables(wide)
    perm_csv = OUT_DIR / 'perm_tests_primary_vs_sensitivity.csv'
    wide.to_csv(perm_csv, index=False)
    print(f'\nSaved: {perm_csv}')

    # --- Test 5 ----------------------------------------------------------------
    if args.skip_tfce:
        print('\n[2/3] Skipping Test 5 (--skip-tfce).')
        return
    print('\n[2/3] Running Test 5: WTA composition within surviving TFCE clusters...')
    primary_ok = TFCE_DIR_PRIMARY.exists()
    sens_ok    = TFCE_DIR_SENS.exists()
    if not primary_ok:
        print(f'  Primary TFCE dir missing: {TFCE_DIR_PRIMARY}')
    if not sens_ok:
        print(f'  Sensitivity TFCE dir missing: {TFCE_DIR_SENS}')
        print('  → Run `python tfce_votc_contrasts.py --exclude-liu` first, then re-run this.')
    if not (primary_ok or sens_ok):
        return

    wta_parts = []
    if primary_ok:
        wta, peak_z = build_control_wta(df, ctrl_subs, 'primary')
        if wta is not None:
            wta_parts.append(tally_wta_in_clusters(TFCE_DIR_PRIMARY, wta, peak_z, 'primary'))
    if sens_ok:
        wta, peak_z = build_control_wta(df, ctrl_subs, 'sensitivity')
        if wta is not None:
            wta_parts.append(tally_wta_in_clusters(TFCE_DIR_SENS, wta, peak_z, 'sensitivity'))
    if wta_parts:
        wta_df = pd.concat(wta_parts, ignore_index=True)
        print_wta_table(wta_df)
        wta_csv = OUT_DIR / 'tfce_wta_primary_vs_sensitivity.csv'
        wta_df.to_csv(wta_csv, index=False)
        print(f'\nSaved: {wta_csv}')

    print('\n[3/3] Done.')


if __name__ == '__main__':
    main()
