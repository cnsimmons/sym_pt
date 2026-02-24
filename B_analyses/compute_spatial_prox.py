#!/usr/bin/env python3
"""
compute_spatial_proximity_subrois.py
────────────────────────────────────
Spatial proximity analysis at the SUB-ROI level.

Splits:
  - house → house_PPA (medial/anterior) + house_TOS (posterior/lateral)
  - object → object_LOC (lateral occipital) + object_pF (posterior fusiform)
  - face → face_FFA (fusiform) + face_STS (superior temporal)
  - word → word_VWFA (fusiform) + word_STG (superior temporal)

For each symmetric sub-ROI (house_PPA, house_TOS, object_LOC, object_pF),
computes Euclidean distance to each asymmetric sub-ROI and to the nearest
asymmetric sub-ROI, paired with geometry preservation.

Output:
  {processed_dir}/group_results/geometry/spatial_proximity_subrois.csv

Usage:
  python compute_spatial_proximity_subrois.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

BASE_DIR   = Path(processed_dir)
GEOM_DIR   = BASE_DIR / 'group_results' / 'geometry'
PEAK_DIR   = BASE_DIR / 'group_results' / 'peak_coords'
OUTPUT_DIR = GEOM_DIR

# Sub-ROIs
SYMMETRIC_SUBS  = ['house_PPA', 'house_TOS', 'object_LOC', 'object_pF']
ASYMMETRIC_SUBS = ['face_FFA', 'face_STS', 'word_VWFA', 'word_STG']
# Also include the parent-level asymmetric ROIs as distance targets
ASYMMETRIC_ALL  = ['face', 'face_FFA', 'face_STS', 'word', 'word_VWFA', 'word_STG']

COORD_COLS = ('peak_x_mni', 'peak_y_mni', 'peak_z_mni')


def normalize_sub(val):
    s = str(val).replace('sub-', '').strip()
    try:
        return f'{int(s):03d}'
    except ValueError:
        return s

def normalize_hemi(val):
    s = str(val).strip().lower()
    if s in ['l', 'left', 'lh']: return 'l'
    if s in ['r', 'right', 'rh']: return 'r'
    return s

def normalize_ses(val):
    try:
        return int(str(val).strip())
    except ValueError:
        return val

def euclidean_dist(row1, row2):
    c1 = np.array([float(row1[c]) for c in COORD_COLS])
    c2 = np.array([float(row2[c]) for c in COORD_COLS])
    if np.any(np.isnan(c1)) or np.any(np.isnan(c2)):
        return np.nan
    return float(np.linalg.norm(c1 - c2))


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    geom_df = pd.read_csv(GEOM_DIR / 'geometry_differential.csv')
    peak_df = pd.read_csv(PEAK_DIR / 'peak_coords.csv')
    print(f'Loaded geometry: {len(geom_df)} rows')
    print(f'Loaded peak coords: {len(peak_df)} rows')

    # Normalize
    peak_df['sub_norm']  = peak_df['sub'].apply(normalize_sub)
    peak_df['hemi_norm'] = peak_df['hemi'].apply(normalize_hemi)
    peak_df['ses_norm']  = peak_df['ses'].apply(normalize_ses)

    # ── Get longitudinal OTC patients (from geometry data) ────────────────
    otc_geom = geom_df[geom_df['group'] == 'OTC'].copy()
    otc_geom = otc_geom[~otc_geom['subject'].str.contains('017', na=False)]
    otc_subjects = sorted(otc_geom['subject_id'].unique())

    print(f'\nLongitudinal OTC patients: {len(otc_subjects)}')

    # Check what sub-ROIs are available in geometry data
    geom_cats = sorted(otc_geom['category'].unique())
    print(f'Available geometry categories: {geom_cats}')
    peak_cats = sorted(peak_df['category'].unique())
    print(f'Available peak categories: {peak_cats}')

    # ── Build proximity table ─────────────────────────────────────────────
    rows = []

    for sid in otc_subjects:
        pt_geom = otc_geom[otc_geom['subject_id'] == sid]
        code = pt_geom.iloc[0]['subject']
        surgery_side = pt_geom.iloc[0]['surgery_side']
        hemi_geom = str(pt_geom.iloc[0]['hemi'])
        ses1 = pt_geom.iloc[0]['session_1']

        sub_norm = normalize_sub(sid)
        hemi_norm = normalize_hemi(hemi_geom)
        ses_norm = normalize_ses(ses1)

        # Get peak coordinates
        pt_peaks = peak_df[
            (peak_df['sub_norm'] == sub_norm) &
            (peak_df['hemi_norm'] == hemi_norm) &
            (peak_df['ses_norm'] == ses_norm)
        ]
        if pt_peaks.empty:
            # Fallback: any session
            cand = peak_df[(peak_df['sub_norm'] == sub_norm) &
                           (peak_df['hemi_norm'] == hemi_norm)]
            if not cand.empty:
                pt_peaks = cand[cand['ses_norm'] == cand['ses_norm'].min()]

        if pt_peaks.empty:
            print(f'  {code}: no peaks found, skipping')
            continue

        avail = sorted(pt_peaks['category'].unique())
        print(f'\n  {code} ({surgery_side}): {len(avail)} ROIs available')

        # Collect asymmetric peak rows (both parent and sub-ROI level)
        asym_peak_dict = {}
        for cat in ASYMMETRIC_ALL:
            cat_rows = pt_peaks[pt_peaks['category'] == cat]
            if not cat_rows.empty:
                asym_peak_dict[cat] = cat_rows.iloc[0]

        if not asym_peak_dict:
            print(f'    No asymmetric peaks, skipping')
            continue

        print(f'    Asymmetric ROIs: {list(asym_peak_dict.keys())}')

        # For each symmetric sub-ROI
        for sym_cat in SYMMETRIC_SUBS:
            sym_rows = pt_peaks[pt_peaks['category'] == sym_cat]
            if sym_rows.empty:
                continue
            sym_peak = sym_rows.iloc[0]

            # Distance to each available asymmetric ROI
            dists = {}
            for asym_cat, asym_peak in asym_peak_dict.items():
                d = euclidean_dist(sym_peak, asym_peak)
                if np.isfinite(d):
                    dists[asym_cat] = d

            if not dists:
                continue

            nearest = min(dists, key=dists.get)
            nearest_dist = dists[nearest]

            # Geometry preservation for this sub-ROI
            geom_row = pt_geom[pt_geom['category'] == sym_cat]
            geom_val = geom_row.iloc[0]['geometry_preservation'] if not geom_row.empty else np.nan

            # Also get parent category geometry
            parent_cat = sym_cat.split('_')[0]  # 'house_PPA' → 'house'
            parent_geom_row = pt_geom[pt_geom['category'] == parent_cat]
            parent_geom = parent_geom_row.iloc[0]['geometry_preservation'] if not parent_geom_row.empty else np.nan

            row = {
                'subject':          code,
                'subject_id':       sid,
                'surgery_side':     surgery_side,
                'intact_hemi':      hemi_geom,
                'sym_subroi':       sym_cat,
                'parent_category':  parent_cat,
                'sym_peak_x':       float(sym_peak[COORD_COLS[0]]),
                'sym_peak_y':       float(sym_peak[COORD_COLS[1]]),
                'sym_peak_z':       float(sym_peak[COORD_COLS[2]]),
                'nearest_asym':     nearest,
                'nearest_dist_mm':  nearest_dist,
                'geometry':         geom_val,
                'parent_geometry':  parent_geom,
            }
            # Add distances to each asymmetric target
            for ac in ASYMMETRIC_ALL:
                row[f'dist_to_{ac}'] = dists.get(ac, np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / 'spatial_proximity_subrois.csv'
    df.to_csv(out_file, index=False)
    print(f'\nSaved: {out_file} ({len(df)} rows)')

    if df.empty:
        print('No data — check sub-ROI availability in geometry CSV.')
        return

    # ── Patient-by-patient table ──────────────────────────────────────────
    print('\n' + '='*100)
    print('SUB-ROI PROXIMITY — PATIENT-BY-PATIENT')
    print('='*100)
    print(f'{"Patient":<10} {"Side":<7} {"Sub-ROI":<14} '
          f'{"Nearest Asym":<14} {"Dist(mm)":<10} {"Geom":<8} {"Parent Geom":<12}')
    print('-'*100)

    for _, r in df.iterrows():
        geom_str = f'{r["geometry"]:.3f}' if np.isfinite(r["geometry"]) else '  N/A'
        parent_str = f'{r["parent_geometry"]:.3f}' if np.isfinite(r["parent_geometry"]) else '  N/A'
        print(f'{r["subject"]:<10} {r["surgery_side"]:<7} {r["sym_subroi"]:<14} '
              f'{r["nearest_asym"]:<14} {r["nearest_dist_mm"]:>8.1f}  '
              f'{geom_str:>8}  {parent_str:>8}')

    # ── Summary by sub-ROI ────────────────────────────────────────────────
    print('\n' + '='*100)
    print('SUMMARY BY SYMMETRIC SUB-ROI')
    print('='*100)

    for cat in SYMMETRIC_SUBS:
        sub = df[df['sym_subroi'] == cat]
        if sub.empty:
            continue
        geom_vals = sub['geometry'].dropna()
        dist_vals = sub['nearest_dist_mm'].dropna()
        print(f'\n  {cat}:')
        if len(dist_vals) > 0:
            print(f'    Nearest asym dist: M={dist_vals.mean():.1f}mm '
                  f'(range: {dist_vals.min():.1f}-{dist_vals.max():.1f})')
        if len(geom_vals) > 0:
            print(f'    Geometry:          M={geom_vals.mean():.3f} '
                  f'(range: {geom_vals.min():.3f}-{geom_vals.max():.3f})')

    # ── Key test: PPA vs TOS ──────────────────────────────────────────────
    print('\n' + '='*100)
    print('KEY COMPARISON: house_PPA vs house_TOS (within each patient)')
    print('='*100)

    ppa_closer_count = 0
    ppa_worse_count = 0
    concordant = 0
    total = 0

    for sid in df['subject_id'].unique():
        pt = df[df['subject_id'] == sid]
        ppa = pt[pt['sym_subroi'] == 'house_PPA']
        tos = pt[pt['sym_subroi'] == 'house_TOS']
        if ppa.empty or tos.empty:
            continue

        total += 1
        p, t = ppa.iloc[0], tos.iloc[0]

        ppa_geom = p['geometry'] if np.isfinite(p['geometry']) else np.nan
        tos_geom = t['geometry'] if np.isfinite(t['geometry']) else np.nan

        if not (np.isfinite(ppa_geom) and np.isfinite(tos_geom)):
            print(f'  {p["subject"]}: PPA dist={p["nearest_dist_mm"]:.1f}mm, '
                  f'TOS dist={t["nearest_dist_mm"]:.1f}mm | '
                  f'PPA geom={ppa_geom}, TOS geom={tos_geom} (incomplete)')
            continue

        ppa_closer = p['nearest_dist_mm'] < t['nearest_dist_mm']
        ppa_worse  = ppa_geom < tos_geom

        if ppa_closer: ppa_closer_count += 1
        if ppa_worse:  ppa_worse_count += 1
        if ppa_closer == ppa_worse: concordant += 1

        print(f'  {p["subject"]}: PPA {"CLOSER" if ppa_closer else "farther"} '
              f'({p["nearest_dist_mm"]:.1f} vs {t["nearest_dist_mm"]:.1f}mm), '
              f'PPA {"WORSE" if ppa_worse else "better"} geom '
              f'({ppa_geom:.3f} vs {tos_geom:.3f})')

    if total > 0:
        print(f'\n  PPA closer to asym: {ppa_closer_count}/{total}')
        print(f'  PPA worse geometry: {ppa_worse_count}/{total}')
        print(f'  Concordance:        {concordant}/{total}')

    # ── Also: LOC vs pF ──────────────────────────────────────────────────
    print('\n' + '='*100)
    print('SECONDARY: object_LOC vs object_pF (within each patient)')
    print('='*100)

    loc_closer_count = 0
    loc_worse_count = 0
    concordant2 = 0
    total2 = 0

    for sid in df['subject_id'].unique():
        pt = df[df['subject_id'] == sid]
        loc = pt[pt['sym_subroi'] == 'object_LOC']
        pf  = pt[pt['sym_subroi'] == 'object_pF']
        if loc.empty or pf.empty:
            continue

        total2 += 1
        l, f = loc.iloc[0], pf.iloc[0]

        loc_geom = l['geometry'] if np.isfinite(l['geometry']) else np.nan
        pf_geom  = f['geometry'] if np.isfinite(f['geometry']) else np.nan

        if not (np.isfinite(loc_geom) and np.isfinite(pf_geom)):
            print(f'  {l["subject"]}: incomplete data')
            continue

        loc_closer = l['nearest_dist_mm'] < f['nearest_dist_mm']
        loc_worse  = loc_geom < pf_geom

        if loc_closer: loc_closer_count += 1
        if loc_worse:  loc_worse_count += 1
        if loc_closer == loc_worse: concordant2 += 1

        print(f'  {l["subject"]}: LOC {"CLOSER" if loc_closer else "farther"} '
              f'({l["nearest_dist_mm"]:.1f} vs {f["nearest_dist_mm"]:.1f}mm), '
              f'LOC {"WORSE" if loc_worse else "better"} geom '
              f'({loc_geom:.3f} vs {pf_geom:.3f})')

    if total2 > 0:
        print(f'\n  LOC closer: {loc_closer_count}/{total2}')
        print(f'  LOC worse geom: {loc_worse_count}/{total2}')
        print(f'  Concordance: {concordant2}/{total2}')

    # ── Overall: proximity vs geometry across ALL symmetric sub-ROIs ──────
    print('\n' + '='*100)
    print('ALL SYMMETRIC SUB-ROIS: Proximity vs Geometry')
    print('='*100)
    valid = df.dropna(subset=['nearest_dist_mm', 'geometry'])
    if len(valid) >= 3:
        from scipy.stats import spearmanr, pearsonr
        rho_s, p_s = spearmanr(valid['nearest_dist_mm'], valid['geometry'])
        rho_p, p_p = pearsonr(valid['nearest_dist_mm'], valid['geometry'])
        print(f'  n = {len(valid)} sub-ROI observations')
        print(f'  Spearman: rho={rho_s:.3f}, p={p_s:.4f}')
        print(f'  Pearson:  r={rho_p:.3f}, p={p_p:.4f}')
        print(f'  (Positive r = farther from asym → better geometry = proximity predicts disruption)')
    else:
        print(f'  Insufficient data (n={len(valid)})')


if __name__ == '__main__':
    main()