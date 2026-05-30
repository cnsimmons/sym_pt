#!/usr/bin/env python3
"""
longitudinal_trajectories.py

Per-session metrics for every OTC patient and control with >1 session.
Tracks three measures over time:
  1. Peak coordinates (peak_x/y/z_mni) per (sub × ses × cat × hemi)
  2. Sum-selectivity (log10 sum_selec_norm) per (sub × ses × cat × hemi)
  3. WTA territory % per (sub × ses × cat × hemi) — % of hemi VOTC voxels
     where this category wins WTA (z > WTA_THRESH)

Output: long CSV in `group_results/longitudinal/longitudinal_metrics.csv`
Format: one row per (sub × ses × measure × cat × hemi × axis_if_applicable)

Subjects in scope:
  - OTC longitudinal (6): sub-004, sub-008, sub-010, sub-021, sub-079, sub-108
  - Controls longitudinal (8): from peak_coords_mni.csv

WTA territory uses pre-built hemi VOTC masks from `group_results/tfce_votc/`.
Peak/sum-sel come from `peak_coords_mni.csv`. Pre/post metadata from
`sub_info4_30.csv`.

Usage
-----
  python longitudinal_trajectories.py              # all three measures
  python longitudinal_trajectories.py --skip-wta   # CSV-only (fast)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

# ── Configuration ────────────────────────────────────────────────────────────
WTA_THRESH = 2.326   # z>2.326 = p<.01 one-tailed (matches sensitivity_analysis)
CAT_COPES  = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
CATEGORIES = list(CAT_COPES.keys())

PEAK_CSV       = Path(processed_dir) / 'group_results' / 'peak_coords' / 'peak_coords_mni.csv'
SUB_INFO_CSV   = Path('/user_data/csimmon2/git_repos/sym_pt/sub_info.csv')
VOTC_MASK_DIR  = Path(processed_dir) / 'group_results' / 'tfce_votc'
OUT_DIR        = Path(processed_dir) / 'group_results' / 'longitudinal'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading & subject selection ─────────────────────────────────────────
def load_data():
    peak = pd.read_csv(PEAK_CSV)
    peak = peak.drop_duplicates(subset=['subject_id', 'session', 'category', 'hemi']).copy()
    info = pd.read_csv(SUB_INFO_CSV)
    info['session'] = info['ses'].str.replace('ses-', '').astype(int)
    info['subject_id'] = info['sub']
    return peak, info[['subject_id', 'session', 'pre_post', 'age']]


def select_longitudinal_subs(peak):
    """Return subjects in OTC or control with >1 session."""
    keep = peak[peak['group'].isin(['OTC', 'control'])]
    n_ses = keep.groupby('subject_id')['session'].nunique()
    return sorted(n_ses[n_ses > 1].index.tolist())


# ── Module 1: Peak coordinates per session ──────────────────────────────────
def compute_peak_metrics(peak, subs):
    rows = []
    for sub in subs:
        sdf = peak[peak['subject_id'] == sub]
        for ses in sorted(sdf['session'].unique()):
            sesdf = sdf[sdf['session'] == ses]
            for _, r in sesdf.iterrows():
                for axis in ('x', 'y', 'z'):
                    val = r[f'peak_{axis}_mni']
                    if pd.notna(val):
                        rows.append({
                            'subject_id': sub, 'session': ses,
                            'group': r['group'], 'intact_hemi': r['intact_hemi'],
                            'surgery_side': r['surgery_side'],
                            'measure': f'peak_{axis}_mni',
                            'parcel': r['category'], 'hemi': r['hemi'],
                            'value': float(val),
                        })
    return pd.DataFrame(rows)


# ── Module 2: Sum-selectivity per session ───────────────────────────────────
def compute_sumsel_metrics(peak, subs):
    rows = []
    for sub in subs:
        sdf = peak[peak['subject_id'] == sub]
        for ses in sorted(sdf['session'].unique()):
            sesdf = sdf[sdf['session'] == ses]
            for _, r in sesdf.iterrows():
                val = r['sum_selec_norm']
                if pd.notna(val) and val > 0:
                    rows.append({
                        'subject_id': sub, 'session': ses,
                        'group': r['group'], 'intact_hemi': r['intact_hemi'],
                        'surgery_side': r['surgery_side'],
                        'measure': 'log10_sum_sel',
                        'parcel': r['category'], 'hemi': r['hemi'],
                        'value': float(np.log10(val)),
                    })
    return pd.DataFrame(rows)


# ── Module 2b: Distinctiveness per session ──────────────────────────────────
def compute_distinctiveness_metrics(peak, subs):
    rows = []
    for sub in subs:
        sdf = peak[peak['subject_id'] == sub]
        for ses in sorted(sdf['session'].unique()):
            sesdf = sdf[sdf['session'] == ses]
            for _, r in sesdf.iterrows():
                val = r['liu_distinctiveness']
                if pd.notna(val):
                    rows.append({
                        'subject_id': sub, 'session': ses,
                        'group': r['group'], 'intact_hemi': r['intact_hemi'],
                        'surgery_side': r['surgery_side'],
                        'measure': 'distinctiveness',
                        'parcel': r['category'], 'hemi': r['hemi'],
                        'value': float(val),
                    })
    return pd.DataFrame(rows)


# ── Module 3: WTA territory % per session ───────────────────────────────────
def load_hemi_masks():
    """Load the LH and RH VOTC masks built during the TFCE pipeline."""
    masks = {}
    for hemi in ('l', 'r'):
        path = VOTC_MASK_DIR / f'votc_{hemi}_mask.nii.gz'
        if not path.exists():
            print(f'WARNING: VOTC mask missing: {path}')
            return None
        masks[hemi] = nib.load(str(path)).get_fdata().astype(bool)
    return masks


def get_zstat_path(sub, ses, cope):
    """Path to MNI-registered zstat for a given (sub, ses, cope)."""
    ses_str = f'{int(ses):02d}'
    direct = (Path(processed_dir) / sub / f'ses-{ses_str}' / 'derivatives' / 'fsl'
              / 'loc' / 'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats'
              / 'zstat1_mni.nii.gz')
    if direct.exists():
        return direct
    # Non-anchor sessions are registered back to the anchor ses first
    base = (Path(processed_dir) / sub / f'ses-{ses_str}' / 'derivatives' / 'fsl'
            / 'loc' / 'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats')
    if base.exists():
        candidates = list(base.glob('zstat1_ses*_mni.nii.gz'))
        if candidates:
            return candidates[0]
    return None


def compute_wta_per_session(sub, ses, masks):
    """For one (sub, ses): load all 4 category zstats, compute WTA per voxel,
    return dict {hemi: {category: pct}}."""
    zmaps = {}
    for cat, cope in CAT_COPES.items():
        zp = get_zstat_path(sub, ses, cope)
        if zp is None:
            return None
        zmaps[cat] = nib.load(str(zp)).get_fdata()
    stack = np.stack([zmaps[c] for c in CATEGORIES], axis=0)  # 4 × X × Y × Z
    wta_idx = np.argmax(stack, axis=0)
    peak_z  = np.max(stack, axis=0)
    selective = peak_z >= WTA_THRESH
    out = {'l': {}, 'r': {}}
    for hemi in ('l', 'r'):
        in_hemi = selective & masks[hemi]
        total = int(in_hemi.sum())
        if total == 0:
            for c in CATEGORIES:
                out[hemi][c] = np.nan
            continue
        for i, c in enumerate(CATEGORIES):
            out[hemi][c] = 100.0 * float((wta_idx[in_hemi] == i).sum()) / total
    return out


def compute_wta_metrics(peak, subs):
    """Loop over (sub, ses), compute WTA territory %, return long DataFrame."""
    masks = load_hemi_masks()
    if masks is None:
        return pd.DataFrame()
    rows = []
    for sub in subs:
        sdf = peak[peak['subject_id'] == sub]
        sub_meta = sdf.iloc[0]
        for ses in sorted(sdf['session'].unique()):
            print(f'  WTA: {sub} ses-{int(ses):02d}...', flush=True)
            result = compute_wta_per_session(sub, ses, masks)
            if result is None:
                print(f'    skipped (missing zstats)')
                continue
            for hemi in ('l', 'r'):
                for cat in CATEGORIES:
                    rows.append({
                        'subject_id':   sub,
                        'session':      ses,
                        'group':        sub_meta['group'],
                        'intact_hemi':  sub_meta['intact_hemi'],
                        'surgery_side': sub_meta['surgery_side'],
                        'measure':      'wta_territory_pct',
                        'parcel':       cat,
                        'hemi':         hemi,
                        'value':        result[hemi][cat],
                    })
    return pd.DataFrame(rows)


# ── Assembly + output ───────────────────────────────────────────────────────
def attach_pre_post(df_long, info):
    return df_long.merge(info, on=['subject_id', 'session'], how='left')


def print_summary(df_long):
    print('\n' + '=' * 70)
    print('LONGITUDINAL SUMMARY')
    print('=' * 70)
    n_subs = df_long['subject_id'].nunique()
    n_rows = len(df_long)
    print(f'Total rows: {n_rows:,}')
    print(f'Subjects: {n_subs}')
    print(f'Measures: {sorted(df_long["measure"].unique())}')
    print()
    print('Sessions per subject:')
    counts = df_long.groupby(['subject_id', 'group'])['session'].nunique() \
                    .reset_index().rename(columns={'session': 'n_sessions'})
    for grp in ('OTC', 'control'):
        sub = counts[counts['group'] == grp].sort_values('subject_id')
        if not len(sub):
            continue
        print(f'  {grp}:')
        for _, r in sub.iterrows():
            pp = df_long[(df_long['subject_id'] == r['subject_id'])]['pre_post'].unique()
            pp_str = ', '.join(sorted([str(p) for p in pp if pd.notna(p)]))
            print(f'    {r["subject_id"]}: {r["n_sessions"]} sessions ({pp_str})')


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-wta', action='store_true',
                        help='Skip WTA territory (slow). CSV-only measures.')
    args = parser.parse_args()

    print('=' * 70)
    print('Longitudinal trajectories: peak coords + sum-sel + WTA territory')
    print('=' * 70)

    peak, info = load_data()
    subs = select_longitudinal_subs(peak)
    print(f'\nLongitudinal subjects: {len(subs)}')
    n_otc  = sum(peak[peak['subject_id'] == s]['group'].iloc[0] == 'OTC' for s in subs)
    n_ctrl = sum(peak[peak['subject_id'] == s]['group'].iloc[0] == 'control' for s in subs)
    print(f'  OTC: {n_otc}, Controls: {n_ctrl}')

    parts = []
    print('\n[1/3] Peak coordinates...')
    parts.append(compute_peak_metrics(peak, subs))
    print('[2/3] Sum-selectivity...')
    parts.append(compute_sumsel_metrics(peak, subs))
    print('[2b/3] Distinctiveness...')
    parts.append(compute_distinctiveness_metrics(peak, subs))
    if args.skip_wta:
        print('[3/3] Skipping WTA territory (--skip-wta).')
    else:
        print('[3/3] WTA territory (loads zstats per session)...')
        parts.append(compute_wta_metrics(peak, subs))

    df_long = pd.concat(parts, ignore_index=True)
    df_long = attach_pre_post(df_long, info)

    out_csv = OUT_DIR / 'longitudinal_metrics.csv'
    df_long.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')

    print_summary(df_long)


if __name__ == '__main__':
    main()