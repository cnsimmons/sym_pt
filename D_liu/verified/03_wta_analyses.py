#!/usr/bin/env python3
"""
wta_analysis.py

Extracts Winner-Take-All (WTA) categorical territory percentages per subject.
Does NOT run statistics. Outputs a clean long-format CSV for downstream stats.

Measures (one CSV, distinguished by `region` + `denominator`):
  - region='otc',     denominator='selective' : % of selective VOTC voxels per category
  - region='otc',     denominator='total'     : % of ALL VOTC voxels per category
                                                 (+ a 'non-selective' category row)
  - region='cluster_*', denominator='selective': % of selective voxels per category
                                                  inside each surviving TFCE cluster

WTA rule: each voxel awarded to the highest-z category; voxel counts as
selective only if max z > WTA_THRESHOLD.

Sessions: ALL post-surgery sessions are extracted (one block of rows per
session), matching the univariate/RSA extractors. The registration anchor is
post_sessions[0]; non-anchor sessions load the ses{anchor}-registered z-maps.
Downstream cross-sectional stats filter to the subject's last session.

Inputs : zstat1*_mni.nii.gz (copes 6-9), VOTC masks + TFCE clusters from the
         TFCE pipeline output dir.
"""

import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from params import (processed_dir, skip_subs, skip_codes,
                           get_sessions, get_post_sessions,
                           is_patient, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────

CATEGORIES = ['face', 'house', 'object', 'word']
COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
WTA_THRESHOLD = 2.326

TFCE_DIR = Path(processed_dir) / 'group_results' / 'tfce_votc_fdr'
OUTPUT_CSV = Path(processed_dir) / 'group_results' / 'wta_percentages.csv'

# Surviving TFCE clusters: (category, hemi, tstat). tstat1=ctrl>pt, tstat2=pt>ctrl.
# face did not survive correction in either hemisphere.
CLUSTERS = [
    ('object', 'l', 1),
    ('house',  'r', 1),
    ('word',   'r', 2),
]

# ── Helper Functions ─────────────────────────────────────────────────────────

def load_zstat(sid, session, cope_num):
    """Load one MNI z-stat. Each session is independently registered to MNI,
    so every session has its own zstat1_mni.nii.gz (there is no
    ses{anchor}-registered MNI file; the anchor naming applies to native
    within-subject space only)."""
    feat = (Path(processed_dir) / sid / f'ses-{session}' / 'derivatives' / 'fsl'
            / 'loc' / 'HighLevel.gfeat' / f'cope{cope_num}.feat' / 'stats')
    zpath = feat / 'zstat1_mni.nii.gz'
    if not zpath.exists():
        return None
    return nib.load(zpath).get_fdata()


def compute_winner(sid, session):
    """Return full-volume winner map (0=non-selective, 1-4=face/house/object/word)
    and the per-voxel max z. Returns (None, None) if any cope is missing."""
    z_maps = []
    for cat in CATEGORIES:
        z = load_zstat(sid, session, COPES[cat])
        if z is None:
            return None, None
        z_maps.append(z)
    z_stack = np.stack(z_maps, axis=-1)            # (X, Y, Z, 4)
    max_z = z_stack.max(axis=-1)
    winner = z_stack.argmax(axis=-1) + 1           # 1=face, 2=house, 3=object, 4=word
    winner[max_z < WTA_THRESHOLD] = 0              # below threshold -> non-selective
    return winner, max_z


def load_cluster_masks():
    """Load surviving TFCE cluster masks keyed by (category, hemi)."""
    cluster_masks = {}
    for cat, hemi, tstat in CLUSTERS:
        p = TFCE_DIR / f'{cat}_{hemi}_pt_vs_ctrl' / f'rand_tfce_corrp_tstat{tstat}.nii.gz'
        if not p.exists():
            print(f'  WARNING: cluster file not found: {p}')
            continue
        cluster_masks[(cat, hemi)] = nib.load(p).get_fdata() > 0.95
    return cluster_masks

# ── Main Execution Pipeline ──────────────────────────────────────────────────

def main():
    print(f'Loading VOTC masks from: {TFCE_DIR}')
    mask_l_path = TFCE_DIR / 'votc_l_mask.nii.gz'
    mask_r_path = TFCE_DIR / 'votc_r_mask.nii.gz'
    if not mask_l_path.exists() or not mask_r_path.exists():
        print('Error: TFCE masks not found. Run the TFCE contrasts script first.')
        sys.exit(1)
    masks = {
        'l': nib.load(mask_l_path).get_fdata() > 0.5,
        'r': nib.load(mask_r_path).get_fdata() > 0.5,
    }
    cluster_masks = load_cluster_masks()

    df_csv = _load_csv()
    all_rows = []

    print('Computing Winner-Take-All territory allocations (all post sessions)...')
    for sc in sorted(df_csv['sub_clean'].unique()):
        if sc in skip_subs:
            continue
        sid = f'sub-{sc}'
        sessions = get_sessions(sc)
        if not sessions or not (Path(processed_dir) / sid).exists():
            continue

        info = get_sub_info(sc, sessions[0])
        group = info.get('group', 'unknown')
        if f'{group}{sc}' in skip_codes or group == 'nonOTC':
            continue

        post = get_post_sessions(sc)
        if not post:
            continue

        pt = is_patient(sc)
        intact_hemi = info.get('intact_hemi', '')

        # Each session is independently registered to MNI, so all post sessions
        # are usable directly (no within-subject anchor needed in MNI space).
        hemis_to_run = ['l', 'r'] if group == 'control' else \
                       [('l' if intact_hemi == 'left' else 'r')]

        for ses_num in post:
            session = f'{ses_num:02d}'
            winner, max_z = compute_winner(sid, session)
            if winner is None:
                print(f'  [{sid} ses-{session}] SKIP: missing z-stats')
                continue

            for hemi in hemis_to_run:
                w_hemi = winner[masks[hemi]]            # 1-D over hemi VOTC voxels
                n_total = w_hemi.size
                n_selective = int((w_hemi > 0).sum())

                base = {
                    'subject_id': sid,
                    'code': f'{group}{sc}',
                    'session': session,
                    'group': group,
                    'status': 'patient' if pt else 'control',
                    'hemi': hemi,
                    'hemi_label': 'intact' if pt else ('left' if hemi == 'l' else 'right'),
                }

                # ── region='otc', denominator='selective' ───────────────────────
                if n_selective > 0:
                    for i, cat in enumerate(CATEGORIES, start=1):
                        cnt = int((w_hemi == i).sum())
                        all_rows.append({**base, 'region': 'otc', 'denominator': 'selective',
                                         'category': cat, 'wta_pct': 100.0 * cnt / n_selective,
                                         'voxel_count': cnt, 'denom_voxels': n_selective})

                # ── region='otc', denominator='total' (incl. non-selective) ──────
                if n_total > 0:
                    for i, cat in enumerate(CATEGORIES, start=1):
                        cnt = int((w_hemi == i).sum())
                        all_rows.append({**base, 'region': 'otc', 'denominator': 'total',
                                         'category': cat, 'wta_pct': 100.0 * cnt / n_total,
                                         'voxel_count': cnt, 'denom_voxels': n_total})
                    ns = int((w_hemi == 0).sum())
                    all_rows.append({**base, 'region': 'otc', 'denominator': 'total',
                                     'category': 'non-selective', 'wta_pct': 100.0 * ns / n_total,
                                     'voxel_count': ns, 'denom_voxels': n_total})

                # ── region='cluster_*', denominator='selective' ──────────────────
                for cat_c, hemi_c, _ in CLUSTERS:
                    if hemi_c != hemi or (cat_c, hemi_c) not in cluster_masks:
                        continue
                    cluster_in_hemi = cluster_masks[(cat_c, hemi_c)][masks[hemi]]
                    w_clust = w_hemi[cluster_in_hemi]
                    n_sel_c = int((w_clust > 0).sum())
                    if n_sel_c == 0:
                        continue
                    region = f'cluster_{cat_c}_{hemi_c}'
                    for i, cat in enumerate(CATEGORIES, start=1):
                        cnt = int((w_clust == i).sum())
                        all_rows.append({**base, 'region': region, 'denominator': 'selective',
                                         'category': cat, 'wta_pct': 100.0 * cnt / n_sel_c,
                                         'voxel_count': cnt, 'denom_voxels': n_sel_c})

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Total rows: {len(df)}')
    print(f'Subjects extracted: {df["subject_id"].nunique()}')
    print(f'Sessions per subject:')
    print(df.groupby("subject_id")["session"].nunique().value_counts().to_string())
    print(f'Regions: {sorted(df["region"].unique())}')


if __name__ == '__main__':
    main()