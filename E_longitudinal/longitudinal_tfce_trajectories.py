#!/usr/bin/env python3
"""
longitudinal_tfce_trajectories.py

Within-subject longitudinal follow-up to the cross-sectional TFCE analysis.
For each subject with >=2 sessions, two complementary measures inside each
surviving cross-sectional TFCE cluster mask:

A - PERSISTENCE (per sub x ses x cluster)
    Three summary stats on the cluster's own category-vs-others z-stat:
        mean_z              - selectivity strength
        sum_z               - territory-loss / gain proxy
        n_voxels_above      - count of voxels with z > WTA_THRESH

B - PREFERENCE SHIFTS (Liu McNemar-style; per sub x ses-pair x cluster)
    Voxel-wise WTA across the 4 categories at both sessions in the pair.
    A voxel enters the transition table only if its peak z > WTA_THRESH at
    BOTH sessions. Output is a 4x4 (from_cat, to_cat) count matrix in long
    format; McNemar tests are done in the notebook from these counts.

CS TFCE survivors (thresholded at corrp > 0.95):
    object_L : cope 8, hemi l, ctrl>pt (tstat1)
    house_R  : cope 7, hemi r, ctrl>pt (tstat1)
    word_R   : cope 9, hemi r, pt>ctrl (tstat2)
    face_L   : cope 6, hemi l, pt>ctrl (tstat2)

Outputs in processed_dir/group_results/longitudinal/tfce/:
    tfce_persistence[_thr###].csv  - long, (sub, ses, cluster) rows
    tfce_transitions[_thr###].csv  - long, (sub, ses_a, ses_b, cluster, from, to) rows
    tfce_run_log[_thr###].txt
    (no suffix when --wta-thresh is the default 2.326)

Usage
-----
    python longitudinal_tfce_trajectories.py                      # default 2.326
    python longitudinal_tfce_trajectories.py --wta-thresh 1.96    # less strict
    python longitudinal_tfce_trajectories.py --skip-b             # A only
    python longitudinal_tfce_trajectories.py --skip-a             # B only
    python longitudinal_tfce_trajectories.py --sub sub-021        # one subject
"""

import sys
import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, get_sessions, get_sub_info, _load_csv


# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_WTA_THRESH = 2.326
CORRP_THRESH       = 0.95

CAT_COPES  = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
CATEGORIES = ['face', 'house', 'object', 'word']

CLUSTER_DEFS = {
    'object_L': {'category': 'object', 'hemi': 'l', 'direction': 'ctrl>pt', 'tstat': 1},
    'house_R':  {'category': 'house',  'hemi': 'r', 'direction': 'ctrl>pt', 'tstat': 1},
    'word_R':   {'category': 'word',   'hemi': 'r', 'direction': 'pt>ctrl', 'tstat': 2},
    'face_L':   {'category': 'face',   'hemi': 'l', 'direction': 'pt>ctrl', 'tstat': 2},
}

PRE_SURGERY_SESSIONS = {}
'''
    'sub-021': {'01'}, 'sub-045': {'01'}, 'sub-047': {'01'}, 'sub-049': {'01'},
    'sub-070': {'01'}, 'sub-073': {'01'}, 'sub-081': {'01'}, 'sub-086': {'01'},
    'sub-108': {'02'},
}
'''

EXTRA_SKIP = {'sub-017', 'control083', 'control085'}

TFCE_DIR = Path(processed_dir) / 'group_results' / 'tfce_votc'
OUT_DIR  = Path(processed_dir) / 'group_results' / 'longitudinal' / 'tfce'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Cluster masks ────────────────────────────────────────────────────────────
def load_cluster_masks():
    masks = {}
    print('Cluster masks (corrp > 0.95):')
    for name, d in CLUSTER_DEFS.items():
        test_dir = TFCE_DIR / f'{d["category"]}_{d["hemi"]}_pt_vs_ctrl'
        corrp_p = test_dir / f'rand_tfce_corrp_tstat{d["tstat"]}.nii.gz'
        if not corrp_p.exists():
            print(f'  ERROR: missing {corrp_p}')
            return None, None
        img = nib.load(str(corrp_p))
        corrp = img.get_fdata()
        mask  = corrp > CORRP_THRESH
        masks[name] = mask
        print(f'  {name:9s}  {int(mask.sum()):>5,} voxels   '
              f'(cope {CAT_COPES[d["category"]]}, {d["direction"]}, tstat{d["tstat"]})')
    return masks, img.affine


# ── Subject selection ────────────────────────────────────────────────────────
def select_longitudinal_subjects(restrict_to=None):
    df = _load_csv()
    subs = {}
    for sc in sorted(df['sub_clean'].unique()):
        sid = f'sub-{sc}'
        if restrict_to and sid != restrict_to:
            continue
        if sid in EXTRA_SKIP:
            continue
        sessions = get_sessions(sc)
        if len(sessions) < 2:
            continue
        info = get_sub_info(sc, sessions[0])
        group = info.get('group', 'unknown')
        if group not in ('OTC', 'control'):
            continue
        post_sessions = [f'{s:02d}' for s in sessions
                         if f'{s:02d}' not in PRE_SURGERY_SESSIONS.get(sid, set())]
        if not post_sessions:
            continue
        intact = info.get('intact_hemi', '')
        subs[sid] = {
            'sessions':    [f'{s:02d}' for s in sessions],
            'anchor':      post_sessions[0],
            'group':       group,
            'intact_hemi': intact if group == 'OTC' else 'both',
            'code':        info.get('code', ''),
        }
    return subs


# ── zstat path resolver ──────────────────────────────────────────────────────
def find_zstat_path(sid, ses, anchor, cope):
    base = (Path(processed_dir) / sid / f'ses-{ses}' / 'derivatives' / 'fsl'
            / 'loc' / 'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats')
    if ses == anchor:
        p = base / 'zstat1_mni.nii.gz'
        return p if p.exists() else None
    p_anchor = base / f'zstat1_ses{anchor}_mni.nii.gz'
    if p_anchor.exists():
        return p_anchor
    p_self = base / 'zstat1_mni.nii.gz'
    return p_self if p_self.exists() else None


def load_session_zstats(sid, ses, anchor):
    vols = {}
    for cat, cope in CAT_COPES.items():
        p = find_zstat_path(sid, ses, anchor, cope)
        if p is None:
            return None, cat
        vols[cat] = nib.load(str(p)).get_fdata()
    return vols, None


# ── Option A: persistence stats ─────────────────────────────────────────────
def compute_persistence(vols, cluster_masks, wta_thresh):
    out = []
    for name, d in CLUSTER_DEFS.items():
        z = vols[d['category']]
        mask = cluster_masks[name]
        vals = z[mask]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out.append({'cluster': name, 'mean_z': np.nan, 'sum_z': np.nan,
                        'n_voxels_above': 0})
            continue
        out.append({
            'cluster':        name,
            'mean_z':         float(vals.mean()),
            'sum_z':          float(vals.sum()),
            'n_voxels_above': int((vals > wta_thresh).sum()),
        })
    return out


# ── Option B: WTA + transitions ─────────────────────────────────────────────
def compute_wta(vols):
    stack = np.stack([vols[c] for c in CATEGORIES], axis=0)
    return np.argmax(stack, axis=0), np.max(stack, axis=0)


def transitions_in_cluster(wta_a, peakz_a, wta_b, peakz_b, mask, wta_thresh):
    in_mask = mask & (peakz_a > wta_thresh) & (peakz_b > wta_thresh)
    if not in_mask.any():
        return {}
    wa = wta_a[in_mask]
    wb = wta_b[in_mask]
    counts = {}
    for i in range(4):
        for j in range(4):
            n = int(((wa == i) & (wb == j)).sum())
            if n > 0:
                counts[(CATEGORIES[i], CATEGORIES[j])] = n
    return counts


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wta-thresh', type=float, default=DEFAULT_WTA_THRESH,
                        help=f'WTA threshold (default {DEFAULT_WTA_THRESH} ≈ p<.01; '
                             f'try 1.96 ≈ p<.05, 1.645 ≈ p<.10)')
    parser.add_argument('--skip-a', action='store_true', help='Skip persistence (A)')
    parser.add_argument('--skip-b', action='store_true', help='Skip transitions (B)')
    parser.add_argument('--sub', default=None, help='Restrict to single subject')
    args = parser.parse_args()

    wta_thresh = args.wta_thresh
    suffix = '' if wta_thresh == DEFAULT_WTA_THRESH else f'_thr{int(round(wta_thresh*100)):03d}'

    t0 = time.time()
    log_lines = []

    print('=' * 72)
    print('Longitudinal TFCE follow-up: persistence (A) + preference shifts (B)')
    print(f'WTA threshold: z > {wta_thresh}   Output suffix: "{suffix}"')
    print('=' * 72)

    print('\n[1/4] Loading cluster masks...')
    cluster_masks, _ = load_cluster_masks()
    if cluster_masks is None:
        print('ABORT: cluster masks missing.')
        sys.exit(1)

    print('\n[2/4] Selecting longitudinal subjects...')
    subs = select_longitudinal_subjects(restrict_to=args.sub)
    n_otc  = sum(1 for v in subs.values() if v['group'] == 'OTC')
    n_ctrl = sum(1 for v in subs.values() if v['group'] == 'control')
    print(f'  OTC: {n_otc}, Controls: {n_ctrl}, total: {len(subs)}')
    for sid in sorted(subs):
        m = subs[sid]
        print(f'  {sid:10s} ({m["group"]:7s}, intact={m["intact_hemi"]:5s}) '
              f'sessions={m["sessions"]} anchor=ses-{m["anchor"]}')

    print('\n[3/4] Loading zstats, extracting per-session measures...')
    persist_rows = []
    wta_cache = {}

    for sid in sorted(subs):
        m = subs[sid]
        for ses in m['sessions']:
            vols, missing = load_session_zstats(sid, ses, m['anchor'])
            if vols is None:
                msg = f'  SKIP {sid}/ses-{ses}: missing cope {missing} zstat'
                print(msg); log_lines.append(msg)
                continue
            log_lines.append(f'OK   {sid}/ses-{ses}: anchor=ses-{m["anchor"]}')

            if not args.skip_a:
                for row in compute_persistence(vols, cluster_masks, wta_thresh):
                    row.update({
                        'subject_id':  sid,
                        'session':     int(ses),
                        'code':        m['code'],
                        'group':       m['group'],
                        'intact_hemi': m['intact_hemi'],
                    })
                    persist_rows.append(row)

            if not args.skip_b:
                wta_cache[(sid, ses)] = compute_wta(vols)

            print(f'  {sid}/ses-{ses}: persistence done, '
                  f'{time.time()-t0:.0f}s elapsed')

    if not args.skip_a and persist_rows:
        persist_df = pd.DataFrame(persist_rows)
        persist_df = persist_df[['subject_id', 'session', 'code', 'group',
                                  'intact_hemi', 'cluster', 'mean_z', 'sum_z',
                                  'n_voxels_above']]
        out_a = OUT_DIR / f'tfce_persistence{suffix}.csv'
        persist_df.to_csv(out_a, index=False)
        print(f'\n  Saved A: {out_a}  ({len(persist_df):,} rows)')

    if not args.skip_b:
        print('\n[4/4] Computing voxelwise transitions per session-pair...')
        trans_rows = []
        for sid in sorted(subs):
            m = subs[sid]
            sessions_with_wta = [s for s in m['sessions'] if (sid, s) in wta_cache]
            if len(sessions_with_wta) < 2:
                print(f'  SKIP {sid}: <2 sessions with usable zstats')
                continue
            for ses_a, ses_b in combinations(sessions_with_wta, 2):
                wta_a, peakz_a = wta_cache[(sid, ses_a)]
                wta_b, peakz_b = wta_cache[(sid, ses_b)]
                for cluster, mask in cluster_masks.items():
                    counts = transitions_in_cluster(wta_a, peakz_a,
                                                    wta_b, peakz_b, mask, wta_thresh)
                    for (fc, tc), n in counts.items():
                        trans_rows.append({
                            'subject_id':  sid,
                            'ses_a':       int(ses_a),
                            'ses_b':       int(ses_b),
                            'code':        m['code'],
                            'group':       m['group'],
                            'intact_hemi': m['intact_hemi'],
                            'cluster':     cluster,
                            'from_cat':    fc,
                            'to_cat':      tc,
                            'count':       n,
                        })
                print(f'  {sid}: ses-{ses_a} -> ses-{ses_b} done')
        if trans_rows:
            trans_df = pd.DataFrame(trans_rows)
            trans_df = trans_df[['subject_id', 'ses_a', 'ses_b', 'code', 'group',
                                  'intact_hemi', 'cluster', 'from_cat', 'to_cat',
                                  'count']]
            out_b = OUT_DIR / f'tfce_transitions{suffix}.csv'
            trans_df.to_csv(out_b, index=False)
            print(f'\n  Saved B: {out_b}  ({len(trans_df):,} rows)')

    log_path = OUT_DIR / f'tfce_run_log{suffix}.txt'
    with log_path.open('w') as f:
        f.write('\n'.join(log_lines))
    print(f'\nRun log: {log_path}')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
