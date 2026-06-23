#!/usr/bin/env python3
"""
combat_07_harmonize_wta.py — recompute Winner-Take-All from HARMONIZED z-maps.

Principled WTA harmonization (option B): instead of harmonizing the post-argmax
category counts (which breaks the compositional logic), this recomputes the winner
map FROM the harmonized per-category z-values that combat_02 already produced.

The selectivity threshold cannot be the fixed z=2.326 after ComBat (harmonized
values are no longer calibrated z). Instead the threshold is PERCENTILE-MATCHED
PER SUBJECT: for each subject x hemi, the fraction of VOTC voxels that pass
z>2.326 in the RAW data is computed, and the same top fraction is taken as
'selective' in the harmonized data (per-subject (1-frac) quantile of harmonized
max-z). This pins the MEANING of the threshold (how much cortex counts as
selective) rather than an off-scale number.

Inputs:
  combat_inputs/features_{hemi}.npz       raw per-category z  (for the frac)
  combat_harmonized/harmonized_{hemi}.npz harmonized per-category z (X_face..X_word)
  group_results/tfce_votc_harmonized/{cat}_{hemi}_pt_vs_ctrl/...  (cluster rows)

Output:
  group_results/wta_percentages_harmonized.csv   (region='otc'/'selective' + cluster_* rows)

Schema matches wta_percentages.csv so 05_stats_harmony can read it (--wta).
denominator='total' (whole-volume non-selective) rows are NOT reproduced here
(harmonized matrices are VOTC-only); stats use the 'selective' rows.
"""
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

GIT        = Path('/user_data/csimmon2/git_repos/sym_pt')
PROC       = Path('/user_data/csimmon2/sym_pt')
RAW_DIR    = GIT / 'F_harmonization' / 'combat_inputs'
HARM_DIR   = GIT / 'F_harmonization' / 'combat_harmonized'
COVARS     = GIT / 'F_harmonization' / 'combat_inputs' / 'covars.csv'   # written by combat_01
TFCE_HARM  = PROC / 'group_results' / 'tfce_votc_harmonized'
OUT_CSV    = PROC / 'group_results' / 'wta_percentages_harmonized.csv'

CATEGORIES = ['face', 'house', 'object', 'word']   # argmax order: 1=face..4=word
RAW_THRESH = 2.326
# surviving harmonized-corrected clusters (cat, hemi, tstat); matches 03_wta CLUSTERS
CLUSTERS   = [('object', 'l', 1), ('house', 'r', 1), ('word', 'r', 2)]


def load_npz(path):
    f = np.load(path, allow_pickle=True)
    return {k: f[k] for k in f.files}


def aligned_stack(d, prefix):
    """Return (subjects, [n_sub,n_vox,4]).
    Harmonized npz stores a single 'subs' array (all categories same order);
    raw npz stores per-category 'subs_*'. Handle both."""
    if 'subs' in d:                                  # harmonized: one shared order
        common = list(d['subs'])
        stacks = [np.asarray(d[f'{prefix}{c}'], dtype=np.float32) for c in CATEGORIES]
        return common, np.stack(stacks, axis=-1)
    # raw: intersect per-category lists
    sub_lists = {c: list(d[f'subs_{c}']) for c in CATEGORIES}
    common = [s for s in sub_lists['face']
              if all(s in sub_lists[c] for c in CATEGORIES)]
    idx = {c: {s: i for i, s in enumerate(sub_lists[c])} for c in CATEGORIES}
    stacks = []
    for c in CATEGORIES:
        rows = [d[f'{prefix}{c}'][idx[c][s]] for s in common]
        stacks.append(np.array(rows, dtype=np.float32))   # [n_sub, n_vox]
    return common, np.stack(stacks, axis=-1)               # [n_sub, n_vox, 4]


def main():
    cov = pd.read_csv(COVARS).set_index('subject_id')
    rows = []

    # ---- precompute cluster masks restricted to each hemi VOTC (for cluster rows) ----
    cluster_idx = {}   # (cat,hemi) -> boolean over that hemi's VOTC voxels
    for hemi in ['l', 'r']:
        raw = load_npz(RAW_DIR / f'features_{hemi}.npz')
        votc = raw['mask'].astype(bool)
        for cat, h, tstat in CLUSTERS:
            if h != hemi:
                continue
            p = TFCE_HARM / f'{cat}_{hemi}_pt_vs_ctrl' / f'rand_tfce_corrp_tstat{tstat}.nii.gz'
            if not p.exists():
                print(f"  WARNING: cluster file missing {p}")
                continue
            cl = nib.load(str(p)).get_fdata() > 0.95
            cluster_idx[(cat, hemi)] = cl[votc]            # 1-D over hemi VOTC voxels

    for hemi in ['l', 'r']:
        raw  = load_npz(RAW_DIR  / f'features_{hemi}.npz')
        harm = load_npz(HARM_DIR / f'harmonized_{hemi}.npz')

        subs_raw,  Zraw  = aligned_stack(raw,  'X_')        # raw z   [n,vox,4]
        subs_harm, Zharm = aligned_stack(harm, 'X_')        # harmonized z
        # use subjects present in BOTH
        common = [s for s in subs_harm if s in subs_raw]
        ri = {s: i for i, s in enumerate(subs_raw)}
        hi = {s: i for i, s in enumerate(subs_harm)}
        print(f"\n[{hemi.upper()}H] {len(common)} subjects, {Zharm.shape[1]} VOTC voxels")

        for sid in common:
            zr = Zraw[ri[sid]]        # [vox,4] raw
            zh = Zharm[hi[sid]]       # [vox,4] harmonized
            # raw selective fraction (the meaning we preserve)
            frac = float((zr.max(axis=1) > RAW_THRESH).mean())
            if frac <= 0:
                continue
            # per-subject harmonized threshold = (1-frac) quantile of harmonized max-z
            maxh = zh.max(axis=1)
            thr  = np.quantile(maxh, 1.0 - frac)
            sel  = maxh > thr                                   # selective voxels
            winner = zh.argmax(axis=1) + 1                      # 1=face..4=word
            winner[~sel] = 0

            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            if sid not in cov.index:
                continue
            c = cov.loc[sid]
            base = {'subject_id': sid, 'code': c.get('code', ''),
                    'session': c['session'], 'group': c['group'],
                    'status': 'control' if c['group'] == 'control' else c['group'],
                    'hemi': hemi, 'hemi_label': 'intact' if c['group'] != 'control' else hemi}

            # region='otc', denominator='selective'
            for k, cat in enumerate(CATEGORIES, start=1):
                cnt = int((winner == k).sum())
                rows.append({**base, 'region': 'otc', 'denominator': 'selective',
                             'category': cat, 'wta_pct': 100.0 * cnt / n_sel,
                             'voxel_count': cnt, 'denom_voxels': n_sel})

            # region='cluster_*', denominator='selective' (within this hemi's clusters)
            for (ccat, chemi), cmask in cluster_idx.items():
                if chemi != hemi:
                    continue
                wsub = winner[cmask]
                nsub = int((wsub > 0).sum())
                if nsub == 0:
                    continue
                for k, cat in enumerate(CATEGORIES, start=1):
                    cnt = int((wsub == k).sum())
                    rows.append({**base, 'region': f'cluster_{ccat}_{chemi}',
                                 'denominator': 'selective', 'category': cat,
                                 'wta_pct': 100.0 * cnt / nsub,
                                 'voxel_count': cnt, 'denom_voxels': nsub})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} rows, {out.subject_id.nunique()} subjects)")
    print("regions:", sorted(out.region.unique()))
    print("Next: 05_stats_harmony --wta wta_percentages_harmonized.csv (needs --wta flag).")


if __name__ == '__main__':
    main()