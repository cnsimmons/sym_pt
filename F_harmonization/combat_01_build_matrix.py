#!/usr/bin/env python3
"""
combat_01_build_matrix.py — Assemble the ComBat feature matrix (step 2).

Mirrors tfce_votc_contrasts.py EXACTLY by importing its own functions, so the
subjects, sessions, zstat paths, VOTC masks, and copes are identical by
construction. Harmonized maps therefore feed straight back into the same TFCE.

For each hemisphere (l, r) and each category (face/house/object/word = copes
6/7/8/9, cat-vs-all-others, raw zstat — TFCE 'others' mode), extracts each
subject's voxelwise zstat within that hemisphere's VOTC mask.

Outputs to F_harmonization/combat_inputs/:
  covars.csv               one row per subject: group, scanner, age, sex, intact_hemi
  votc_{hemi}_mask.nii.gz  VOTC masks (for writing harmonized maps back later)
  features_{hemi}.npz      per-category [n_subj x n_vox] matrices + subject order + mask

Run on the cluster (needs processed data + FSL/nilearn):  python combat_01_build_matrix.py
"""
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

GIT_DIR = '/user_data/csimmon2/git_repos/sym_pt'
sys.path.insert(0, GIT_DIR)
sys.path.insert(0, str(Path(GIT_DIR) / 'D_liu'))

import tfce_votc_contrasts as tfce          # reuse its subject/session/path/mask logic
from sym_pt_params import processed_dir

OUT_DIR     = Path(GIT_DIR) / 'F_harmonization' / 'combat_inputs'
SCANNER_CSV = Path(GIT_DIR) / 'F_harmonization' / 'sub_info_scanner.csv'
MODE        = 'others'                       # raw zstat, matches the primary TFCE run
COPES       = tfce.COPES_BY_MODE[MODE]       # {'face':6,'house':7,'object':8,'word':9}
CATEGORIES  = tfce.CATEGORIES                # ['face','house','object','word']
HEMIS       = tfce.HEMIS                     # ['l','r']


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. VOTC masks: reuse TFCE's builder (it writes to tfce.OUT_DIR) ---
    tfce.OUT_DIR = OUT_DIR
    print('Building VOTC masks...')
    masks = tfce.build_votc_masks_and_save()         # {'l': path, 'r': path}

    # --- 2. subjects/sessions: EXACTLY as TFCE selects them ---
    print('\nLoading subjects (mirroring TFCE)...')
    subjects = tfce.load_subjects()
    scan = pd.read_csv(SCANNER_CSV)

    # --- 3. covariate table (batch = scanner; covariates to preserve in step 3) ---
    covar_rows = []
    for sid, info in subjects.items():
        ses = f"ses-{info['session']}"
        row = scan[(scan['sub'] == sid) & (scan['ses'] == ses)]
        if not len(row):
            print(f'  WARNING: no scanner-csv row for {sid} {ses} -- skipped')
            continue
        r = row.iloc[0]
        covar_rows.append({
            'subject_id':  sid,
            'session':     info['session'],
            'group':       info['group'],         # control / OTC
            'scanner':     r['scanner'],          # Verio / Prisma  (batch)
            'age':         r['age'],
            'sex':         r['sex'],
            'intact_hemi': info['intact_hemi'],   # left/right; 'both' for controls
        })
    covars = pd.DataFrame(covar_rows).sort_values('subject_id').reset_index(drop=True)
    covars.to_csv(OUT_DIR / 'covars.csv', index=False)
    print(f'  covars: {len(covars)} subjects | scanner = {covars.scanner.value_counts().to_dict()}')
    print(f'           group = {covars.group.value_counts().to_dict()}')

    # --- 4. voxelwise extraction per hemisphere x category, within VOTC mask ---
    for hemi in HEMIS:
        mask_img = nib.load(str(masks[hemi]))
        mask = mask_img.get_fdata().astype(bool)
        n_vox = int(mask.sum())
        print(f'\n[{hemi.upper()}H] VOTC mask = {n_vox:,} voxels')

        saved = {'mask': mask, 'affine': mask_img.affine, 'shape': np.array(mask_img.shape)}
        for cat in CATEGORIES:
            cope = COPES[cat]
            rows, subs, missing = [], [], []
            for sid in covars['subject_id']:
                info = subjects[sid]
                p = tfce.get_zstat_path(sid, info['session'], info['first_session'], cope)
                if not p.exists():
                    missing.append(sid)
                    continue
                d = nib.load(str(p)).get_fdata()
                rows.append(d[mask].astype(np.float32))
                subs.append(sid)
            X = np.vstack(rows) if rows else np.empty((0, n_vox), np.float32)
            saved[f'X_{cat}']    = X
            saved[f'subs_{cat}'] = np.array(subs)
            msg = f'  {cat:7s} cope{cope}: {X.shape[0]:3d} subjects x {n_vox} vox'
            if missing:
                msg += f'   (missing {len(missing)}: {missing[:3]}{"..." if len(missing) > 3 else ""})'
            print(msg)

        np.savez_compressed(OUT_DIR / f'features_{hemi}.npz', **saved)
        print(f'  -> features_{hemi}.npz')

    print(f'\nDone. ComBat inputs written to {OUT_DIR}')


if __name__ == '__main__':
    main()