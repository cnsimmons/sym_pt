#!/usr/bin/env python3
"""
combat_01_build_matrix.py — assemble the ComBat feature matrix (step 2).

WRAPPER around the VERIFIED manuscript TFCE (verified/02_tfce_analyses): it imports
that module and uses ITS load_subjects / get_zstat_path / build_votc_masks_and_save
/ COPES_BY_MODE. So subjects, sessions, copes, and VOTC masks are IDENTICAL to the
manuscript pipeline by construction.

Extracts RAW zstat within VOTC per category (the 0.0 threshold is applied later at
merge, exactly as the verified pipeline does — not at extraction).

Outputs to F_harmonization/combat_inputs/:
  covars.csv (subject_id, session, group, scanner, age, sex)
  votc_{hemi}_mask.nii.gz
  features_{hemi}.npz  (per-category [n_subj x n_vox] + subject order + mask)
"""
import importlib.util
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

GIT_DIR  = Path('/user_data/csimmon2/git_repos/sym_pt')
VTFCE    = GIT_DIR / 'D_liu' / 'verified' / '02_tfce_analyses_not_as_verified.py'
OUT_DIR  = GIT_DIR / 'F_harmonization' / 'combat_inputs'
SCANNER  = GIT_DIR / 'F_harmonization' / 'sub_info_scanner.csv'
MODE     = 'others'

sys.path.insert(0, str(GIT_DIR))
# load the verified manuscript TFCE module (filename starts with a digit)
spec = importlib.util.spec_from_file_location('verified_tfce', str(VTFCE))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

COPES = v.COPES_BY_MODE[MODE]          # {'face':6,'house':7,'object':8,'word':9}
CATEGORIES, HEMIS = v.CATEGORIES, v.HEMIS


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v.OUT_DIR = OUT_DIR                 # verified mask builder writes here
    print('Building VOTC masks (verified)...')
    masks = v.build_votc_masks_and_save()

    print('Loading subjects (verified load_subjects)...')
    subjects = v.load_subjects()
    scan = pd.read_csv(SCANNER)

    # covariates: group from verified; scanner/age/sex from scanner csv
    rows = []
    for sid, info in subjects.items():
        ses = f"ses-{info['session']}"
        r = scan[(scan['sub'] == sid) & (scan['ses'] == ses)]
        if not len(r):
            print(f'  WARNING: no scanner row for {sid} {ses} -- skipped')
            continue
        r = r.iloc[0]
        rows.append({'subject_id': sid, 'session': info['session'],
                     'group': info['group'], 'scanner': r['scanner'],
                     'age': r['age'], 'sex': r['sex']})
    covars = pd.DataFrame(rows).sort_values('subject_id').reset_index(drop=True)
    covars.to_csv(OUT_DIR / 'covars.csv', index=False)
    print(f"  covars: {len(covars)} subjects | "
          f"scanner={covars.scanner.value_counts().to_dict()} | "
          f"group={covars.group.value_counts().to_dict()}")

    for hemi in HEMIS:
        mimg = nib.load(str(masks[hemi]))
        mask = mimg.get_fdata().astype(bool)
        nvox = int(mask.sum())
        print(f"\n[{hemi.upper()}H] VOTC = {nvox:,} vox")
        saved = {'mask': mask, 'affine': mimg.affine, 'shape': np.array(mimg.shape)}
        for cat in CATEGORIES:
            cope = COPES[cat]
            X, subs, miss = [], [], []
            for sid in covars['subject_id']:
                info = subjects[sid]
                p = v.get_zstat_path(sid, info['session'], info['first_session'], cope)
                if not p.exists():
                    miss.append(sid); continue
                X.append(nib.load(str(p)).get_fdata()[mask].astype(np.float32))
                subs.append(sid)
            saved[f'X_{cat}']    = np.vstack(X) if X else np.empty((0, nvox), np.float32)
            saved[f'subs_{cat}'] = np.array(subs)
            msg = f"  {cat:7s} cope{cope}: {len(subs)} subj"
            if miss:
                msg += f"  (missing {len(miss)}: {miss[:3]})"
            print(msg)
        np.savez_compressed(OUT_DIR / f'features_{hemi}.npz', **saved)
        print(f"  -> features_{hemi}.npz")

    print(f"\nDone -> {OUT_DIR}")


if __name__ == '__main__':
    main()