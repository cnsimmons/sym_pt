#!/usr/bin/env python3
"""
combat_03_tfce_harmonized.py — TFCE on harmonized maps (step 4), VERIFIED-matched.

WRAPPER around verified/02_tfce_analyses: it reuses that module's own
merge_zstats / write_design_files / run_randomise / load_subjects / masks, so the
run is IDENTICAL to the manuscript TFCE in every respect (subjects, sessions,
masks, threshold=0.0, design, 10k perms, seed 42) EXCEPT the input maps are
ComBat-harmonized.

For each subject it writes a harmonized copy of the verified zstat map (VOTC voxels
overwritten with harmonized values), then hands those to verified's own merge +
randomise. The ONLY new code here is overwriting VOTC voxels.

Output: processed_dir/group_results/tfce_votc_harmonized/  (mirrors tfce_votc_fdr)
Compute-heavy -> submit via SLURM (submit_combat_tfce.sh).

Verify after running: the design.mat and subject order here are produced by
verified's own write_design_files / load_subjects, so they will match tfce_votc_fdr.
"""
import importlib.util
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

GIT_DIR = Path('/user_data/csimmon2/git_repos/sym_pt')
VTFCE   = GIT_DIR / 'D_liu' / 'verified' / '02_tfce_analyses_not_as_verified.py'
HARM    = GIT_DIR / 'F_harmonization' / 'combat_harmonized'
MODE    = 'others'

sys.path.insert(0, str(GIT_DIR))
spec = importlib.util.spec_from_file_location('verified_tfce', str(VTFCE))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

from params import processed_dir
OUT_DIR   = Path(processed_dir) / 'group_results' / 'tfce_votc_harmonized'
HARM_MAPS = OUT_DIR / 'harmonized_subject_maps'
COPES     = v.COPES_BY_MODE[MODE]
CATEGORIES, HEMIS = v.CATEGORIES, v.HEMIS
THRESH    = v.OTHERS_DEFAULT_THRESH       # 0.0 — identical to verified


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HARM_MAPS.mkdir(parents=True, exist_ok=True)
    v.OUT_DIR = OUT_DIR
    masks = v.build_votc_masks_and_save()
    subjects = v.load_subjects()
    ctrl_sids = [s for s, i in subjects.items() if i['group'] == 'control']
    pt_sids   = [s for s, i in subjects.items() if i['group'] == 'OTC']

    harm = {h: np.load(HARM / f'harmonized_{h}.npz', allow_pickle=True) for h in HEMIS}
    idx  = {h: {sid: i for i, sid in enumerate(list(harm[h]['subs']))} for h in HEMIS}

    def write_harmonized_map(sid, cat, hemi):
        """Verified zstat map with VOTC voxels overwritten by harmonized values."""
        info = subjects[sid]
        src = v.get_zstat_path(sid, info['session'], info['first_session'], COPES[cat])
        img = nib.load(str(src))
        data = img.get_fdata().astype(np.float32)
        mask = harm[hemi]['mask'].astype(bool)
        data[mask] = harm[hemi][f'X_{cat}'][idx[hemi][sid]]
        out = HARM_MAPS / f'{sid}_{cat}_{hemi}_harm.nii.gz'
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(out))
        return out

    for cat in CATEGORIES:
        for hemi in HEMIS:
            pt_for_hemi = [s for s in pt_sids if subjects[s]['hemi'] == hemi]
            ctrl = [s for s in ctrl_sids if s in idx[hemi]]
            pt   = [s for s in pt_for_hemi if s in idx[hemi]]
            if len(ctrl) < 5 or len(pt) < 3:
                print(f'[{cat}_{hemi}] SKIP (n_ctrl={len(ctrl)}, n_pt={len(pt)})')
                continue
            test_dir = OUT_DIR / f'{cat}_{hemi}_pt_vs_ctrl'
            test_dir.mkdir(parents=True, exist_ok=True)
            print(f'[{cat}_{hemi}] n_ctrl={len(ctrl)} n_pt={len(pt)} -> writing harmonized maps')

            paths  = [write_harmonized_map(s, cat, hemi) for s in (ctrl + pt)]
            merged = test_dir / 'merged_zstat.nii.gz'
            v.merge_zstats(paths, merged, threshold=THRESH)                       # verified merge + 0.0 thresh
            mat, con = v.write_design_files(str(test_dir / 'design'), len(ctrl), len(pt))  # verified design
            v.run_randomise(merged, test_dir / 'rand', masks[hemi], mat, con,
                            10000, 'tfce', v.DEFAULT_CLUSTER_THRESH)              # verified randomise

    print(f'\nDone -> {OUT_DIR}')


if __name__ == '__main__':
    main()