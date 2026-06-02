#!/usr/bin/env python3
"""
03_wta_analysis.py

Extracts Winner-Take-All (WTA) categorical territory percentages for each subject.
Does NOT run statistical comparisons (LMMs/permutations). Outputs a clean CSV
to be used by downstream statistical scripts.

Pipeline:
  - Load MNI-space z-stats for Face, House, Object, Word (Baseline contrasts).
  - Apply Left and Right Harvard-Oxford VOTC masks.
  - Voxel allegiance is awarded to the category with the highest z-stat, 
    provided that max z-stat > 2.326 (p<.01 one-tailed).
  - Calculate the percentage of *selective* VOTC territory won by each category.

Output: D_liu/wta_percentages_v1.csv
"""

import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from nilearn import datasets as nl_datasets

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from params import (processed_dir, should_skip, get_post_sessions,
                    is_patient, get_sub_info, _load_csv)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = Path(processed_dir)
OUTPUT_DIR  = Path('/user_data/csimmon2/git_repos/sym_pt/D_liu')
OUTPUT_NAME = 'wta_percentages_v1.csv'

SEL_Z_THRESH = float(norm.ppf(0.99)) # ≈2.326
CATEGORIES = ['face', 'house', 'object', 'word']
# Baseline copes for WTA competition
COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

def get_votc_masks():
    """Build left and right VOTC masks from Harvard-Oxford atlas."""
    ho_atlas = nl_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_img = ho_atlas.maps if isinstance(ho_atlas.maps, nib.Nifti1Image) else nib.load(ho_atlas.maps)
    ho_data = ho_img.get_fdata()
    
    names = ['Temporal Fusiform', 'Temporal Occipital Fusiform', 
             'Parahippocampal', 'Lingual', 'Lateral Occipital']
    
    full_mask = np.zeros(ho_data.shape, dtype=bool)
    for i, label in enumerate(ho_atlas.labels):
        if any(n in label for n in names):
            full_mask |= (ho_data == i)

    mid_x = ho_data.shape[0] // 2
    l_mask, r_mask = full_mask.copy(), full_mask.copy()
    l_mask[mid_x:, :, :] = False
    r_mask[:mid_x, :, :] = False
    
    return {'l': l_mask, 'r': r_mask}

def load_zstat(sid, session, first_session, cope_num):
    feat = BASE_DIR / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc' / 'HighLevel.gfeat'
    zname = 'zstat1_mni.nii.gz' if session == first_session else f'zstat1_ses{first_session}_mni.nii.gz'
    zpath = feat / f'cope{cope_num}.feat' / 'stats' / zname
    
    if not zpath.exists():
        return None
    return nib.load(zpath).get_fdata()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_NAME
    
    print('Building VOTC masks...')
    masks = get_votc_masks()
    
    df_csv = _load_csv()
    subjects = sorted(df_csv['sub_clean'].unique())
    
    all_rows = []
    
    print('Computing Winner-Take-All territory allocations...')
    for sc in subjects:
        sid = f'sub-{sc}'
        if should_skip(sid): continue
            
        post = get_post_sessions(sc)
        if not post or not (BASE_DIR / sid).exists(): continue
            
        # WTA is currently cross-sectional (first post-surgery session)
        session = f'{post[0]:02d}'
        info = get_sub_info(sc, post[0])
        pt = is_patient(sc)
        group = info.get('group', 'unknown')
        
        # Load the 4 category maps
        z_maps = []
        missing = False
        for cat in CATEGORIES:
            z = load_zstat(sid, session, session, COPES[cat])
            if z is None:
                missing = True
                break
            z_maps.append(z)
            
        if missing:
            print(f'  [{sid}] SKIP: missing z-stats')
            continue
            
        # Stack maps: Shape (X, Y, Z, 4)
        z_stack = np.stack(z_maps, axis=-1)
        
        # Identify the winner in every voxel
        winner_idx = np.argmax(z_stack, axis=-1)  # 0=face, 1=house, 2=object, 3=word
        max_z = np.max(z_stack, axis=-1)
        
        # Determine valid hemispheres to process
        hemis_to_run = ['l', 'r'] if group == 'control' else [('l' if info.get('intact_hemi') == 'left' else 'r')]
        
        for hemi in hemis_to_run:
            hemi_mask = masks[hemi]
            
            # Mask to specific hemisphere and apply selectivity threshold
            valid_voxels = hemi_mask & (max_z > SEL_Z_THRESH)
            n_selective_voxels = valid_voxels.sum()
            
            if n_selective_voxels == 0:
                continue
                
            winning_cats = winner_idx[valid_voxels]
            
            for i, cat in enumerate(CATEGORIES):
                voxels_won = (winning_cats == i).sum()
                pct_won = 100.0 * (voxels_won / n_selective_voxels)
                
                all_rows.append({
                    'subject_id': sid,
                    'code': f"{group}{sc}",
                    'session': session,
                    'group': group,
                    'status': 'patient' if pt else 'control',
                    'hemi': hemi,
                    'hemi_label': 'intact' if pt else ('left' if hemi == 'l' else 'right'),
                    'category': cat,
                    'wta_pct': pct_won,
                    'voxel_count': voxels_won,
                    'total_selective_voxels': n_selective_voxels
                })
                
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    
    print(f'\nSaved: {out_path}')
    print(f'Total Rows: {len(df)}')
    print(f'Subjects extracted: {df["subject_id"].nunique()}')

if __name__ == '__main__':
    main()