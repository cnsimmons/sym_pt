#!/usr/bin/env python3
"""
02_tfce_analyses.py

Voxelwise group comparisons within VOTC via FSL randomise.
Extracts thresholded group maps (no formal dataframe stats are run here).

Contrast modes (--contrast):
  - 'others'   (default): copes 6/7/8/9 (cat-vs-all-others).
  - 'baseline': copes 15/16/17/18 (cat-vs-baseline).

Usage:
  python 02_tfce_analyses.py
"""

import os, sys, time, argparse, subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import datasets as nl_datasets, image as nl_image

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from params import (processed_dir, should_skip, get_post_sessions,
                    is_patient, get_sub_info, _load_csv)

COPES_BY_MODE = {
    'others':   {'face': 6,  'house': 7,  'object': 8,  'word': 9},
    'baseline': {'face': 15, 'house': 16, 'object': 17, 'word': 18},
}
CATEGORIES = ['face', 'house', 'object', 'word']
HEMIS = ['l', 'r']
N_PERM_DEFAULT = 10000
BASELINE_DEFAULT_THRESH = 2.58
DEFAULT_CLUSTER_THRESH = 3.09  # ~p<.001 one-tailed

OUT_DIR = None

def get_out_dir(args):
    base = args.out_name if args.out_name else ('tfce_votc_catbaseline' if args.contrast == 'baseline' else 'tfce_votc')
    return Path(processed_dir) / 'group_results' / base

def build_votc_masks_and_save():
    ho_atlas = nl_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_img = ho_atlas.maps if isinstance(ho_atlas.maps, nib.Nifti1Image) else nib.load(ho_atlas.maps)
    ho_data = ho_img.get_fdata()
    ho_labels = ho_atlas.labels

    VOTC_LABEL_NAMES = [
        'Temporal Fusiform Cortex, anterior division',
        'Temporal Fusiform Cortex, posterior division',
        'Temporal Occipital Fusiform Cortex',
        'Parahippocampal Gyrus, anterior division',
        'Parahippocampal Gyrus, posterior division',
        'Lingual Gyrus',
        'Lateral Occipital Cortex, superior division',
        'Lateral Occipital Cortex, inferior division',
    ]
    full = np.zeros(ho_data.shape, dtype=bool)
    for name in VOTC_LABEL_NAMES:
        matches = [i for i, l in enumerate(ho_labels) if name in l]
        if matches:
            full |= (ho_data == matches[0])

    mid_x = ho_data.shape[0] // 2
    masks = {}
    for hemi, slc in [('l', slice(mid_x, None)), ('r', slice(None, mid_x))]:
        m = full.copy()
        m[slc] = False
        path = OUT_DIR / f'votc_{hemi}_mask.nii.gz'
        nib.save(nib.Nifti1Image(m.astype(np.uint8), ho_img.affine), path)
        masks[hemi] = path
    return masks

def load_subjects():
    df = _load_csv()
    subjects = {}
    for sc in sorted(df['sub_clean'].unique()):
        sid = f'sub-{sc}'
        
        if should_skip(sid):
            continue
            
        post = get_post_sessions(sc)
        if not post or not (Path(processed_dir) / sid).exists():
            continue
            
        first_post = post[0]
        info = get_sub_info(sc, first_post)
        pt = is_patient(sc)
        
        # Exclude nonOTC from group maps
        if info.get('group', 'unknown') == 'nonOTC':
            continue

        intact = info.get('intact_hemi', '')
        subjects[sid] = {
            'session': f'{first_post:02d}',
            'first_session': f'{first_post:02d}',
            'group': info.get('group', 'unknown'),
            'hemi': ('l' if intact == 'left' else 'r') if pt else None,
            'intact_hemi': intact if pt else 'both',
        }
    return subjects

def get_zstat_path(sid, session, first_session, cope_num):
    feat = (Path(processed_dir) / sid / f'ses-{session}' / 'derivatives' / 'fsl'
            / 'loc' / 'HighLevel.gfeat' / f'cope{cope_num}.feat' / 'stats')
    if session == first_session:
        return feat / 'zstat1_mni.nii.gz'
    else:
        return feat / f'zstat1_ses{first_session}_mni.nii.gz'

def write_design_files(out_prefix, n_ctrl, n_pt):
    n_total = n_ctrl + n_pt
    mat_path = Path(f'{out_prefix}.mat')
    con_path = Path(f'{out_prefix}.con')
    with mat_path.open('w') as f:
        f.write(f'/NumWaves 2\n/NumPoints {n_total}\n/Matrix\n')
        for _ in range(n_ctrl): f.write('1 0\n')
        for _ in range(n_pt): f.write('0 1\n')
    with con_path.open('w') as f:
        f.write('/ContrastName1 ctrl_gt_pt\n/ContrastName2 pt_gt_ctrl\n')
        f.write('/NumWaves 2\n/NumContrasts 2\n/Matrix\n')
        f.write('1 -1\n-1 1\n')
    return mat_path, con_path

def merge_zstats(subject_paths, out_path, threshold=None):
    if threshold is None:
        subprocess.run(['fslmerge', '-t', str(out_path)] + [str(p) for p in subject_paths], check=True)
        return
    vols, ref_affine, ref_header = [], None, None
    for p in subject_paths:
        img = nib.load(p)
        thr_img = nl_image.threshold_img(img, threshold=threshold, two_sided=False)
        vols.append(thr_img.get_fdata().astype(np.float32))
        if ref_affine is None:
            ref_affine, ref_header = img.affine, img.header
    nib.save(nib.Nifti1Image(np.stack(vols, axis=-1), ref_affine, ref_header), str(out_path))

def run_randomise(input_4d, out_prefix, mask, design_mat, design_con, n_perm, inference, cluster_thresh):
    cmd = ['randomise', '-i', str(input_4d), '-o', str(out_prefix), '-m', str(mask), 
           '-d', str(design_mat), '-t', str(design_con), '-R', '-n', str(n_perm), '--seed=42']
    if inference in ('tfce', 'both'): cmd += ['-T']
    if inference in ('cluster', 'both'): cmd += ['-c', str(cluster_thresh)]
    subprocess.run(cmd, check=True)

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast', choices=['others', 'baseline'], default='others')
    parser.add_argument('--category', choices=CATEGORIES, default=None)
    parser.add_argument('--hemi', choices=HEMIS, default=None)
    parser.add_argument('--n-perm', type=int, default=N_PERM_DEFAULT)
    parser.add_argument('--thresh', type=float, default=None)
    parser.add_argument('--inference', choices=['tfce', 'cluster', 'both'], default='tfce')
    parser.add_argument('--cluster-thresh', type=float, default=DEFAULT_CLUSTER_THRESH)
    parser.add_argument('--out-name', type=str, default=None)
    args = parser.parse_args()

    OUT_DIR = get_out_dir(args)
    CAT_COPES = COPES_BY_MODE[args.contrast]
    subject_threshold = args.thresh if args.thresh is not None else (BASELINE_DEFAULT_THRESH if args.contrast == 'baseline' else None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks = build_votc_masks_and_save()
    subjects = load_subjects()
    
    ctrl_sids = [s for s, i in subjects.items() if i['group'] == 'control']
    pt_sids = [s for s, i in subjects.items() if i['group'] == 'OTC']
    
    cats_to_run = [args.category] if args.category else CATEGORIES
    hemis_to_run = [args.hemi] if args.hemi else HEMIS

    for cat in cats_to_run:
        for hemi in hemis_to_run:
            test_dir = OUT_DIR / f'{cat}_{hemi}_pt_vs_ctrl'
            test_dir.mkdir(parents=True, exist_ok=True)
            
            pt_for_hemi = [s for s in pt_sids if subjects[s]['hemi'] == hemi]
            ctrl_paths = [p for s in ctrl_sids if (p := get_zstat_path(s, subjects[s]['session'], subjects[s]['first_session'], CAT_COPES[cat])).exists()]
            pt_paths = [p for s in pt_for_hemi if (p := get_zstat_path(s, subjects[s]['session'], subjects[s]['first_session'], CAT_COPES[cat])).exists()]

            if len(ctrl_paths) < 5 or len(pt_paths) < 3: continue
            
            merged = test_dir / 'merged_zstat.nii.gz'
            merge_zstats(ctrl_paths + pt_paths, merged, threshold=subject_threshold)
            mat_path, con_path = write_design_files(str(test_dir / 'design'), len(ctrl_paths), len(pt_paths))
            run_randomise(merged, test_dir / 'rand', masks[hemi], mat_path, con_path, args.n_perm, args.inference, args.cluster_thresh)

if __name__ == '__main__':
    main()