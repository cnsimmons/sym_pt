#!/usr/bin/env python3
"""
tfce_votc_contrasts.py

Voxelwise group comparisons within VOTC via FSL randomise.

Contrast modes (--contrast):
  - 'others'   (default): copes 6/7/8/9 (cat-vs-all-others).
  - 'baseline': copes 15/16/17/18 (cat-vs-baseline).

Subject-level threshold (--thresh):
  Each subject's zstat thresholded at z > THRESHOLD (one-sided; below → 0)
  before merging (Ayzenberg 2023 convention). Default: 2.58 for 'baseline',
  no threshold for 'others'.

Group-level inference (--inference):
  - 'tfce'   (default): TFCE-FWE via permutation
  - 'cluster': cluster-extent FWE, voxel-forming t > --cluster-thresh
  - 'both':   both methods on the same run
  Default cluster threshold: 3.09 (~p<.001 one-tailed).

Output (--out-name):
  Custom output dir under group_results/.

References:
  - Smith & Nichols (2009) NeuroImage — TFCE
  - Ayzenberg et al. (2023) Dev Cogn Neurosci — subject-level threshold
"""

import os, sys, time, argparse, subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import datasets as nl_datasets, image as nl_image

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
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

EXTRA_SKIP = {'sub-017', 'control083', 'control085'}
LIU_OVERLAP_SUBS = {'sub-004', 'sub-021', 'sub-044', 'sub-099'}
PRE_SURGERY_SESSIONS = {
    'sub-021': {'01'}, 'sub-045': {'01'}, 'sub-047': {'01'}, 'sub-049': {'01'},
    'sub-070': {'01'}, 'sub-073': {'01'}, 'sub-081': {'01'}, 'sub-086': {'01'},
    'sub-108': {'02'},
}

OUT_DIR = None


def get_out_dir(args):
    if args.out_name:
        base = args.out_name
    else:
        base = 'tfce_votc_catbaseline' if args.contrast == 'baseline' else 'tfce_votc'
    if args.exclude_liu:
        base = base + '_excl_liu'
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
        print(f'  {hemi.upper()}H VOTC mask: {int(m.sum()):,} voxels → {path.name}')
    return masks


def load_subjects():
    df = _load_csv()
    subjects = {}
    for sc in sorted(df['sub_clean'].unique()):
        if sc in skip_subs:
            continue
        sid = f'sub-{sc}'
        sessions = get_sessions(sc)
        if not sessions or not (Path(processed_dir) / sid).exists():
            continue
        info = get_sub_info(sc, sessions[0])
        pt = is_patient(sc)
        intact = info.get('intact_hemi', '')
        group = info.get('group', 'unknown')
        code_str = f"{group}{sc}"
        if code_str in EXTRA_SKIP or sid in EXTRA_SKIP:
            continue
        if group == 'nonOTC':
            continue

        post_sessions = []
        for s in sessions:
            ses_str = f'{s:02d}'
            if ses_str in PRE_SURGERY_SESSIONS.get(sid, set()):
                continue
            post_sessions.append(ses_str)
        if not post_sessions:
            continue

        subjects[sid] = {
            'session': post_sessions[0],
            'first_session': post_sessions[0],
            'group': group,
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
        for _ in range(n_ctrl):
            f.write('1 0\n')
        for _ in range(n_pt):
            f.write('0 1\n')
    with con_path.open('w') as f:
        f.write('/ContrastName1 ctrl_gt_pt\n')
        f.write('/ContrastName2 pt_gt_ctrl\n')
        f.write('/NumWaves 2\n/NumContrasts 2\n/Matrix\n')
        f.write('1 -1\n')
        f.write('-1 1\n')
    return mat_path, con_path


def merge_zstats(subject_paths, out_path, threshold=None):
    if threshold is None:
        cmd = ['fslmerge', '-t', str(out_path)] + [str(p) for p in subject_paths]
        subprocess.run(cmd, check=True)
        return
    vols, ref_affine, ref_header = [], None, None
    for p in subject_paths:
        img = nib.load(p)
        thr_img = nl_image.threshold_img(img, threshold=threshold, two_sided=False)
        data = thr_img.get_fdata().astype(np.float32)
        vols.append(data)
        if ref_affine is None:
            ref_affine, ref_header = img.affine, img.header
    merged = np.stack(vols, axis=-1)
    nib.save(nib.Nifti1Image(merged, ref_affine, ref_header), str(out_path))


def run_randomise(input_4d, out_prefix, mask, design_mat, design_con,
                   n_perm, inference, cluster_thresh):
    """Run randomise with TFCE and/or cluster-extent inference.

    inference: 'tfce' | 'cluster' | 'both'
    cluster_thresh: voxel-forming t threshold for cluster mode
    """
    cmd = [
        'randomise',
        '-i', str(input_4d),
        '-o', str(out_prefix),
        '-m', str(mask),
        '-d', str(design_mat),
        '-t', str(design_con),
        '-R',                        # raw t-stats
        '-n', str(n_perm),
        '--seed=42',
    ]
    if inference in ('tfce', 'both'):
        cmd += ['-T']
    if inference in ('cluster', 'both'):
        cmd += ['-c', str(cluster_thresh)]
    print(f'    Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def main():
    global OUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast', choices=['others', 'baseline'], default='others')
    parser.add_argument('--category', choices=CATEGORIES, default=None)
    parser.add_argument('--hemi', choices=HEMIS, default=None)
    parser.add_argument('--n-perm', type=int, default=N_PERM_DEFAULT)
    parser.add_argument('--thresh', type=float, default=None,
                        help="Subject-level zstat threshold.")
    parser.add_argument('--inference', choices=['tfce', 'cluster', 'both'],
                        default='tfce',
                        help="Group-level inference: 'tfce' (default), "
                             "'cluster' (extent-based FWE), or 'both'.")
    parser.add_argument('--cluster-thresh', type=float, default=DEFAULT_CLUSTER_THRESH,
                        help=f"Voxel-forming t-threshold for cluster-extent inference "
                             f"(default {DEFAULT_CLUSTER_THRESH}, ~p<.001 one-tailed).")
    parser.add_argument('--out-name', type=str, default=None)
    parser.add_argument('--exclude-liu', action='store_true')
    args = parser.parse_args()

    OUT_DIR = get_out_dir(args)
    CAT_COPES = COPES_BY_MODE[args.contrast]

    if args.thresh is not None:
        subject_threshold = args.thresh
    elif args.contrast == 'baseline':
        subject_threshold = BASELINE_DEFAULT_THRESH
    else:
        subject_threshold = None

    if args.exclude_liu:
        EXTRA_SKIP.update(LIU_OVERLAP_SUBS)
        print('SENSITIVITY MODE: excluding Liu overlap patients')

    print('=' * 70)
    print(f'Contrast: {args.contrast}  |  copes: {CAT_COPES}')
    print(f'Subject threshold: {subject_threshold}')
    print(f'Inference: {args.inference}'
          + (f' (cluster t>{args.cluster_thresh})' if args.inference in ('cluster', 'both') else ''))
    print(f'Output: {OUT_DIR}')
    print('=' * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cats_to_run = [args.category] if args.category else CATEGORIES
    hemis_to_run = [args.hemi] if args.hemi else HEMIS

    print('\nBuilding VOTC masks...')
    masks = build_votc_masks_and_save()

    print('\nLoading subjects...')
    subjects = load_subjects()
    ctrl_sids = [s for s, i in subjects.items() if i['group'] == 'control']
    pt_sids = [s for s, i in subjects.items() if i['group'] == 'OTC']
    pt_LH = [s for s in pt_sids if subjects[s]['hemi'] == 'l']
    pt_RH = [s for s in pt_sids if subjects[s]['hemi'] == 'r']
    print(f'  Controls: {len(ctrl_sids)}, OTC: {len(pt_sids)} '
          f'(LH-intact: {len(pt_LH)}, RH-intact: {len(pt_RH)})')

    summary_rows = []
    t0 = time.time()

    for cat in cats_to_run:
        cope = CAT_COPES[cat]
        for hemi in hemis_to_run:
            test_name = f'{cat}_{hemi}_pt_vs_ctrl'
            test_dir = OUT_DIR / test_name
            test_dir.mkdir(parents=True, exist_ok=True)
            print(f'\n[{test_name}] cope={cope}  {time.time()-t0:.0f}s elapsed')

            pt_for_hemi = [s for s in pt_sids if subjects[s]['hemi'] == hemi]
            print(f'  n_ctrl={len(ctrl_sids)}, n_pt={len(pt_for_hemi)}')

            ctrl_paths, missing_ctrl = [], []
            for s in ctrl_sids:
                p = get_zstat_path(s, subjects[s]['session'],
                                   subjects[s]['first_session'], cope)
                if p.exists():
                    ctrl_paths.append(p)
                else:
                    missing_ctrl.append(s)

            pt_paths, missing_pt = [], []
            for s in pt_for_hemi:
                p = get_zstat_path(s, subjects[s]['session'],
                                   subjects[s]['first_session'], cope)
                if p.exists():
                    pt_paths.append(p)
                else:
                    missing_pt.append(s)

            if missing_ctrl:
                print(f'  WARNING: missing zstat for {len(missing_ctrl)} ctrl: {missing_ctrl[:3]}...')
            if missing_pt:
                print(f'  WARNING: missing zstat for {len(missing_pt)} pt: {missing_pt}')

            if len(ctrl_paths) < 5 or len(pt_paths) < 3:
                print(f'  SKIP: insufficient subjects')
                continue

            merged = test_dir / 'merged_zstat.nii.gz'
            thresh_note = f' (threshold-img at z>{subject_threshold})' if subject_threshold is not None else ''
            print(f'  Merging {len(ctrl_paths) + len(pt_paths)} subjects{thresh_note}...')
            merge_zstats(ctrl_paths + pt_paths, merged, threshold=subject_threshold)

            design_prefix = test_dir / 'design'
            mat_path, con_path = write_design_files(
                str(design_prefix), len(ctrl_paths), len(pt_paths))

            randomise_prefix = test_dir / 'rand'
            mask_path = masks[hemi]
            run_randomise(merged, randomise_prefix, mask_path,
                          mat_path, con_path, args.n_perm,
                          args.inference, args.cluster_thresh)

            summary_rows.append({
                'category': cat, 'hemi': hemi, 'cope': cope,
                'subject_threshold': subject_threshold,
                'inference': args.inference,
                'cluster_thresh': args.cluster_thresh if args.inference in ('cluster', 'both') else None,
                'n_ctrl': len(ctrl_paths), 'n_pt': len(pt_paths),
                'output_dir': str(test_dir),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / 'tfce_summary.csv', index=False)
    print(f'\nDone in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()