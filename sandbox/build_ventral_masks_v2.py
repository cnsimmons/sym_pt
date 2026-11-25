#!/usr/bin/env python3
"""
build_ventral_masks.py - a ventral OTC mask (no superior lateral occipital cortex) and an LO search
mask without it. Nothing existing is modified.

  --check     Rebuild the CURRENT whole-OTC masks from the Harvard-Oxford probability atlas and
              report which definition reproduces group_results/tfce_votc_harmonized/votc_{l,r}_mask
              voxel for voxel (compared in world coordinates; those masks are RAS, the FSL atlas LAS).
              Read-only.
  --write-otc Same definition minus HO 21 (LOC superior) -> sandbox/ventral_otc/votc7_{l,r}_mask.nii.gz.
              Runs only if --check finds an exact match.
  --write-lo  Per subject, first session (as 14_create_searchmasks.py): ROIs/{l,r}_object_LOC_inf_
              searchmask.nii.gz = HO 22 only (LOC inferior), built with 14_create_searchmasks.py's own
              functions (25%, no dilation, FLIRT mni2anat, nearest neighbour). Writes only that new
              file name, as 15_add_ppa_strict_searchmasks.py does for house_PPA_strict.
  --dry-run   With --write-lo: list what would be written.
  --sub 039   With --write-lo: one subject.

Run from /user_data/csimmon2/git_repos/sym_pt/sandbox in (fmri), FSL loaded.
"""
import argparse
import glob
import importlib.util
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

GIT   = Path(os.environ.get('SYMPT_GIT', '/user_data/csimmon2/git_repos/sym_pt'))
PROC  = Path(os.environ.get('SYMPT_PROC', '/user_data/csimmon2/sym_pt'))
FSL   = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
ATLAS = Path(f'{FSL}/data/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz')
MAXPROB = Path(f'{FSL}/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')
ORIG  = {h: PROC / 'group_results' / 'tfce_votc_harmonized' / f'votc_{h}_mask.nii.gz' for h in 'lr'}
OUT   = GIT / 'sandbox' / 'ventral_otc'

# The Methods: anterior + posterior temporal fusiform (36, 37), occipital fusiform, anterior +
# posterior parahippocampal (33, 34), lingual (35), LOC superior + inferior (21, 22). "Occipital
# fusiform" could be HO 38 (temporal occipital fusiform) or 39 (occipital fusiform gyrus): test both.
BASE  = [21, 22, 33, 34, 35, 36, 37]
CANDIDATE_EXTRA = {'+39': [39], '+38': [38], '+38+39': [38, 39]}
LOC_SUPERIOR, LOC_INFERIOR = 21, 22


def world_set(data, affine):
    ijk = np.argwhere(data > 0)
    return {tuple(np.round(c, 1)) for c in nib.affines.apply_affine(affine, ijk)}


def build(atlas, affine, labels, rule):
    if rule == 'maxprob25':                        # 3-D label image: value = HO index + 1
        binary = np.isin(np.rint(atlas).astype(int), [l + 1 for l in labels])
    else:
        comb = np.max(atlas[..., labels], axis=3)
        binary = comb > 25 if rule == '>25' else comb >= 25
    xs = np.array([nib.affines.apply_affine(affine, [i, 0, 0])[0] for i in range(binary.shape[0])])
    out = {}
    for midline in ['left', 'none']:
        left = xs < 0 if midline == 'none' else xs <= 0
        right = xs > 0
        out[midline] = {'l': binary & left[:, None, None], 'r': binary & right[:, None, None]}
    return out


def check():
    img = nib.load(str(ATLAS)); atlas = img.get_fdata(); aff = img.affine
    mimg = nib.load(str(MAXPROB)); matlas = mimg.get_fdata(); maff = mimg.affine
    orig = {}
    for h, p in ORIG.items():
        oi = nib.load(str(p)); orig[h] = world_set(oi.get_fdata(), oi.affine)
    print(f'current masks: left {len(orig["l"])} voxels, right {len(orig["r"])} voxels')
    matches = []
    for (name, extra), rule in itertools.product(CANDIDATE_EXTRA.items(), ['>25', '>=25', 'maxprob25']):
        A, AF = (matlas, maff) if rule == 'maxprob25' else (atlas, aff)
        for midline, m in build(A, AF, BASE + extra, rule).items():
            got = {h: world_set(m[h], AF) for h in 'lr'}
            same = all(got[h] == orig[h] for h in 'lr')
            dice = [2 * len(got[h] & orig[h]) / (len(got[h]) + len(orig[h])) for h in 'lr']
            print(f'  labels {BASE + extra}  {rule:9s} midline->{midline:4s}: '
                  f'L {len(got["l"]):6d} R {len(got["r"]):6d}  Dice L {dice[0]:.4f} R {dice[1]:.4f}'
                  f'{"   EXACT MATCH" if same else ""}')
            if same:
                matches.append((BASE + extra, rule, midline))
    if not matches:
        print('\nNo definition reproduces the current masks: STOP, nothing written.')
        return None
    print(f'\n{len(matches)} definition(s) reproduce the current masks exactly:')
    for m in matches:
        print(f'  labels {m[0]}, {m[1]}, midline -> {m[2]}')
    return matches, {'prob': img, 'max': mimg}


def write_otc():
    res = check()
    if res is None:
        sys.exit(1)
    matches, imgs = res
    versions = []
    for labels, rule, midline in matches:   # every matching definition must give the same new mask
        im = imgs['max'] if rule == 'maxprob25' else imgs['prob']
        keep = [l for l in labels if l != LOC_SUPERIOR]
        versions.append((keep, build(im.get_fdata(), im.affine, keep, rule)[midline], im.affine))
    if any(not all((v[1][h] == versions[0][1][h]).all() for h in 'lr') for v in versions[1:]):
        sys.exit('The matching definitions disagree once LOC superior is dropped: STOP, nothing written.')
    keep, m, aff = versions[0]
    OUT.mkdir(parents=True, exist_ok=True)
    for h in 'lr':
        p = OUT / f'votc7_{h}_mask.nii.gz'
        nib.save(nib.Nifti1Image(m[h].astype(np.float32), aff), str(p))
        print(f'wrote {p}: {int(m[h].sum())} voxels (labels {keep})')


def write_lo(sub=None, dry_run=False):
    hits = sorted(glob.glob(str(GIT / 'pipeline' / '*' / '14_create_searchmasks.py')))
    if len(hits) != 1:
        sys.exit(f'expected one 14_create_searchmasks.py under pipeline/*/, found {hits}')
    spec = importlib.util.spec_from_file_location('csm', hits[0])
    csm = importlib.util.module_from_spec(spec); spec.loader.exec_module(csm)
    csm.CATEGORY_PARCELS = {'object_LOC_inf': {'indices': [LOC_INFERIOR], 'names': ['Lateral Occipital inferior']}}
    atlas_img, atlas_data = csm.load_atlas()
    subs = [sub.replace('sub-', '')] if sub else sorted(
        d.replace('sub-', '') for d in os.listdir(csm.processed_dir)
        if d.startswith('sub-') and d.replace('sub-', '') not in csm.skip_subs)
    total = 0
    for s in subs:
        ses = csm.get_sessions(s)
        if not ses:
            continue
        print(f'=== sub-{s} ses-{ses[0]:02d} ===')
        total += csm.create_searchmasks_for_subject(s, ses[0], atlas_img, atlas_data, csm.PROB_THRESHOLD, dry_run)
    print(f'{"Would write" if dry_run else "Wrote"}: {total} object_LOC_inf masks')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true'); g.add_argument('--write-otc', action='store_true')
    g.add_argument('--write-lo', action='store_true')
    ap.add_argument('--sub'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.check: check()
    elif a.write_otc: write_otc()
    else: write_lo(a.sub, a.dry_run)
