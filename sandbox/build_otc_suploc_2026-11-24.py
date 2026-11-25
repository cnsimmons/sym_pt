#!/usr/bin/env python3
"""
build_otc_suploc_2026-11-24.py - superior-LOC-only part of the current OTC mask

suploc = the current OTC mask restricted to Lateral Occipital Cortex, superior division
         (Harvard-Oxford index 21): exactly the voxels votc7 removed. Together with votc7
         it partitions the current mask.
Atlas: Harvard-Oxford cortical, max-probability, 25%, 2 mm (map value = index + 1).
Hemisphere split as in the current masks: left x < 0, right x >= 0 (midline to the right).

Each step must pass before the next; nothing is written unless steps 1 and 2 pass:
  1. the atlas XML names match the expected label names
  2. rebuilding the current masks from the atlas reproduces them exactly
     (confirms the atlas, the eight labels and the midline rule)
  3. build suploc; report voxels per hemisphere; check suploc + votc7 = current, no overlap

Writes only <git>/sandbox/ventral_otc/suploc_{l,r}_mask.nii.gz and refuses to overwrite.
The current masks and votc7 are only read.

Usage (cluster, fmri env):
  python build_otc_suploc_2026-11-24.py --check   # steps 1-2 only, writes nothing
  python build_otc_suploc_2026-11-24.py           # steps 1-3
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import nibabel as nib

FSLDIR = Path(os.environ.get('FSLDIR', '/opt/fsl/6.0.3'))
GIT = Path(os.environ.get('SYMPT_GIT', '/user_data/csimmon2/git_repos/sym_pt'))
PROC = Path(os.environ.get('SYMPT_PROC', '/user_data/csimmon2/sym_pt'))
ATLAS = FSLDIR / 'data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
XML = FSLDIR / 'data/atlases/HarvardOxford-Cortical.xml'
ORIG = {h: PROC / 'group_results/tfce_votc_harmonized' / f'votc_{h}_mask.nii.gz' for h in 'lr'}
OUT = GIT / 'sandbox/ventral_otc'
VOTC7 = {h: OUT / f'votc7_{h}_mask.nii.gz' for h in 'lr'}
SUPLOC = {h: OUT / f'suploc_{h}_mask.nii.gz' for h in 'lr'}

CURRENT = [21, 22, 33, 34, 35, 36, 37, 38]
KEEP = [21]
NAMES = {21: 'Lateral Occipital Cortex, superior division',
         22: 'Lateral Occipital Cortex, inferior division',
         33: 'Parahippocampal Gyrus, anterior division',
         34: 'Parahippocampal Gyrus, posterior division',
         35: 'Lingual Gyrus',
         36: 'Temporal Fusiform Cortex, anterior division',
         37: 'Temporal Fusiform Cortex, posterior division',
         38: 'Temporal Occipital Fusiform Cortex'}


def stop(msg):
    print('STOP: ' + msg)
    sys.exit(1)


def check_names():
    if not XML.exists():
        stop(f'atlas XML not found: {XML}')
    found = {int(e.get('index')): (e.text or '').strip() for e in ET.parse(XML).getroot().iter('label')}
    bad = [(i, n, found.get(i)) for i, n in NAMES.items() if (found.get(i) or '').lower() != n.lower()]
    if bad:
        stop('label names differ from the expected ones: ' + '; '.join(f'{i}: expected "{n}", atlas "{g}"' for i, n, g in bad))
    print(f'step 1 ok: all {len(NAMES)} label names match the atlas XML')


def labels_on(img, atlas):
    """Atlas index (0-based, -1 = unlabeled) and world x for every voxel of img's grid."""
    lab = np.rint(np.asarray(atlas.dataobj)).astype(int) - 1
    shp = img.shape[:3]
    ijk = np.indices(shp).reshape(3, -1).T
    xyz = nib.affines.apply_affine(img.affine, ijk)
    a = np.rint(nib.affines.apply_affine(np.linalg.inv(atlas.affine), xyz)).astype(int)
    ok = np.all((a >= 0) & (a < np.array(lab.shape)), axis=1)
    out = np.full(len(ijk), -1)
    out[ok] = lab[a[ok, 0], a[ok, 1], a[ok, 2]]
    return out.reshape(shp), xyz[:, 0].reshape(shp)


def side(x, h):
    return x < 0 if h == 'l' else x >= 0


def load_current(atlas):
    cur = {}
    for h, p in ORIG.items():
        if not p.exists():
            stop(f'current mask not found: {p}\n  find it with: find {PROC}/group_results -name "votc_{h}_mask.nii.gz"')
        img = nib.load(str(p))
        lab, x = labels_on(img, atlas)
        cur[h] = (img, np.asarray(img.dataobj) > 0, lab, x)
    return cur


def check_rebuild(cur):
    for h, (img, m, lab, x) in cur.items():
        rb = np.isin(lab, CURRENT) & side(x, h)
        if not np.array_equal(rb, m):
            stop(f'{h}: rebuilding the current mask does not reproduce it '
                 f'(current {int(m.sum())}, rebuild {int(rb.sum())}, current only {int((m & ~rb).sum())}, '
                 f'rebuild only {int((rb & ~m).sum())}). Nothing written.')
        print(f'step 2 ok: {h} current mask rebuilt exactly ({int(m.sum())} voxels)')


def build(cur):
    existing = [str(p) for p in SUPLOC.values() if p.exists()]
    if existing:
        stop('output exists, refusing to overwrite: ' + ', '.join(existing))
    OUT.mkdir(parents=True, exist_ok=True)
    for h, (img, m, lab, x) in cur.items():
        new = m & np.isin(lab, KEEP)
        msg = f'{h}: current {int(m.sum())} -> suploc {int(new.sum())}'
        if VOTC7[h].exists():
            v7 = np.asarray(nib.load(str(VOTC7[h])).dataobj) > 0
            msg += (f'; suploc + votc7 == current: {bool(np.array_equal(new | v7, m))}'
                    f'; overlap with votc7: {int((new & v7).sum())}')
        nib.save(nib.Nifti1Image(new.astype(np.float32), img.affine, img.header), str(SUPLOC[h]))
        print(f'step 3: {msg}\n        wrote {SUPLOC[h]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='steps 1-2 only; write nothing')
    a = ap.parse_args()
    if not ATLAS.exists():
        stop(f'atlas not found: {ATLAS}')
    atlas = nib.load(str(ATLAS))
    check_names()
    cur = load_current(atlas)
    check_rebuild(cur)
    if not a.check:
        build(cur)
