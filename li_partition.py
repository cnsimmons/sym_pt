#!/usr/bin/env python3
"""Laterality index computed within LATERAL vs MEDIAL OTC, not the whole parcel.

WHY
  Nordt et al. (2021, Nat Hum Behav 5:1686-1697) report that developmental
  change in category selectivity is significant in LATERAL VTC and not
  statistically significant in medial VTC. The OTC parcel used elsewhere in this
  project contains both, so an LI computed over the whole parcel averages a
  subregion where selectivity changes with one where it does not. That would
  compress |LI| toward zero and remove the variance an age effect would act on
  -- which is what we observe (all four categories p > .10 vs age).

  This script recomputes the same voxel-count LI within three partitions of the
  same Harvard-Oxford labels, so the three are directly comparable.

PARTITIONS (approximate; the HO atlas has no mid-fusiform-sulcus boundary)
  whole   all 8 labels -- reproduces extract_selective_voxel_counts.py
  lateral Lateral Occipital Cortex sup + inf, Temporal Occipital Fusiform
  medial  Parahippocampal ant + post, Lingual, Temporal Fusiform ant + post

  Temporal Occipital Fusiform straddles the lateral/medial boundary and is
  assigned to lateral here. The `--tof-medial` flag moves it, so the assignment
  can be shown not to drive the result.

HEMISPHERE SPLIT
  By MNI x from the image affine (x < 0 = left), NOT by array position. Two
  scripts in this project split positionally with opposite conventions; doing it
  from the affine removes the ambiguity.

VALIDITY CHECK
  The `whole` partition must reproduce the existing per-category L/R means
  (controls, z>2.33, first session): word 1356 L / 761 R, face 1192 L / 1760 R,
  object 3827 L / 3291 R, house 2438 L / 2938 R. If it does not, the rewrite is
  not faithful and nothing else in the output should be trusted.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets as nl_datasets
from nilearn.image import resample_to_img

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
PROC = Path('/user_data/csimmon2/sym_pt')

CATEGORIES = ['face', 'house', 'object', 'word']
COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
THRESHOLDS = [2.33, 1.96]
CAP_EXCLUDE = ['sub-091', 'sub-095', 'sub-096']
MIN_TOTAL = 10          # matches the manuscript's stability floor

LATERAL_LABELS = [
    'Lateral Occipital Cortex, superior division',
    'Lateral Occipital Cortex, inferior division',
    'Temporal Occipital Fusiform Cortex',
]
MEDIAL_LABELS = [
    'Parahippocampal Gyrus, anterior division',
    'Parahippocampal Gyrus, posterior division',
    'Lingual Gyrus',
    'Temporal Fusiform Cortex, anterior division',
    'Temporal Fusiform Cortex, posterior division',
]


def _load_ages():
    """Scan age per subject from sub_info.csv, if a usable column exists."""
    f = GIT / 'sub_info.csv'
    if not f.exists():
        print('\n(no sub_info.csv -- skipping the age analysis)')
        return None
    s = pd.read_csv(f)
    idc = [c for c in s.columns if c.lower() in
           ('sub', 'sub_clean', 'subject_id', 'sid', 'participant_id')]
    agec = [c for c in s.columns if 'age' in c.lower() and 'surg' not in
            c.lower() and 'seizure' not in c.lower() and 'onset' not in c.lower()]
    if not idc or not agec:
        print('\n(sub_info.csv has no usable id/age column: %s)'
              % s.columns.tolist())
        return None
    print('\nage column used: %s   (id: %s)' % (agec[0], idc[0]))
    out = s[[idc[0], agec[0]]].copy()
    out.columns = ['subject_id', 'age']
    out['subject_id'] = out['subject_id'].astype(str).apply(
        lambda x: x if x.startswith('sub-') else 'sub-%s' % x)
    out = out.dropna().drop_duplicates('subject_id').set_index('subject_id')
    return out


def _mean_x(mask, img):
    ijk = np.argwhere(mask)
    return nib.affines.apply_affine(img.affine, ijk)[:, 0].mean()


def load_verified():
    """Import the verified TFCE module for its subject loader and path builder."""
    cands = sorted((GIT / 'D_liu' / 'verified').glob('02_tfce*.py'))
    if not cands:
        sys.exit('Cannot find D_liu/verified/02_tfce*.py')
    spec = importlib.util.spec_from_file_location('verified_tfce', str(cands[0]))
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    return v


def build_masks(tof_medial=False):
    """Return {partition: {hemi: bool array}} plus the atlas image."""
    lat = list(LATERAL_LABELS)
    med = list(MEDIAL_LABELS)
    if tof_medial:
        lat.remove('Temporal Occipital Fusiform Cortex')
        med.append('Temporal Occipital Fusiform Cortex')

    atlas = nl_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) \
        else nib.load(atlas.maps)
    data = img.get_fdata()
    labels = atlas.labels

    def build(names):
        m = np.zeros(data.shape, dtype=bool)
        for n in names:
            hits = [i for i, l in enumerate(labels) if n in l]
            if not hits:
                print('  WARNING label not found: %s' % n)
                continue
            m |= (data == hits[0])
        return m

    parts = {'lateral': build(lat), 'medial': build(med)}
    parts['whole'] = parts['lateral'] | parts['medial']

    # hemisphere by MNI x from the affine, not array position
    ijk = np.indices(data.shape).reshape(3, -1).T
    x = nib.affines.apply_affine(img.affine, ijk)[:, 0].reshape(data.shape)
    left = x < 0
    right = x > 0

    out = {}
    for p, m in parts.items():
        out[p] = {'l': m & left, 'r': m & right}
    return out, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    ap.add_argument('--no-age-cap', action='store_true')
    ap.add_argument('--under18', action='store_true',
                    help="restrict the age analysis to <18, matching Nordt")
    ap.add_argument('--tof-medial', action='store_true',
                    help='assign Temporal Occipital Fusiform to medial instead')
    args = ap.parse_args()

    v = load_verified()
    masks, atlas_img = build_masks(args.tof_medial)

    print('atlas orientation: %s' % str(nib.aff2axcodes(atlas_img.affine)))
    print('mean MNI x  L=%+.1f  R=%+.1f  (L must be negative)'
          % (_mean_x(masks['whole']['l'], atlas_img),
             _mean_x(masks['whole']['r'], atlas_img)))
    print('partition voxel counts (2 mm MNI):')
    for p in ('whole', 'lateral', 'medial'):
        print('   %-8s L %6d   R %6d'
              % (p, masks[p]['l'].sum(), masks[p]['r'].sum()))

    subjects = v.load_subjects()
    mask_cache = {}
    rows = []
    for sid, info in sorted(subjects.items()):
        if not args.no_age_cap and sid in CAP_EXCLUDE:
            continue
        for cat in CATEGORIES:
            path = v.get_zstat_path(sid, info['session'], info['first_session'],
                                    COPES[cat])
            if not Path(path).exists():
                continue
            zimg = nib.load(str(path))
            z = zimg.get_fdata()
            key = (zimg.shape, zimg.affine.tobytes())
            if key not in mask_cache:
                mn = {}
                for pp in ('whole', 'lateral', 'medial'):
                    mn[pp] = {}
                    for hh in ('l', 'r'):
                        mi = nib.Nifti1Image(
                            masks[pp][hh].astype(np.uint8), atlas_img.affine)
                        mn[pp][hh] = resample_to_img(
                            mi, zimg, interpolation='nearest'
                        ).get_fdata() > 0.5
                mask_cache[key] = mn
            mask_native = mask_cache[key]
            for p in ('whole', 'lateral', 'medial'):
                for hemi in ('l', 'r'):
                    m = mask_native[p][hemi]
                    vals = z[m]
                    vals = vals[np.isfinite(vals)]
                    for thr in THRESHOLDS:
                        rows.append(dict(
                            subject_id=sid, group=info['group'],
                            intact_hemi=info.get('intact_hemi'),
                            partition=p, hemi=hemi, category=cat,
                            threshold=thr,
                            n_selective=int((vals > thr).sum()),
                            n_parcel=int(m.sum())))
    d = pd.DataFrame(rows)
    print('\n%d rows, %d subjects' % (len(d), d.subject_id.nunique()))

    # ---- validity check against the known whole-parcel means ----
    EXPECT = {'word': (1356, 761), 'face': (1192, 1760),
              'object': (3827, 3291), 'house': (2438, 2938)}
    c = d[(d.group == 'control') & (d.partition == 'whole') &
          (d.threshold == 2.33)]
    piv = c.pivot_table(index='category', columns='hemi',
                        values='n_selective', aggfunc='mean')
    print('\nVALIDITY CHECK — whole parcel, controls, z>2.33')
    print('   %-8s %8s %8s   %8s %8s' % ('cat', 'L', 'R', 'expect L', 'expect R'))
    for cat in CATEGORIES:
        if cat not in piv.index:
            continue
        L, R = piv.loc[cat, 'l'], piv.loc[cat, 'r']
        eL, eR = EXPECT[cat]
        ok = abs(L - eL) / max(eL, 1) < .10 and abs(R - eR) / max(eR, 1) < .10
        print('   %-8s %8.0f %8.0f   %8d %8d  %s'
              % (cat, L, R, eL, eR, 'ok' if ok else '<-- MISMATCH'))
    print('   (age cap changes n, so small deviations are expected; a large')
    print('    deviation means the mask rebuild is not faithful.)')

    # ---- LI per partition ----
    for thr in THRESHOLDS:
        print('\n' + '=' * 66)
        print('LATERALITY INDEX  (controls, z > %s)' % thr)
        print('=' * 66)
        print('   %-8s %8s %8s %8s %8s   %s'
              % ('partition', 'word', 'face', 'object', 'house', 'n'))
        for p in ('whole', 'lateral', 'medial'):
            s = d[(d.group == 'control') & (d.partition == p) &
                  (d.threshold == thr)]
            w = s.pivot_table(index=['subject_id', 'category'],
                              columns='hemi', values='n_selective')
            if 'l' not in w or 'r' not in w:
                continue
            w['tot'] = w['l'] + w['r']
            w['LI'] = (w['l'] - w['r']) / w['tot'].replace(0, np.nan)
            li = w['LI'].unstack('category')
            tot = w['tot'].unstack('category')
            li = li[(tot >= MIN_TOTAL).all(axis=1)]
            for kind, vals in (('signed', li), ('|LI|', li.abs())):
                m = vals.mean()
                print('   %-8s %8.3f %8.3f %8.3f %8.3f   %d  %s'
                      % (p,
                         m.get('word', np.nan), m.get('face', np.nan),
                         m.get('object', np.nan), m.get('house', np.nan),
                         len(li), kind))
            # rank check on |LI|
            a = li.abs().mean()
            order = a.sort_values(ascending=False).index.tolist()
            print('        |LI| order: %s   %s'
                  % (' > '.join(order),
                     'MATCHES a priori' if order[:2] == ['word', 'face']
                     else '<-- ORDER CHANGED'))

    # ---- age effect per partition ----
    age = _load_ages()
    if age is not None and args.under18:
        age = age[age['age'] < 18]
    if age is not None:
        from scipy.stats import pearsonr
        for thr in THRESHOLDS:
            print('\n' + '=' * 66)
            print('SIGNED LI vs AGE  (controls, z > %s)' % thr)
            print('=' * 66)
            for p in ('whole', 'lateral', 'medial'):
                s = d[(d.group == 'control') & (d.partition == p) &
                      (d.threshold == thr)]
                w = s.pivot_table(index=['subject_id', 'category'],
                                  columns='hemi', values='n_selective')
                w['tot'] = w['l'] + w['r']
                w['LI'] = (w['l'] - w['r']) / w['tot'].replace(0, np.nan)
                li = w['LI'].unstack('category')
                tot = w['tot'].unstack('category')
                li = li[(tot >= MIN_TOTAL).all(axis=1)]
                li = li.join(age, how='inner')
                out = []
                for cat in CATEGORIES:
                    x = li[[cat, 'age']].dropna()
                    if len(x) < 5:
                        out.append('%s n/a' % cat); continue
                    r, pv = pearsonr(x['age'], x[cat])
                    out.append('%s r=%+.2f p=%.3f%s'
                               % (cat, r, pv, '*' if pv < .05 else ''))
                print('   %-8s n=%d  %s' % (p, len(li), '  '.join(out)))
            print('   (also run with --under18 to match Nordt\'s age range)')

    if args.csv:
        d.to_csv(args.csv, index=False)
        print('\nwrote %s' % args.csv)


if __name__ == '__main__':
    main()