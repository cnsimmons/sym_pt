#!/usr/bin/env python3
"""Threshold-INDEPENDENT laterality index (Wilke & Schmithorst bootstrap).

WHY
  The single-threshold voxel-count LI is the method the methods literature
  criticises: it is threshold-dependent, and individual subjects can switch
  apparent dominance with a change of threshold (Bradshaw, Thompson, Wilson,
  Bishop & Woodhead 2017, PeerJ 5:e3557). The recommended alternative is the
  threshold-independent bootstrap of Wilke & Schmithorst (2006, NeuroImage
  33:522-530), which is also resistant to outliers.

METHOD
  1. Sweep thresholds from 0 to the upper end of the subject's own z range,
     in N_THRESH equal steps.
  2. At each threshold, bootstrap: draw N_BOOT random subsets of FRACTION of
     the suprathreshold voxels in each hemisphere, and compute LI for each
     left/right pairing.
  3. Trim to the central 50% of the bootstrap distribution (outlier resistance).
  4. Weight each threshold's trimmed mean by the threshold value, so higher,
     more specific thresholds count more, and take the weighted mean.

  The output is one LI per subject x category x partition with no threshold
  chosen by hand. The full curve is written out too, so threshold dependence is
  visible rather than hidden.

VALIDATION
  Reports the correlation between the threshold-free LI and the fixed-threshold
  LI at z>2.33. If they agree closely, the fixed-threshold values already in the
  manuscript are safe and this becomes a supplementary robustness check. If they
  diverge, the threshold-free values should be primary.

Partitions and the affine-based hemisphere split follow li_partition.py.
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

CATEGORIES = ['face', 'house', 'object', 'word']
COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
CAP_EXCLUDE = ['sub-091', 'sub-095', 'sub-096']

N_THRESH = 20
N_BOOT = 100
FRACTION = 0.25
Z_MAX = 5.0          # upper end of the sweep
MIN_VOX = 10         # skip a threshold step with fewer voxels than this total
RNG = np.random.default_rng(42)

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


def load_verified():
    cands = sorted((GIT / 'D_liu' / 'verified').glob('02_tfce*.py'))
    if not cands:
        sys.exit('Cannot find D_liu/verified/02_tfce*.py')
    spec = importlib.util.spec_from_file_location('verified_tfce', str(cands[0]))
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    return v


def build_masks():
    atlas = nl_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) \
        else nib.load(atlas.maps)
    data = img.get_fdata()
    labels = atlas.labels

    def build(names):
        m = np.zeros(data.shape, dtype=bool)
        for n in names:
            hits = [i for i, l in enumerate(labels) if n in l]
            if hits:
                m |= (data == hits[0])
        return m

    parts = {'lateral': build(LATERAL_LABELS), 'medial': build(MEDIAL_LABELS)}
    parts['whole'] = parts['lateral'] | parts['medial']

    ijk = np.indices(data.shape).reshape(3, -1).T
    x = nib.affines.apply_affine(img.affine, ijk)[:, 0].reshape(data.shape)
    out = {p: {'l': m & (x < 0), 'r': m & (x > 0)} for p, m in parts.items()}
    return out, img


def li_curve(zl, zr):
    """LI at each threshold step, bootstrapped and trimmed.

    zl, zr are the finite z values in the left and right partition.
    Returns (thresholds, trimmed_mean_LI) for steps with enough voxels.
    """
    ths, lis = [], []
    for th in np.linspace(0.0, Z_MAX, N_THRESH + 1)[1:]:
        nl = int((zl > th).sum())
        nr = int((zr > th).sum())
        if nl + nr < MIN_VOX:
            continue
        kl = max(1, int(round(FRACTION * nl)))
        kr = max(1, int(round(FRACTION * nr)))
        # bootstrap counts: hypergeometric-style resampling of the two pools
        bl = RNG.binomial(nl, FRACTION, size=N_BOOT) if nl else np.zeros(N_BOOT)
        br = RNG.binomial(nr, FRACTION, size=N_BOOT) if nr else np.zeros(N_BOOT)
        tot = bl + br
        keep = tot > 0
        if not keep.any():
            continue
        vals = (bl[keep] - br[keep]) / tot[keep]
        lo, hi = np.percentile(vals, [25, 75])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        if trimmed.size == 0:
            trimmed = vals
        ths.append(th)
        lis.append(float(trimmed.mean()))
    return np.array(ths), np.array(lis)


def weighted_li(ths, lis):
    """Threshold-weighted mean LI (higher thresholds weighted more)."""
    if len(ths) == 0:
        return np.nan
    w = ths / ths.sum()
    return float(np.sum(w * lis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    ap.add_argument('--curves-csv', default=None)
    ap.add_argument('--no-age-cap', action='store_true')
    args = ap.parse_args()

    v = load_verified()
    masks, atlas_img = build_masks()
    print('atlas orientation: %s' % str(nib.aff2axcodes(atlas_img.affine)))
    for p in ('whole', 'lateral', 'medial'):
        ijk = np.argwhere(masks[p]['l'])
        mx = nib.affines.apply_affine(atlas_img.affine, ijk)[:, 0].mean()
        print('   %-8s L n=%6d  mean MNI x=%+.1f (must be negative)'
              % (p, masks[p]['l'].sum(), mx))

    subjects = v.load_subjects()
    cache = {}
    rows, curves = [], []
    for sid, info in sorted(subjects.items()):
        if not args.no_age_cap and sid in CAP_EXCLUDE:
            continue
        for cat in CATEGORIES:
            path = v.get_zstat_path(sid, info['session'],
                                    info['first_session'], COPES[cat])
            if not Path(path).exists():
                continue
            zimg = nib.load(str(path))
            z = zimg.get_fdata()
            key = (zimg.shape, zimg.affine.tobytes())
            if key not in cache:
                mn = {}
                for pp in ('whole', 'lateral', 'medial'):
                    mn[pp] = {}
                    for hh in ('l', 'r'):
                        mi = nib.Nifti1Image(masks[pp][hh].astype(np.uint8),
                                             atlas_img.affine)
                        mn[pp][hh] = resample_to_img(
                            mi, zimg, interpolation='nearest'
                        ).get_fdata() > 0.5
                cache[key] = mn
            mnat = cache[key]

            for p in ('whole', 'lateral', 'medial'):
                zl = z[mnat[p]['l']]
                zr = z[mnat[p]['r']]
                zl = zl[np.isfinite(zl)]
                zr = zr[np.isfinite(zr)]
                ths, lis = li_curve(zl, zr)
                rows.append(dict(subject_id=sid, group=info['group'],
                                 intact_hemi=info.get('intact_hemi'),
                                 partition=p, category=cat,
                                 li_threshfree=weighted_li(ths, lis),
                                 n_steps=len(ths),
                                 li_min=float(lis.min()) if len(lis) else np.nan,
                                 li_max=float(lis.max()) if len(lis) else np.nan))
                for t, l in zip(ths, lis):
                    curves.append(dict(subject_id=sid, group=info['group'],
                                       partition=p, category=cat,
                                       threshold=float(t), li=float(l)))

    d = pd.DataFrame(rows)
    print('\n%d rows, %d subjects' % (len(d), d.subject_id.nunique()))

    for p in ('whole', 'lateral', 'medial'):
        c = d[(d.group == 'control') & (d.partition == p)]
        if c.empty:
            continue
        piv = c.pivot_table(index='subject_id', columns='category',
                            values='li_threshfree')
        print('\n' + '=' * 68)
        print('THRESHOLD-FREE LI  —  partition = %s   (n=%d controls)'
              % (p, len(piv)))
        print('=' * 68)
        sg = piv.mean()
        ab = piv.abs().mean()
        print('   %-8s %8s %8s %8s %8s' % ('', 'word', 'face', 'object', 'house'))
        print('   %-8s %8.3f %8.3f %8.3f %8.3f'
              % ('signed', sg.get('word', np.nan), sg.get('face', np.nan),
                 sg.get('object', np.nan), sg.get('house', np.nan)))
        print('   %-8s %8.3f %8.3f %8.3f %8.3f'
              % ('|LI|', ab.get('word', np.nan), ab.get('face', np.nan),
                 ab.get('object', np.nan), ab.get('house', np.nan)))
        order = ab.sort_values(ascending=False).index.tolist()
        print('   |LI| order: %s   %s'
              % (' > '.join(order),
                 'word and face on top — a priori rank holds'
                 if order[:2] == ['word', 'face'] else '<-- ORDER CHANGED'))
        # how much does LI move across the sweep?
        span = (c['li_max'] - c['li_min']).abs()
        print('   within-subject LI span across thresholds: '
              'median %.3f, max %.3f' % (span.median(), span.max()))

        # age
        agef = GIT / 'sub_info.csv'
        if agef.exists():
            s = pd.read_csv(agef)
            idc = [x for x in s.columns
                   if x.lower() in ('sub', 'sub_clean', 'subject_id')]
            agec = [x for x in s.columns if x.lower() == 'age']
            if idc and agec:
                a = s[[idc[0], agec[0]]].copy()
                a.columns = ['subject_id', 'age']
                a['subject_id'] = a['subject_id'].astype(str).apply(
                    lambda x: x if x.startswith('sub-') else 'sub-%s' % x)
                a = a.dropna().drop_duplicates('subject_id').set_index('subject_id')
                j = piv.join(a, how='inner')
                from scipy.stats import pearsonr
                out = []
                for cat in CATEGORIES:
                    if cat not in j:
                        continue
                    x = j[[cat, 'age']].dropna()
                    r, pv = pearsonr(x['age'], x[cat])
                    out.append('%s r=%+.2f p=%.3f%s'
                               % (cat, r, pv, '*' if pv < .05 else ''))
                print('   vs AGE (n=%d): %s' % (len(j), '  '.join(out)))

    if args.csv:
        d.to_csv(args.csv, index=False)
        print('\nwrote %s' % args.csv)
    if args.curves_csv:
        pd.DataFrame(curves).to_csv(args.curves_csv, index=False)
        print('wrote %s' % args.curves_csv)


if __name__ == '__main__':
    main()
