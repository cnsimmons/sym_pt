#!/usr/bin/env python3
"""
otc_rsm_rosenke.py — whole-OTC category RSM, group and individual differences.

METHOD FOLLOWS
  Rosenke, Van den Hurk, Margalit, Op de Beeck, Grill-Spector & Weiner (2020),
  "Extensive individual differences of category information in ventral temporal
  cortex in the congenitally blind", bioRxiv 10.1101/2020.06.14.151092.

  They face the same structural problem as this study: one large cortical
  expanse, four categories, two groups, and a population whose functional
  topography cannot be assumed to align across individuals. Their solution
  avoids both a small ROI and a voxelwise searchlight.

WHAT IT DOES
  1  One LARGE anatomically-defined OTC ROI per hemisphere. No spheres.
  2  Category-vs-all-others selectivity maps (the existing copes 6-9).
  3  Per subject, correlate the SPATIAL PATTERN of category A's selectivity
     against category B's, across every OTC voxel -> a 4x4 RSM.
  4  The 6 off-diagonal entries (odRSM) are the subject's summary. Group tests
     operate on those 6 numbers, not on voxels.

WHY THIS AND NOT THE SEARCHLIGHT TFCE
  A voxelwise searchlight requires the effect to sit at the same MNI coordinates
  across subjects. In a resection cohort it will not, which is why the
  searchlight found one cluster while the ROI RSA found clear effects. Here the
  RSM is computed WITHIN subject over the whole ROI, and only six numbers cross
  subjects, so anatomical correspondence only has to get the ROI boundary right.

THREE ANALYSES, all from Rosenke

  A  GROUP DIFFERENCE. Per pair, patient vs matched-hemisphere control, by
     permutation on the mean difference (10,000 label shuffles). Plus an
     omnibus on odRSM-to-odRSM similarity across groups.

  B  WITHIN-GROUP HETEROGENEITY. Each subject's odRSM correlated against every
     other member of their own group, averaged. Then bootstrapped standard
     deviations compared between groups. In Rosenke this was the headline:
     within-group correlation 0.88 in sighted vs 0.17 in blind. With 12 patients
     differing in etiology, resection extent and age at surgery, a group mean
     may be hiding exactly this.

  B2 PER-CATEGORY DISTINCTIVENESS over the whole OTC parcel. Each category's
     value is the mean of the 3 pair entries containing it — the same
     construction as liu_distinctiveness, but over the OTC parcel rather than a
     7 mm sphere. Higher = more similar to the other categories = LESS distinct.
     Reported with the between-subject SD per category and the patient/control
     SD ratio, which is what answers whether one category carries the excess
     between-patient spread seen in B or whether it is spread across all four.

  C  CROSS-GROUP INDIVIDUAL SIMILARITY. For each patient, is their odRSM more
     similar to other patients or to controls? Rosenke found a subset of blind
     subjects whose RSMs resembled sighted individuals more than other blind
     individuals — a result invisible to any group-mean analysis.

SPLIT-HALF RELIABILITY — the one part that needs run-level data
  Rosenke splits odd vs even runs, so the RSM DIAGONAL is the split-half
  reliability of each category's own selectivity pattern. That is what separates
  "patients' representations are DIFFERENT" from "patients' representations are
  NOISIER", and it is the obvious reviewer question for the rVWFA result.

  Without a data split the diagonal is 1.0 by construction and carries no
  information, and off-diagonal correlations are inflated by shared noise
  equally in both groups. The group COMPARISON remains valid; the absolute
  values are not comparable to Rosenke's.

  The script reports whether run-level copes exist, so the option is visible
  rather than assumed away. It does not require them.

NO HARMONIZATION, DELIBERATELY
  ComBat adjusts each voxel x category feature separately. This measure is a
  correlation of SPATIAL PATTERNS ACROSS categories, so per-category adjustment
  would distort precisely the quantity being measured — the same reason the
  WTA analysis in this project is excluded from harmonization. Scanner is
  reported as a balance check instead, and can be permuted as a nuisance
  variable with --scanner-check.

ORIENTATION
  The OTC masks are ('R','A','S') and the zstat maps are ('L','A','S'): same
  world coordinates, opposite array order along x. Every mask is resampled onto
  the data grid before indexing.

Usage
  python otc_rsm_rosenke.py
  python otc_rsm_rosenke.py --csv otc_rsm.csv
  python otc_rsm_rosenke.py --scanner-check
"""
import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

GIT  = Path('/user_data/csimmon2/git_repos/sym_pt')
PROC = Path('/user_data/csimmon2/sym_pt')
OUT  = PROC / 'group_results' / 'otc_rsm'
SCAN = GIT / 'F_harmonization' / 'sub_info_scanner.csv'

_VTFCE = [GIT / 'D_liu' / 'verified' / '02_tfce_analyses_dontuse_useharmony.py',
          GIT / 'D_liu' / 'verified' / '02_tfce_analyses_not_as_verified.py',
          GIT / 'D_liu' / 'verified' / '02_tfce_analyses.py']
VTFCE = next((p for p in _VTFCE if p.exists()), None)
if VTFCE is None:
    found = sorted((GIT / 'D_liu' / 'verified').glob('02_tfce*.py'))
    sys.exit(f'Cannot find the verified TFCE module. Present: '
             f'{[p.name for p in found] or "none"}')

sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('verified_tfce', str(VTFCE))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

CATEGORIES = v.CATEGORIES                      # face, house, object, word
COPES      = v.COPES_BY_MODE['others']         # cat vs all others: 6,7,8,9
HEMIS      = v.HEMIS
PAIRS      = [(a, b) for i, a in enumerate(CATEGORIES) for b in CATEGORIES[i + 1:]]
N_PERM     = 10000
N_BOOT     = 10000
RNG        = np.random.default_rng(42)


# ── data ─────────────────────────────────────────────────────────────────────

def masked_patterns(sid, info, hemi, mask_path):
    """The four category-vs-others patterns within the OTC mask, as a [4, nvox]
    array. Returns None if any cope is missing."""
    paths = [v.get_zstat_path(sid, info['session'], info['first_session'],
                              COPES[c]) for c in CATEGORIES]
    if not all(p.exists() for p in paths):
        return None, None
    img0 = nib.load(str(paths[0]))
    m = resample_to_img(nib.load(str(mask_path)), img0,
                        interpolation='nearest').get_fdata() > 0.5
    X = np.vstack([nib.load(str(p)).get_fdata()[m] for p in paths])
    good = np.isfinite(X).all(0)
    return X[:, good], int(good.sum())


def subject_rsm(X):
    """4x4 Fisher-z RSM of the spatial patterns. Diagonal is 1 by construction
    without a data split, so only the off-diagonal is used downstream."""
    if X is None or X.shape[1] < 50 or (X.std(axis=1) == 0).any():
        return None
    return np.arctanh(np.clip(np.corrcoef(X), -0.999, 0.999))


def od(rsm):
    """The 6 unique off-diagonal entries, in PAIRS order."""
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    return np.array([rsm[idx[a], idx[b]] for a, b in PAIRS])


# ── statistics ───────────────────────────────────────────────────────────────

def perm_diff(a, b, n=N_PERM):
    """Two-sample permutation on the mean difference (b - a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b]); na = len(a)
    k = 0
    for _ in range(n):
        p = RNG.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


def within_group_similarity(mat):
    """For each row, mean correlation of its odRSM against every other row."""
    n = len(mat)
    out = np.full(n, np.nan)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if not others:
            continue
        out[i] = np.mean([np.corrcoef(mat[i], mat[j])[0, 1] for j in others])
    return out


def boot_sd(vals, n=N_BOOT):
    """Bootstrapped distribution of the sample SD (Rosenke Fig 4C)."""
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.array([np.nan])
    return np.array([np.std(RNG.choice(vals, len(vals), replace=True), ddof=1)
                     for _ in range(n)])


def per_category(mat):
    """Per-category whole-OTC distinctiveness.

    For each category, the mean of the 3 pair entries containing it. Same
    construction as `liu_distinctiveness` in the sphere analyses, but computed
    over the whole OTC parcel. Higher = more similar to the other categories =
    LESS distinct.

    mat is [n_subjects, 6] in PAIRS order; returns [n_subjects, 4] in
    CATEGORIES order.
    """
    mat = np.asarray(mat, float)
    out = np.full((len(mat), len(CATEGORIES)), np.nan)
    for k, c in enumerate(CATEGORIES):
        cols = [i for i, (a, b) in enumerate(PAIRS) if c in (a, b)]
        if len(cols) != 3:
            raise ValueError(f'{c}: expected 3 pairs, got {len(cols)}')
        out[:, k] = np.nanmean(mat[:, cols], axis=1)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    ap.add_argument('--scanner-check', action='store_true',
                    help='also report each pair split by scanner')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    v.OUT_DIR = OUT
    masks = v.build_votc_masks_and_save()
    subjects = v.load_subjects()

    print(f'{len(subjects)} subjects '
          f'({sum(1 for i in subjects.values() if i["group"] == "control")} control, '
          f'{sum(1 for i in subjects.values() if i["group"] == "OTC")} OTC)')

    # is a split-half possible? report, do not assume
    runs = sorted((PROC / next(iter(subjects)) ).glob('ses-*/derivatives/fsl/loc/run-*'))
    nrun = {}
    for sid, info in subjects.items():
        d = PROC / sid / f"ses-{info['session']}" / 'derivatives' / 'fsl' / 'loc'
        nrun[sid] = len(list(d.glob('run-*/1stLevel.feat/stats/zstat1*.nii.gz')))
    have = sum(1 for n in nrun.values() if n >= 2)
    print(f'\nSPLIT-HALF: {have}/{len(nrun)} subjects have >=2 run-level cope maps '
          f'(median {int(np.median(list(nrun.values())))}).')
    print('  Running WITHOUT a split. The RSM diagonal is therefore 1.0 by')
    print('  construction and is not reported; only the 6 off-diagonal entries')
    print('  are used. Group comparisons stay valid; absolute values are not')
    print("  comparable to Rosenke's split-half numbers.")

    # ---- per-subject odRSMs -------------------------------------------------
    rows = []
    for sid, info in sorted(subjects.items()):
        hemis = HEMIS if info['hemi'] is None else [info['hemi']]
        for hemi in hemis:
            X, nv = masked_patterns(sid, info, hemi, masks[hemi])
            rsm = subject_rsm(X)
            if rsm is None:
                print(f'  skip {sid} {hemi}: patterns unusable')
                continue
            e = od(rsm)
            r = dict(subject_id=sid, group=info['group'], hemi=hemi,
                     intact_hemi=info['intact_hemi'], n_vox=nv)
            for (a, b), val in zip(PAIRS, e):
                r[f'{a}-{b}'] = float(val)
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f'\n{len(df)} subject x hemisphere odRSMs   '
          f'(mean {df.n_vox.mean():.0f} voxels)')

    paircols = [f'{a}-{b}' for a, b in PAIRS]
    results = []

    for hemi in HEMIS:
        d = df[df['hemi'] == hemi]
        ctl = d[d['group'] == 'control']
        pt = d[(d['group'] == 'OTC') & (d['intact_hemi'] ==
               ('left' if hemi == 'l' else 'right'))]
        if len(ctl) < 5 or len(pt) < 3:
            print(f'\n[{hemi.upper()}H] too few (ctl={len(ctl)}, pt={len(pt)})')
            continue

        print('\n' + '=' * 74)
        print(f'[{hemi.upper()}H]  {len(ctl)} controls  vs  {len(pt)} '
              f'{"LH" if hemi == "l" else "RH"}-intact patients')
        print('=' * 74)

        # A. per-pair group difference
        print('\nA. GROUP DIFFERENCE per category pair '
              '(Fisher-z pattern correlation across OTC)')
        print(f'   {"pair":14s} {"ctrl":>7s} {"pt":>7s} {"diff":>8s} {"p":>8s}')
        ps = []
        for c in paircols:
            diff, p = perm_diff(ctl[c].values, pt[c].values)
            star = ' *' if (p == p and p < .05) else ''
            print(f'   {c:14s} {ctl[c].mean():7.3f} {pt[c].mean():7.3f} '
                  f'{diff:+8.3f} {p:8.4f}{star}')
            ps.append(p)
            results.append(dict(hemi=hemi, analysis='pair', level=c,
                                ctrl=ctl[c].mean(), pt=pt[c].mean(),
                                diff=diff, p=p, n_ctrl=len(ctl), n_pt=len(pt)))
        ps = np.array(ps, float)
        order = np.argsort(ps)
        q = np.minimum.accumulate((ps[order] * len(ps) /
                                   (np.arange(len(ps)) + 1))[::-1])[::-1]
        qq = np.empty_like(q); qq[order] = np.clip(q, 0, 1)
        print('   BH-FDR across the 6 pairs: ' +
              ', '.join(f'{c.split("-")[0][0]}{c.split("-")[1][0]}={x:.3f}'
                        for c, x in zip(paircols, qq)))

        # B. within-group heterogeneity
        print('\nB. WITHIN-GROUP HETEROGENEITY (Rosenke Fig 4)')
        wc = within_group_similarity(ctl[paircols].values)
        wp = within_group_similarity(pt[paircols].values)
        print(f'   controls: mean r to own group = {np.nanmean(wc):+.3f} '
              f'(SD {np.nanstd(wc, ddof=1):.3f})')
        print(f'   patients: mean r to own group = {np.nanmean(wp):+.3f} '
              f'(SD {np.nanstd(wp, ddof=1):.3f})')
        dmean, pmean = perm_diff(wc, wp)
        print(f'   difference in mean similarity: {dmean:+.3f}, p = {pmean:.4f}'
              + ('  *' if pmean == pmean and pmean < .05 else ''))
        bc, bp = boot_sd(wc), boot_sd(wp)
        pspread = (np.sum(bp <= bc) + 1) / (len(bc) + 1)
        print(f'   bootstrapped SD: controls {np.nanmean(bc):.3f}, '
              f'patients {np.nanmean(bp):.3f}, p = {pspread:.4f}'
              + ('  *' if pspread < .05 else ''))
        print('   Higher patient SD = the group mean is hiding real '
              'between-patient variability.')
        results.append(dict(hemi=hemi, analysis='heterogeneity', level='mean_r',
                            ctrl=float(np.nanmean(wc)), pt=float(np.nanmean(wp)),
                            diff=dmean, p=pmean, n_ctrl=len(ctl), n_pt=len(pt)))
        results.append(dict(hemi=hemi, analysis='heterogeneity', level='boot_sd',
                            ctrl=float(np.nanmean(bc)), pt=float(np.nanmean(bp)),
                            diff=float(np.nanmean(bp) - np.nanmean(bc)),
                            p=float(pspread), n_ctrl=len(ctl), n_pt=len(pt)))

        # B2. per-category distinctiveness over the whole OTC parcel
        print('\nB2. PER-CATEGORY WHOLE-OTC DISTINCTIVENESS')
        print('    mean of the 3 pairs containing each category; '
              'higher = LESS distinct')
        pc_c = per_category(ctl[paircols].values)
        pc_p = per_category(pt[paircols].values)
        print(f'   {"category":9s} {"ctrl":>7s} {"pt":>7s} {"diff":>8s} '
              f'{"p":>8s}  {"SDctrl":>7s} {"SDpt":>7s} {"ratio":>6s} '
              f'{"p_sd":>7s}')
        ps2 = []
        for k, c in enumerate(CATEGORIES):
            a, b = pc_c[:, k], pc_p[:, k]
            diff, p = perm_diff(a, b)
            sd_c = float(np.nanstd(a, ddof=1))
            sd_p = float(np.nanstd(b, ddof=1))
            bc2, bp2 = boot_sd(a), boot_sd(b)
            p_sd = float((np.sum(bp2 <= bc2) + 1) / (len(bc2) + 1))
            ratio = sd_p / sd_c if sd_c > 0 else np.nan
            print(f'   {c:9s} {np.nanmean(a):7.3f} {np.nanmean(b):7.3f} '
                  f'{diff:+8.3f} {p:8.4f}'
                  + ('*' if p == p and p < .05 else ' ')
                  + f' {sd_c:7.3f} {sd_p:7.3f} {ratio:6.2f} {p_sd:7.4f}'
                  + ('*' if p_sd < .05 else ''))
            ps2.append(p)
            results.append(dict(hemi=hemi, analysis='per_category', level=c,
                                ctrl=float(np.nanmean(a)),
                                pt=float(np.nanmean(b)),
                                diff=diff, p=p, sd_ctrl=sd_c, sd_pt=sd_p,
                                sd_ratio=float(ratio), p_sd=p_sd,
                                n_ctrl=len(ctl), n_pt=len(pt)))
        ps2 = np.array(ps2, float)
        o2 = np.argsort(ps2)
        q2 = np.minimum.accumulate((ps2[o2] * len(ps2) /
                                   (np.arange(len(ps2)) + 1))[::-1])[::-1]
        qq2 = np.empty_like(q2); qq2[o2] = np.clip(q2, 0, 1)
        print('   BH-FDR across the 4 categories: ' +
              ', '.join(f'{c}={x:.3f}' for c, x in zip(CATEGORIES, qq2)))
        print('   ratio > 1 = that category is more variable across patients '
              'than across controls.')
        print('   These 4 values are NOT independent: each pair enters two '
              'categories.')

        # C. cross-group individual similarity
        print('\nC. CROSS-GROUP INDIVIDUAL SIMILARITY (Rosenke Fig 5)')
        print('   per patient: mean r to other patients vs mean r to controls')
        C, P = ctl[paircols].values, pt[paircols].values
        pids = pt['subject_id'].values
        flipped = []
        for i, sid in enumerate(pids):
            own = [np.corrcoef(P[i], P[j])[0, 1]
                   for j in range(len(P)) if j != i]
            oth = [np.corrcoef(P[i], C[j])[0, 1] for j in range(len(C))]
            mark = '  <- closer to CONTROLS' if np.mean(oth) > np.mean(own) else ''
            if mark:
                flipped.append(sid)
            print(f'     {sid:10s} own {np.mean(own):+.3f}   '
                  f'ctrl {np.mean(oth):+.3f}{mark}')
            results.append(dict(hemi=hemi, analysis='cross_group', level=sid,
                                ctrl=float(np.mean(oth)), pt=float(np.mean(own)),
                                diff=float(np.mean(oth) - np.mean(own)),
                                p=np.nan, n_ctrl=len(ctl), n_pt=len(pt)))
        print(f'   {len(flipped)}/{len(pids)} patients resemble controls more '
              'than their own group.')

        # scanner balance
        if args.scanner_check and SCAN.exists():
            sc = pd.read_csv(SCAN)
            mm = d.merge(sc[['sub', 'scanner']].drop_duplicates('sub'),
                         left_on='subject_id', right_on='sub', how='left')
            print('\n   scanner balance: ' +
                  str(mm.groupby(['group', 'scanner']).size().to_dict()))

    if args.csv:
        pd.DataFrame(results).to_csv(args.csv, index=False)
        df.to_csv(str(args.csv).replace('.csv', '_persubject.csv'), index=False)
        print(f'\nwrote {args.csv} and *_persubject.csv')


if __name__ == '__main__':
    main()
