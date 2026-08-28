"""RSM DIAGONAL — per-category split-half reliability of each category's own
spatial selectivity pattern, plus reliability of the odRSM vector. Both
computed in each subject's NATIVE space.

THE DIAGONAL IS THE PER-CATEGORY WHOLE-OTC MEASURE.
  For each category, correlate its spatial pattern from odd runs against the
  same category's pattern from even runs, across every OTC voxel. Four numbers
  per subject, one per category, INDEPENDENT of each other -- no pairs, no
  collapse of pair values. This is Rosenke's own primary measure ("the
  reliability of distributed category selectivity"), and it is the whole-OTC
  analogue of asking whether a category is coherently represented at all.

  Lower reliability = that category's selectivity pattern is not reproducible
  within subject = it is not stably represented across the parcel.

  Contrast with the off-diagonal (odRSM): those 6 entries are BETWEEN-category
  pattern correlations, so every one of them involves two categories and no
  single entry can be attributed to one category. The diagonal has no such
  problem.

Why native space: reliability is a within-subject quantity, so it needs no
cross-subject voxel correspondence. The MNI OTC mask is pulled into each
subject's own functional space with FSL's standard2example_func.mat, which
sidesteps the patients' mirror-flip registration entirely.

Reliability is the correlation between the 6-entry odRSM vector from odd runs
and the same vector from even runs -- the reliability of exactly the quantity
the group statistic is computed on.

Session rule: patients last session, controls first session.

Reports per-subject reliability and the disattenuated within-group similarity,
where each pairwise inter-subject correlation is divided by the geometric mean
of the two subjects' reliabilities.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PROC = Path('/user_data/csimmon2/sym_pt')
GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
PERSUB = GIT / 'C_results' / 'rosenke_persubject.csv'

CATEGORIES = ['face', 'house', 'object', 'word']
COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}
PAIRS = [(a, b) for i, a in enumerate(CATEGORIES) for b in CATEGORIES[i + 1:]]
PAIRCOLS = ['%s-%s' % (a, b) for a, b in PAIRS]
CAP_EXCLUDE = ['sub-091', 'sub-095', 'sub-096']
MIN_REL = 0.1


def sessions(sid):
    return sorted(d.name.replace('ses-', '')
                  for d in (PROC / sid).glob('ses-*')
                  if (d / 'derivatives' / 'fsl' / 'loc').is_dir())


def pick_session(sid, group):
    ses = sessions(sid)
    if not ses:
        return None
    return ses[-1] if group == 'OTC' else ses[0]


def run_dirs(sid, ses):
    base = PROC / sid / ('ses-%s' % ses) / 'derivatives' / 'fsl' / 'loc'
    out = []
    for d in sorted(base.glob('run-*')):
        stats = d / '1stLevel.feat' / 'stats'
        if all((stats / ('zstat%d.nii.gz' % COPES[c])).exists()
               for c in CATEGORIES):
            out.append(d)
    return out


def mask_to_native(mask_mni, rundir, tmp):
    """Resample the MNI OTC mask into this run's functional space."""
    reg = rundir / '1stLevel.feat' / 'reg'
    xfm = reg / 'standard2example_func.mat'
    ref = reg / 'example_func.nii.gz'
    if not (xfm.exists() and ref.exists()):
        return None
    out = Path(tmp) / 'mask_native.nii.gz'
    subprocess.run(['flirt', '-in', str(mask_mni), '-ref', str(ref),
                    '-applyxfm', '-init', str(xfm),
                    '-interp', 'nearestneighbour', '-out', str(out)],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return nib.load(str(out)).get_fdata() > 0.5


def half_patterns(rundirs, m):
    """Mean zstat pattern per category across a set of runs, within mask m."""
    rows = []
    for c in CATEGORIES:
        acc = []
        for d in rundirs:
            p = d / '1stLevel.feat' / 'stats' / ('zstat%d.nii.gz' % COPES[c])
            acc.append(nib.load(str(p)).get_fdata()[m])
        rows.append(np.mean(acc, axis=0))
    X = np.vstack(rows)
    good = np.isfinite(X).all(0)
    return X[:, good]


def od_vector(X):
    if X.shape[1] < 50 or (X.std(axis=1) == 0).any():
        return None
    R = np.arctanh(np.clip(np.corrcoef(X), -0.999, 0.999))
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    return np.array([R[idx[a], idx[b]] for a, b in PAIRS])


def reliability(sid, group, mask_mni):
    ses = pick_session(sid, group)
    if ses is None:
        return np.nan, 0, np.full(len(CATEGORIES), np.nan)
    rd = run_dirs(sid, ses)
    if len(rd) < 2:
        return np.nan, len(rd), np.full(len(CATEGORIES), np.nan)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            m = mask_to_native(mask_mni, rd[0], tmp)
            if m is None or m.sum() < 50:
                return np.nan, len(rd), np.full(len(CATEGORIES), np.nan)
            odd = half_patterns(rd[0::2], m)
            even = half_patterns(rd[1::2], m)
    except Exception as exc:
        print('   !! %s: %s' % (sid, exc))
        return np.nan, len(rd), np.full(len(CATEGORIES), np.nan)
    # per-category diagonal: same category, odd vs even runs
    diag = np.full(len(CATEGORIES), np.nan)
    for k in range(len(CATEGORIES)):
        u, w = odd[k], even[k]
        if np.std(u) > 0 and np.std(w) > 0:
            diag[k] = float(np.corrcoef(u, w)[0, 1])

    a, b = od_vector(odd), od_vector(even)
    vec = np.nan if (a is None or b is None) else float(np.corrcoef(a, b)[0, 1])
    return vec, len(rd), diag


def within_group_similarity(mat, rel=None):
    """Leave-self-out mean pairwise correlation of odRSM vectors. If rel is
    given, each pairwise correlation is divided by the geometric mean of the
    two subjects' reliabilities (disattenuation)."""
    n = len(mat)
    out = np.full(n, np.nan)
    for i in range(n):
        vals = []
        for j in range(n):
            if j == i:
                continue
            r = np.corrcoef(mat[i], mat[j])[0, 1]
            if rel is not None:
                denom = np.sqrt(max(rel[i], 0.0) * max(rel[j], 0.0))
                if not np.isfinite(denom) or denom < MIN_REL:
                    continue
                r = r / denom
            vals.append(r)
        if vals:
            out[i] = np.mean(vals)
    return out


def perm_diff(a, b, n=10000, rng=None):
    """Two-sample permutation on the mean difference (b - a)."""
    rng = rng or np.random.default_rng(42)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    k = 0
    for _ in range(n):
        p = rng.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask-l', required=True)
    ap.add_argument('--mask-r', required=True)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--age-cap', action='store_true')
    args = ap.parse_args()

    if not PERSUB.exists():
        sys.exit('not found: %s\nRun otc_rsm_rosenke.py --csv first.' % PERSUB)
    df = pd.read_csv(PERSUB)
    if args.age_cap:
        df = df[~df['subject_id'].isin(CAP_EXCLUDE)].copy()
        print('age cap applied, %d subject x hemisphere rows' % len(df))

    masks = {'l': args.mask_l, 'r': args.mask_r}
    results = []

    # reliability is per subject x session, not per hemisphere -- compute once
    print('\nSPLIT-HALF RELIABILITY of the odRSM vector (native space)')
    print('   %-10s %-8s %5s %7s   %s'
          % ('subject', 'grp', 'runs', 'vec',
             ' '.join('%6s' % c[:6] for c in CATEGORIES)))
    rel_rows = []
    for sid, sub in df.groupby('subject_id'):
        group = sub['group'].iloc[0]
        hemi = sub['hemi'].iloc[0]
        r, nrun, diag = reliability(sid, group, masks[hemi])
        shown = '%+7.3f' % r if np.isfinite(r) else '    nan'
        dshown = ' '.join(('%+6.3f' % x) if np.isfinite(x) else '   nan'
                          for x in diag)
        print('   %-10s %-8s %5d %s   %s'
              % (sid, group, nrun, shown, dshown))
        row = dict(subject_id=sid, group=group, n_run=nrun, reliability=r)
        for k, c in enumerate(CATEGORIES):
            row['diag_' + c] = diag[k]
        rel_rows.append(row)
    rel = pd.DataFrame(rel_rows)
    diagcols = ['diag_' + c for c in CATEGORIES]
    df = df.merge(rel[['subject_id', 'reliability', 'n_run'] + diagcols],
                  on='subject_id', how='left')

    n_ok = int(np.isfinite(df['reliability']).sum())
    print('\n%d/%d rows have a usable reliability estimate' % (n_ok, len(df)))

    for hemi in ['l', 'r']:
        d = df[df['hemi'] == hemi]
        ctl = d[d['group'] == 'control']
        pt = d[(d['group'] == 'OTC') &
               (d['intact_hemi'] == ('left' if hemi == 'l' else 'right'))]
        if len(ctl) < 5 or len(pt) < 3:
            print('\n[%sH] too few (ctl=%d, pt=%d)'
                  % (hemi.upper(), len(ctl), len(pt)))
            continue

        print('\n' + '=' * 70)
        print('[%sH]  %d controls  vs  %d patients'
              % (hemi.upper(), len(ctl), len(pt)))
        print('=' * 70)

        rc = ctl['reliability'].values
        rp = pt['reliability'].values
        drel, prel = perm_diff(rc, rp)
        print('   mean reliability: controls %+.3f (SD %.3f), '
              'patients %+.3f (SD %.3f)'
              % (np.nanmean(rc), np.nanstd(rc, ddof=1),
                 np.nanmean(rp), np.nanstd(rp, ddof=1)))
        print('   group difference in reliability: %+.3f, p = %.4f%s'
              % (drel, prel, '  *' if prel == prel and prel < .05 else ''))
        print('   >> If patient reliability is much lower, the raw consistency')
        print('      gap is partly a reliability artifact (Byrge et al. 2015).')

        print('\n   RSM DIAGONAL — per-category pattern reliability '
              '(odd vs even runs)')
        print('   lower = that category is less stably represented '
              'across the parcel')
        print('   %-8s %8s %8s %8s %8s  %8s %8s %6s'
              % ('category', 'ctrl', 'pt', 'diff', 'p', 'SDctrl', 'SDpt',
                 'ratio'))
        pd_ = []
        for c in CATEGORIES:
            a = ctl['diag_' + c].values
            b = pt['diag_' + c].values
            diff, p = perm_diff(a, b)
            sa = float(np.nanstd(a, ddof=1))
            sb = float(np.nanstd(b, ddof=1))
            ratio = sb / sa if sa > 0 else np.nan
            print('   %-8s %+8.3f %+8.3f %+8.3f %8.4f%s %8.3f %8.3f %6.2f'
                  % (c, np.nanmean(a), np.nanmean(b), diff, p,
                     '*' if p == p and p < .05 else ' ', sa, sb, ratio))
            pd_.append(p)
            results.append(dict(hemi=hemi, analysis='diagonal', level=c,
                                ctrl=float(np.nanmean(a)),
                                pt=float(np.nanmean(b)),
                                diff=diff, p=p, sd_ctrl=sa, sd_pt=sb,
                                sd_ratio=float(ratio),
                                n_ctrl=len(ctl), n_pt=len(pt)))
        pd_ = np.array(pd_, float)
        o = np.argsort(pd_)
        qd = np.minimum.accumulate((pd_[o] * len(pd_) /
                                    (np.arange(len(pd_)) + 1))[::-1])[::-1]
        qqd = np.empty_like(qd); qqd[o] = np.clip(qd, 0, 1)
        print('   BH-FDR across the 4 categories: ' +
              ', '.join('%s=%.3f' % (c, x) for c, x in zip(CATEGORIES, qqd)))
        print('   These 4 values ARE independent — no pair enters two of them.')

        C, P = ctl[PAIRCOLS].values, pt[PAIRCOLS].values
        raw_c = np.nanmean(within_group_similarity(C))
        raw_p = np.nanmean(within_group_similarity(P))
        cor_c = np.nanmean(within_group_similarity(C, rc))
        cor_p = np.nanmean(within_group_similarity(P, rp))

        print('\n   %-22s %8s %8s %8s' % ('', 'ctrl', 'pt', 'gap'))
        print('   %-22s %+8.3f %+8.3f %+8.3f'
              % ('raw consistency', raw_c, raw_p, raw_p - raw_c))
        print('   %-22s %+8.3f %+8.3f %+8.3f'
              % ('disattenuated', cor_c, cor_p, cor_p - cor_c))
        print('   >> If the disattenuated gap stays large and negative, the')
        print('      heterogeneity is not explained by data reliability.')

    if args.csv:
        df.to_csv(args.csv, index=False)
        if results:
            rp = str(args.csv).replace('.csv', '_groupstats.csv')
            pd.DataFrame(results).to_csv(rp, index=False)
            print('\nwrote %s and %s' % (args.csv, rp))
        else:
            print('\nwrote %s' % args.csv)


if __name__ == '__main__':
    main()
