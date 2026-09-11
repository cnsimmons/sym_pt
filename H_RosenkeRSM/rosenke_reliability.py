"""Split-half reliability of the odRSM vector, computed in each subject's
NATIVE space.

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
        return np.nan, 0
    rd = run_dirs(sid, ses)
    if len(rd) < 2:
        return np.nan, len(rd)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            m = mask_to_native(mask_mni, rd[0], tmp)
            if m is None or m.sum() < 50:
                return np.nan, len(rd)
            odd = half_patterns(rd[0::2], m)
            even = half_patterns(rd[1::2], m)
    except Exception as exc:
        print('   !! %s: %s' % (sid, exc))
        return np.nan, len(rd)
    a, b = od_vector(odd), od_vector(even)
    if a is None or b is None:
        return np.nan, len(rd)
    return float(np.corrcoef(a, b)[0, 1]), len(rd)


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

    # reliability is per subject x session, not per hemisphere -- compute once
    print('\nSPLIT-HALF RELIABILITY of the odRSM vector (native space)')
    print('   %-10s %-8s %5s %7s' % ('subject', 'grp', 'runs', 'rel'))
    rel_rows = []
    for sid, sub in df.groupby('subject_id'):
        group = sub['group'].iloc[0]
        hemi = sub['hemi'].iloc[0]
        r, nrun = reliability(sid, group, masks[hemi])
        shown = '%+7.3f' % r if np.isfinite(r) else '    nan'
        print('   %-10s %-8s %5d %s' % (sid, group, nrun, shown))
        rel_rows.append(dict(subject_id=sid, group=group,
                             n_run=nrun, reliability=r))
    rel = pd.DataFrame(rel_rows)
    df = df.merge(rel[['subject_id', 'reliability', 'n_run']],
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
        print('\nwrote %s' % args.csv)


if __name__ == '__main__':
    main()
