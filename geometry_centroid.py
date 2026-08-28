#!/usr/bin/env python3
"""Which hemisphere's representational geometry does each preserved hemisphere
resemble?

THE TEST
  Controls give two centroids per ROI: the mean 6-pair geometry vector of their
  LEFT hemispheres and of their RIGHT hemispheres. Each patient contributes one
  6-pair vector from their preserved hemisphere. Ask which centroid it is closer
  to. This is a per-subject classification, not a group omnibus -- if the
  hemispheres have distinct representational geometries and patients' preserved
  hemisphere has shifted, that shows up as patients being assigned to the wrong
  side.

VALIDATION FIRST (this is the part that can fail)
  Before classifying patients, classify CONTROLS against their own centroids,
  leave-one-out. If a control's left hemisphere is not reliably closer to the
  left centroid than the right centroid, then the two centroids are not
  separable and nothing downstream means anything. That accuracy is the ceiling
  for the patient test.

  Reported for Euclidean distance and for correlation, since a shift in overall
  level and a shift in pattern shape are different claims.

SIGN
  fisher_r is a raw similarity (higher = more similar = less separated). Signs
  are therefore the reverse of the "oriented" scale used in marlene_table. For a
  distance test the orientation is irrelevant as long as it is consistent.

SESSION RULE
  Patients last session, controls first session. Age cap by subject id.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
RSA = GIT / 'D_liu' / 'rsa_v1_harmonized.csv'

ROIS = ['face_FFA', 'house_PPA', 'object_LOC', 'word_VWFA']
PAIRS = ['face-house', 'face-object', 'face-word',
         'house-object', 'house-word', 'object-word']
CAP_EXCLUDE = ['sub-091', 'sub-095', 'sub-096']
N_PERM = 10000
RNG = np.random.default_rng(42)


def load(age_cap=True):
    d = pd.read_csv(RSA)
    d = d[d['pair'].notna() & d['category'].isin(ROIS)]
    if age_cap:
        d = d[~d['subject_id'].isin(CAP_EXCLUDE)]
    # session rule
    keep = []
    for (sid, hemi), g in d.groupby(['subject_id', 'hemi']):
        grp = g['group'].iloc[0]
        s = g['ses_num'].max() if grp == 'OTC' else g['ses_num'].min()
        keep.append(g[g['ses_num'] == s])
    return pd.concat(keep, ignore_index=True)


def vectors(d, roi, hemi, group, intact=None):
    """One row per subject, columns = the 6 pairs, in PAIRS order."""
    s = d[(d['category'] == roi) & (d['hemi'] == hemi) & (d['group'] == group)]
    if intact is not None:
        s = s[s['intact_hemi'] == intact]
    w = s.pivot_table(index='subject_id', columns='pair', values='fisher_r')
    w = w.reindex(columns=PAIRS)
    return w.dropna()


def closer_to(v, cL, cR, metric):
    """Return +1 if v is closer to the LEFT centroid, -1 if closer to RIGHT."""
    if metric == 'euclid':
        dL = np.linalg.norm(v - cL)
        dR = np.linalg.norm(v - cR)
        return 1 if dL < dR else -1
    rL = np.corrcoef(v, cL)[0, 1]
    rR = np.corrcoef(v, cR)[0, 1]
    return 1 if rL > rR else -1


def binom_p(k, n):
    """Two-sided exact binomial against p=0.5, without scipy."""
    from math import comb
    if n == 0:
        return np.nan
    probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= obs + 1e-12)))


def validate(L, R, metric):
    """Leave-one-out classification of control hemispheres."""
    ok = 0
    tot = 0
    ids = L.index.intersection(R.index)
    for sid in ids:
        for true, W, other in ((1, L, R), (-1, R, L)):
            v = W.loc[sid].values
            cSelf = W.drop(index=sid).values.mean(0)
            cOther = other.drop(index=sid, errors='ignore').values.mean(0)
            cL, cR = (cSelf, cOther) if true == 1 else (cOther, cSelf)
            got = closer_to(v, cL, cR, metric)
            ok += int(got == true)
            tot += 1
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-age-cap', action='store_true')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    d = load(age_cap=not args.no_age_cap)
    print('%d rows, %d subjects' % (len(d), d['subject_id'].nunique()))

    rows = []
    for metric in ('euclid', 'corr'):
        print('\n' + '=' * 74)
        print('METRIC: %s' % metric)
        print('=' * 74)

        for roi in ROIS + ['POOLED']:
            if roi == 'POOLED':
                # concatenate the 4 ROIs into one 24-length vector per subject
                def pooled(group, hemi, intact=None):
                    parts = []
                    for r in ROIS:
                        w = vectors(d, r, hemi, group, intact)
                        w.columns = ['%s|%s' % (r, c) for c in w.columns]
                        parts.append(w)
                    out = parts[0]
                    for p in parts[1:]:
                        out = out.join(p, how='inner')
                    return out
                L = pooled('control', 'l')
                R = pooled('control', 'r')
                PL = pooled('OTC', 'l', 'left')
                PR = pooled('OTC', 'r', 'right')
            else:
                L = vectors(d, roi, 'l', 'control')
                R = vectors(d, roi, 'r', 'control')
                PL = vectors(d, roi, 'l', 'OTC', 'left')
                PR = vectors(d, roi, 'r', 'OTC', 'right')

            vok, vtot = validate(L, R, metric)
            vacc = vok / vtot if vtot else np.nan
            vp = binom_p(vok, vtot)
            flag = '' if vp == vp and vp < .05 else '   <-- CENTROIDS NOT SEPARABLE'
            print('\n%s   ctrl L n=%d, R n=%d' % (roi, len(L), len(R)))
            print('   validation (control LOO): %d/%d = %.1f%%  p=%.4f%s'
                  % (vok, vtot, 100 * vacc, vp, flag))
            rows.append(dict(metric=metric, roi=roi, test='control_loo',
                             k=vok, n=vtot, acc=vacc, p=vp))

            cL = L.values.mean(0)
            cR = R.values.mean(0)
            for label, P, expect in (('LH-intact pt (preserved LEFT)', PL, 1),
                                     ('RH-intact pt (preserved RIGHT)', PR, -1)):
                if len(P) < 3:
                    print('   %s: n=%d, skipped' % (label, len(P)))
                    continue
                calls = np.array([closer_to(P.loc[s].values, cL, cR, metric)
                                  for s in P.index])
                nL = int((calls == 1).sum())
                n = len(calls)
                p = binom_p(nL, n)
                native = 'LEFT' if expect == 1 else 'RIGHT'
                print('   %s  n=%d' % (label, n))
                print('      closer to LH centroid: %d/%d   (native side = %s)'
                      % (nL, n, native))
                mism = n - nL if expect == 1 else nL
                print('      assigned to the OTHER hemisphere: %d/%d  '
                      'binomial p=%.4f%s'
                      % (mism, n, p, ' *' if p == p and p < .05 else ''))
                rows.append(dict(metric=metric, roi=roi, test=label,
                                 k=nL, n=n, acc=nL / n, p=p))

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print('\nwrote %s' % args.csv)


if __name__ == '__main__':
    main()
