#!/usr/bin/env python3
"""
heterogeneity_tsnr.py — does the LH-intact heterogeneity result survive data
quality matching?

CONTEXT
  otc_rsm_rosenke.py found LH-intact patients less internally consistent than
  controls (within-group odRSM r = 0.404 vs 0.670, p = .0001), while RH-intact
  patients were indistinguishable from controls (0.533 vs 0.593, n.s.).

  quality_check.py then showed that LH-intact patients ALSO have worse data:
  tSNR -23.1 (p = .030) and motion +0.037 (p = .019). RH-intact differs on
  neither. Lower tSNR lowers between-subject pattern correlations mechanically,
  so the heterogeneity result and the confound point at the same group.

THREE TESTS
  1  tSNR thresholding. Recompute the within-group similarity at increasing tSNR
     floors, applied to BOTH groups so they stay matched. If the patient-control
     gap closes as the floor rises, the result was data quality.
  2  Within-patient correlation between a subject's own tSNR and how similar
     their odRSM is to the rest of their group. A strong positive correlation is
     direct evidence that noise drives the measure.
  3  Same for motion.

READING IT
  gap persists at every floor, r near 0    heterogeneity is real
  gap closes as the floor rises            it was data quality; drop the analysis
  in between                               not resolvable at this n; drop it

  With 13 LH-intact patients, thresholding costs power fast. A gap that merely
  becomes non-significant is NOT evidence the effect was real, so the printout
  reports n at every step.

Usage
  python heterogeneity_tsnr.py --rsm otc_rsm_persubject.csv --quality quality_check.csv
"""
import argparse

import numpy as np
import pandas as pd

PAIRS = ['face-house', 'face-object', 'face-word',
         'house-object', 'house-word', 'object-word']
RNG = np.random.default_rng(42)


def within_group_similarity(M):
    n = len(M)
    return np.array([np.mean([np.corrcoef(M[i], M[j])[0, 1]
                              for j in range(n) if j != i]) for i in range(n)])


def perm(a, b, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b]); na = len(a); k = 0
    for _ in range(n):
        p = RNG.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rsm', default='otc_rsm_persubject.csv')
    ap.add_argument('--quality', default='quality_check.csv')
    args = ap.parse_args()

    d = pd.read_csv(args.rsm)
    q = pd.read_csv(args.quality)
    d = d.merge(q[['subject_id', 'tsnr', 'motion']], on='subject_id', how='left')
    if d['tsnr'].isna().any():
        print(f"  {d['tsnr'].isna().sum()} rows without quality data — dropped")
        d = d.dropna(subset=['tsnr'])

    for hemi, side in [('l', 'left'), ('r', 'right')]:
        h = d[d['hemi'] == hemi]
        ctl = h[h['group'] == 'control']
        pt = h[(h['group'] == 'OTC') & (h['intact_hemi'] == side)]
        if len(pt) < 4:
            continue

        lab = 'LH-intact' if side == 'left' else 'RH-intact'
        print('\n' + '=' * 70)
        print(f'{lab}  —  within-group odRSM similarity under tSNR matching')
        print('=' * 70)
        print(f"{'floor':>8s} {'nC':>4s} {'nP':>4s} {'ctrl r':>8s} {'pt r':>8s} "
              f"{'diff':>8s} {'p':>8s}")

        for cut in [0, 60, 80, 100, 120]:
            c2 = ctl[ctl['tsnr'] >= cut]
            p2 = pt[pt['tsnr'] >= cut]
            if len(p2) < 4 or len(c2) < 5:
                print(f"{'>=' + str(cut):>8s} {len(c2):4d} {len(p2):4d}"
                      '   (too few)')
                continue
            wc = within_group_similarity(c2[PAIRS].values)
            wp = within_group_similarity(p2[PAIRS].values)
            dd, pp = perm(wc, wp)
            name = 'all' if cut == 0 else f'>={cut}'
            print(f"{name:>8s} {len(c2):4d} {len(p2):4d} {wc.mean():+8.3f} "
                  f"{wp.mean():+8.3f} {dd:+8.3f} {pp:8.4f}"
                  + ('  *' if pp < .05 else ''))

        wp = within_group_similarity(pt[PAIRS].values)
        wc = within_group_similarity(ctl[PAIRS].values)
        print()
        for var in ['tsnr', 'motion']:
            rp = np.corrcoef(pt[var].values, wp)[0, 1]
            rc = np.corrcoef(ctl[var].values, wc)[0, 1]
            print(f'  r({var:6s}, similarity to own group): '
                  f'patients {rp:+.3f} (n={len(pt)}), '
                  f'controls {rc:+.3f} (n={len(ctl)})')
        print('  A strong positive r for tSNR (or negative for motion) means the')
        print('  measure is tracking data quality rather than organization.')


if __name__ == '__main__':
    main()
