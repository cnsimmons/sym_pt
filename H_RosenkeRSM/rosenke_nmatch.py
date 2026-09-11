"""n-matched control resampling for the odRSM within-group similarity gap.

Reads the per-subject table written by otc_rsm_rosenke.py --csv.
Controls are subsampled to the patient n so the two mean-pairwise-similarity
values are computed over the same number of subjects. Also reports the
age-capped cohort used by the primary analyses.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path

CSV = Path('/user_data/csimmon2/git_repos/sym_pt/C_results/rosenke_persubject.csv')
CAP_EXCLUDE = ['sub-091', 'sub-095', 'sub-096']   # age > 23
N_ITER = 2000
RNG = np.random.default_rng(42)

PAIRCOLS = ['face-house', 'face-object', 'face-word',
            'house-object', 'house-word', 'object-word']


def within_group_similarity(mat):
    """Leave-self-out mean pairwise correlation of odRSM vectors."""
    n = len(mat)
    out = np.full(n, np.nan)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if not others:
            continue
        out[i] = np.mean([np.corrcoef(mat[i], mat[j])[0, 1] for j in others])
    return out


def run(df, label):
    print('\n' + '=' * 70)
    print(label)
    print('=' * 70)
    for hemi in ['l', 'r']:
        d = df[df['hemi'] == hemi]
        ctl = d[d['group'] == 'control'][PAIRCOLS].values
        pt = d[(d['group'] == 'OTC') &
               (d['intact_hemi'] == ('left' if hemi == 'l' else 'right'))
               ][PAIRCOLS].values
        if len(ctl) < 5 or len(pt) < 3:
            print(f'\n[{hemi.upper()}H] too few (ctl={len(ctl)}, pt={len(pt)})')
            continue

        wp = np.nanmean(within_group_similarity(pt))
        wc_full = np.nanmean(within_group_similarity(ctl))

        # controls subsampled to the patient n
        sub = np.empty(N_ITER)
        for k in range(N_ITER):
            idx = RNG.choice(len(ctl), len(pt), replace=False)
            sub[k] = np.nanmean(within_group_similarity(ctl[idx]))

        p = (np.sum(sub <= wp) + 1) / (N_ITER + 1)
        lo, hi = np.percentile(sub, [2.5, 97.5])

        print(f'\n[{hemi.upper()}H]  ctl n={len(ctl)}  pt n={len(pt)}')
        print(f'   patient mean r          {wp:+.3f}')
        print(f'   control mean r, full n  {wc_full:+.3f}')
        print(f'   control mean r, n={len(pt):<2d}      {sub.mean():+.3f}  '
              f'[95% {lo:+.3f}, {hi:+.3f}]')
        print(f'   gap at matched n        {wp - sub.mean():+.3f}')
        print(f'   p (patient <= matched control)  {p:.4f}'
              + ('  *' if p < .05 else ''))
        print(f'   n-matching moved the control value by '
              f'{sub.mean() - wc_full:+.3f}')


def main():
    if not CSV.exists():
        sys.exit(f'not found: {CSV}\nRun otc_rsm_rosenke.py --csv first.')
    df = pd.read_csv(CSV)
    print(f'{CSV.name}: {len(df)} subject x hemisphere rows')

    run(df, 'FULL COHORT (no age cap) — matches otc_rsm_rosenke.py')

    capped = df[~df['subject_id'].isin(CAP_EXCLUDE)]
    dropped = sorted(set(df['subject_id']) & set(CAP_EXCLUDE))
    print(f'\n\nage cap drops: {dropped or "none present"}')
    run(capped, 'AGE-CAPPED COHORT (<=23) — matches the primary analyses')


if __name__ == '__main__':
    main()
