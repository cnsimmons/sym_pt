#!/usr/bin/env python3
"""
marlene_grid.py — the 15-cell grid per measure that Marlene asked for.

THREE GROUP COMPARISONS
  1  A = LH controls      B = LH-intact pt
  2  A = RH controls      B = RH-intact pt
  3  A = LH-intact pt     B = RH-intact pt      (12 vs 12)
  beta always refers to group B.

FIVE SPECIFICATIONS per comparison
  binA   binary  word+face          vs  house+object
  binB   binary  word               vs  face+house+object
  binC   binary  word+face+house    vs  object
  cont196  continuous, |LI| at z>1.96 as the category-level predictor
  cont233  continuous, |LI| at z>2.33

= 15 cells per measure. Measures here: peak_z, RSA distinctiveness, RSA geometry.
TFCE is excluded — it has no subject x category value, so it cannot take this
form; its row is the existing cluster table.

WHAT EACH CELL REPORTS
  the CATEGORY x GROUP INTERACTION: does the category effect differ between the
  two groups being compared.

  Model:  value ~ subject fixed effects + category + group:modifier
  The interaction is a single coefficient (1 df). Tested by permuting the GROUP
  label at the SUBJECT level, which preserves each subject's four category values
  and so respects the within-subject structure.

  Reported: estimate (beta), permutation p, and n per group.

SIGN CONVENTIONS — read before interpreting
  peak_z             higher = MORE selective
  distinctiveness    liu_distinctiveness stores SIMILARITY, so higher = LESS
                     distinct. The script FLIPS it so higher = more distinct in
                     all output, making beta signs comparable across measures.
  geometry           PAIR-LEVEL. Unit is subject x ROI x pair, all 6 pairs.
                     stores similarity, flipped so higher = more separated.

                     Geometry must NOT be the mean over pairs involving the
                     preferred category — that is exactly liu_distinctiveness
                     (same 3 pairs; mean abs difference 0.021, arising only
                     because combat_06 harmonizes the two as separate feature
                     blocks). Keeping the pair as the unit preserves the 6-pair
                     structure that made rFFA face-word visible.

                     Pair-level modifiers are the mean of the two categories'
                     codes, so the same five specifications apply:
                       binA  face-word = 1.0, house-object = 0.0, mixed = 0.5
                       binB  any pair containing word = 0.5, others = 0.0
                       binC  pairs not containing object = 1.0, others = 0.5
                       cont  mean |LI| of the two categories in the pair
                     ROI and pair dummies are included alongside subject dummies,
                     so the interaction is within subject, ROI, and pair.
                     NOTE this pools ROIs to keep the grid at 15 cells; an
                     ROI-specific pair effect (e.g. rFFA face-word) needs the
                     separate 6-pair table.

  For binary specs the modifier is 1 for the first-named set, 0 for the second.
  A NEGATIVE beta means the first-named set is relatively MORE affected in
  GROUP B. That is the direction the lateralized-categories hypothesis predicts
  for comparisons 1 and 2.

  For continuous specs the modifier is |LI|, so a NEGATIVE beta means more
  lateralized categories are relatively more affected.

CAVEATS
  - peak_z is NOT ComBat-harmonized (it is not in MEASURES in combat_05)
  - comparison 3 has no control reference, so its interaction is
    patient-group x category, a different quantity from comparisons 1 and 2
  - the two |LI| thresholds barely differ, and object/house SWAP RANK at 1.96
    (0.114 vs 0.115), so the object/house ordering is arbitrary in both
  - age cap <=23 applied by default, per the agreed spec

Usage
  python marlene_grid.py
  python marlene_grid.py --cap -1                 # no age cap
  python marlene_grid.py --roi-set primary        # house_PPA instead of strict
  python marlene_grid.py --n-perm 20000
  python marlene_grid.py --csv grid.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GIT = Path('/user_data/csimmon2/git_repos/sym_pt')
RSA = GIT / 'D_liu' / 'rsa_v1_harmonized.csv'
UNI = GIT / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv'
INFO = GIT / 'sub_info.csv'

ROI_SETS = {
    'primary_strict': ['object_LOC', 'house_PPA_strict', 'face_FFA', 'word_VWFA'],
    'primary':        ['object_LOC', 'house_PPA', 'face_FFA', 'word_VWFA'],
}
# ROI -> its preferred category label, used to pick geometry pairs
PREF = {'object_LOC': 'object', 'house_PPA': 'house',
        'house_PPA_strict': 'house', 'face_FFA': 'face', 'word_VWFA': 'word'}
SHORT = PREF

# control |LI| computed from selective_voxel_counts.csv, both thresholds
LI = {
    2.33: {'word_VWFA': 0.389, 'face_FFA': 0.245,
           'object_LOC': 0.130, 'house_PPA': 0.124, 'house_PPA_strict': 0.124},
    1.96: {'word_VWFA': 0.357, 'face_FFA': 0.234,
           'object_LOC': 0.114, 'house_PPA': 0.115, 'house_PPA_strict': 0.115},
}

CATS = ['face', 'house', 'object', 'word']
PAIRS = ['face-house', 'face-object', 'face-word',
         'house-object', 'house-word', 'object-word']


def pair_modifier(cat_codes):
    """Lift a per-category code to a per-pair code: mean of the two members."""
    return {pr: np.mean([cat_codes[c] for c in pr.split('-')]) for pr in PAIRS}


# per-CATEGORY codes for the binary splits, keyed by short category name
CAT_SPLITS = {
    'binA  word+face | house+object':
        {'word': 1, 'face': 1, 'house': 0, 'object': 0},
    'binB  word | face+house+object':
        {'word': 1, 'face': 0, 'house': 0, 'object': 0},
    'binC  word+face+house | object':
        {'word': 1, 'face': 1, 'house': 1, 'object': 0},
}
LI_CAT = {
    2.33: {'word': 0.389, 'face': 0.245, 'object': 0.130, 'house': 0.124},
    1.96: {'word': 0.357, 'face': 0.234, 'object': 0.114, 'house': 0.115},
}


# binary splits: 1 for the first-named set, 0 for the second
SPLITS = {
    'binA  word+face | house+object':
        {'word_VWFA': 1, 'face_FFA': 1, 'house_PPA': 0, 'house_PPA_strict': 0,
         'object_LOC': 0},
    'binB  word | face+house+object':
        {'word_VWFA': 1, 'face_FFA': 0, 'house_PPA': 0, 'house_PPA_strict': 0,
         'object_LOC': 0},
    'binC  word+face+house | object':
        {'word_VWFA': 1, 'face_FFA': 1, 'house_PPA': 1, 'house_PPA_strict': 1,
         'object_LOC': 0},
}


# ---------------------------------------------------------------- data loading

def _sessions(df, group, rule):
    x = df[df['group'] == group].copy()
    s = x.groupby('subject_id')['session'].agg(rule).rename('sx')
    x = x.join(s, on='subject_id')
    return x[x['session'] == x['sx']].drop(columns=['sx'], errors='ignore')


def _add_age(df, info):
    return df.merge(info[['sub', 'session', 'age']],
                    left_on=['subject_id', 'session'],
                    right_on=['sub', 'session'], how='left')


def load_measure(measure, rois, cap, quiet=False):
    """Return (ctl, pat) long frames with columns subject_id, hemi, roi, val.

    val is oriented so HIGHER = MORE selective / MORE distinct for every measure.
    Patients carry an extra 'intact' column and contain only the intact hemi.
    """
    info = pd.read_csv(INFO)
    info['session'] = info['ses'].str.replace('ses-', '', regex=False).astype(int)

    if measure == 'peak_z':
        d = pd.read_csv(UNI)
        d['session'] = d['session'].astype(int)
        d = d[d['category'].isin(rois)].drop_duplicates(
            ['subject_id', 'session', 'hemi', 'category'])
        d = d.rename(columns={'category': 'roi', 'peak_z': 'val'})
        flip = False

    elif measure == 'distinctiveness':
        d = pd.read_csv(RSA)
        d['session'] = d['session'].astype(int)
        d = d[d['category'].isin(rois)].drop_duplicates(
            ['subject_id', 'session', 'hemi', 'category'])
        d = d.rename(columns={'category': 'roi',
                              'liu_distinctiveness': 'val'})
        flip = True                       # stores similarity

    elif measure == 'geometry':
        d = pd.read_csv(RSA)
        d['session'] = d['session'].astype(int)
        d = d[d['category'].isin(rois)].drop_duplicates(
            ['subject_id', 'session', 'hemi', 'category', 'pair'])
        d = d.rename(columns={'category': 'roi', 'fisher_r': 'val'})
        flip = True                       # stores similarity
    else:
        raise ValueError(measure)

    ctl = _add_age(_sessions(d, 'control', 'min'), info)
    pat = _sessions(d, 'OTC', 'max')
    pat['intact'] = pat['intact_hemi'].map({'left': 'l', 'right': 'r'})
    pat = _add_age(pat[pat['hemi'] == pat['intact']], info)

    if flip:
        ctl['val'] = -ctl['val']
        pat['val'] = -pat['val']

    if cap is not None:
        dc = sorted(ctl.loc[ctl['age'] > cap, 'subject_id'].unique())
        dp = sorted(pat.loc[pat['age'] > cap, 'subject_id'].unique())
        if (dc or dp) and not quiet:
            print(f'  [{measure}] age > {cap:g} excluded — '
                  f'controls {dc}, patients {dp}')
        ctl = ctl[ctl['age'] <= cap]
        pat = pat[pat['age'] <= cap]

    cols = ['subject_id', 'hemi', 'roi', 'val']
    if measure == 'geometry':
        cols = cols + ['pair']
    return ctl[cols], pat[cols + ['intact']]


# ------------------------------------------------------------------- the model

def build_frame(ctl, pat, comparison, rois, pair_level=False):
    """Long frame with subject_id, roi, val, grp. grp=1 is GROUP B."""
    keep = ['subject_id', 'roi', 'val'] + (['pair'] if pair_level else [])
    if comparison == 1:
        a = ctl[ctl['hemi'] == 'l'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'l'][keep].assign(grp=1)
    elif comparison == 2:
        a = ctl[ctl['hemi'] == 'r'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'r'][keep].assign(grp=1)
    elif comparison == 3:
        a = pat[pat['intact'] == 'l'][keep].assign(grp=0)
        b = pat[pat['intact'] == 'r'][keep].assign(grp=1)
    else:
        raise ValueError(comparison)
    df = pd.concat([a, b], ignore_index=True).dropna(subset=['val'])
    # balanced design: require the full complement of cells per subject
    need = len(rois) * (len(PAIRS) if pair_level else 1)
    unit = df.groupby('subject_id').size()
    return df[df['subject_id'].isin(unit[unit == need].index)].copy()


def interaction(df, modifier, n_perm, seed=0):
    """OLS: val ~ subject dummies + category dummies + grp*modifier.

    Subject dummies absorb every subject-level effect (overall signal level,
    age, sex, scanner) so the interaction is estimated purely within subject.
    Permutation shuffles the group label across subjects, keeping each
    subject's four values together.
    """
    df = df.copy()
    key = 'pair' if 'pair' in df.columns else 'roi'
    df['m'] = df[key].map(modifier).astype(float)
    if df['grp'].nunique() < 2 or df['m'].nunique() < 2:
        return np.nan, np.nan, 0, 0

    S = pd.get_dummies(df['subject_id']).astype(float).values
    C = pd.get_dummies(df['roi'], drop_first=True).astype(float).values
    if key == 'pair':
        C = np.hstack([C, pd.get_dummies(df['pair'],
                                         drop_first=True).astype(float).values])
    y = df['val'].to_numpy(float)
    mvec = df['m'].to_numpy(float)

    def beta(grp_vec):
        X = np.hstack([S, C, (grp_vec * mvec)[:, None]])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b[-1]

    obs = beta(df['grp'].to_numpy(float))

    sub = df.drop_duplicates('subject_id').set_index('subject_id')['grp']
    ids, labels = sub.index.to_numpy(), sub.to_numpy(float)
    rng = np.random.default_rng(seed)
    k = 0
    for _ in range(n_perm):
        m = dict(zip(ids, rng.permutation(labels)))
        if abs(beta(df['subject_id'].map(m).to_numpy(float))) >= abs(obs) - 1e-12:
            k += 1
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    return obs, (k + 1) / (n_perm + 1), n0, n1


# ------------------------------------------------------------------------ main

# beta always refers to group B (grp=1). A is grp=0.
COMPARISONS = {
    1: 'A = LH ctrl        B = LH-intact pt',
    2: 'A = RH ctrl        B = RH-intact pt',
    3: 'A = LH-intact pt   B = RH-intact pt',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=float, default=23.0,
                    help='max scan age; -1 for no cap')
    ap.add_argument('--roi-set', choices=list(ROI_SETS), default='primary_strict')
    ap.add_argument('--n-perm', type=int, default=5000)
    ap.add_argument('--measures', nargs='+',
                    default=['peak_z', 'distinctiveness', 'geometry'])
    ap.add_argument('--csv', default=None, help='write the grid to this path')
    args = ap.parse_args()

    rois = ROI_SETS[args.roi_set]
    cap = None if args.cap < 0 else args.cap

    print(f'ROIs: {rois}')
    print(f'age cap: {"none" if cap is None else cap}   '
          f'permutations: {args.n_perm}')
    print('all measures oriented so HIGHER = MORE selective / MORE distinct')
    print('beta < 0  =  the first-named category set is relatively MORE '
          'affected in GROUP B')
    print('           (for continuous specs: more lateralized categories '
          'relatively more affected)')

    roi_specs = list(SPLITS.items()) + [
        ('cont  |LI| z>1.96', LI[1.96]),
        ('cont  |LI| z>2.33', LI[2.33]),
    ]
    pair_specs = [(k, pair_modifier(v)) for k, v in CAT_SPLITS.items()] + [
        ('cont  |LI| z>1.96', pair_modifier(LI_CAT[1.96])),
        ('cont  |LI| z>2.33', pair_modifier(LI_CAT[2.33])),
    ]

    rows = []
    for measure in args.measures:
        ctl, pat = load_measure(measure, rois, cap)
        print(f'\n{"=" * 78}\nMEASURE: {measure}')
        if measure == 'peak_z':
            print('  NOTE peak_z is not ComBat-harmonized')
        pair_level = (measure == 'geometry')
        specs = pair_specs if pair_level else roi_specs
        if pair_level:
            print('  unit = subject x ROI x pair (6 pairs); ROI and pair '
                  'dummies included')
        for comp, cname in COMPARISONS.items():
            df = build_frame(ctl, pat, comp, rois, pair_level)
            print(f'\n  {comp}. {cname}')
            if comp == 3:
                print('     (patient-group x category; no control reference)')
            print(f'     {"specification":32s} {"beta":>9s} {"p":>8s} '
                  f'{"nA":>3s} {"nB":>3s}')
            for sname, mod in specs:
                b, p, n0, n1 = interaction(df, mod, args.n_perm)
                star = ' *' if (p == p and p < .05) else ''
                bs = 'n/a' if b != b else f'{b:+9.3f}'
                ps = 'n/a' if p != p else f'{p:8.4f}'
                print(f'     {sname:32s} {bs} {ps} {n0:3d} {n1:3d}{star}')
                rows.append(dict(measure=measure, comparison=comp,
                                 comparison_name=cname, spec=sname,
                                 beta=b, p=p, n_group_a=n0, n_group_b=n1,
                                 age_cap=cap, roi_set=args.roi_set))

    print(f'\n{"=" * 78}')
    print('TFCE: no subject x category value exists, so it cannot take this '
          'form. Report the existing cluster table instead.')

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'\nWrote {args.csv}  ({len(rows)} cells)')


if __name__ == '__main__':
    main()