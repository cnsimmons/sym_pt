## PLAYGROUND
# ─── Typicality against both control hemispheres ─────────────────────────
# One number per subject per ROI: correlation between that subject's six-pair
# profile and each control hemisphere's mean profile.
#
# The contrast is within-subject: does a patient's profile resemble its own
# hemisphere's controls more, or the other hemisphere's? A negative difference
# in RH-intact patients is the crossing pattern seen at FFA face-word.
#
# Pearson, not Spearman: with only six pairs, rank correlations collapse to a
# handful of possible values and both control hemispheres can return an
# identical rank profile, giving a spurious difference of exactly zero.
#
# Controls are scored leave-one-out against their own hemisphere so the
# self-comparison is not inflated (Weber et al. 2026).
from scipy.stats import pearsonr, wilcoxon

TYP_PAIRS = PAIRS_ORDERED          # fixed order, same for every subject
_r = lambda a, b: pearsonr(a, b)[0]


def _profile(df, sub, ses, roi, hemi):
    d = df[(df['subject_id'] == sub) & (df['ses_num'] == ses) &
           (df['category'] == roi) & (df['hemi'] == hemi)]
    m = dict(zip(d['pair'], d['fisher_r']))
    v = np.array([m.get(p, np.nan) for p in TYP_PAIRS], float)
    return None if np.isnan(v).any() else v


ctrl_keys = sorted(set(zip(rsa_c_long['subject_id'], rsa_c_long['ses_num'])))
pt_keys = sorted(set(zip(rsa_p_long['subject_id'], rsa_p_long['ses_num'],
                         rsa_p_long['intact_hemi'])))

rows = []
for roi in [r for r in ROI_ORDER if r in set(rsa_c_long['category'])]:
    cprof, cmean = {}, {}
    for h in ('l', 'r'):
        p = {k: _profile(rsa_c_long, k[0], k[1], roi, h) for k in ctrl_keys}
        cprof[h] = {k: v for k, v in p.items() if v is not None}
        cmean[h] = np.mean(list(cprof[h].values()), axis=0)

    # sanity check: the two control hemispheres must not be identical
    if roi == [r for r in ROI_ORDER if r in set(rsa_c_long['category'])][0]:
        print('control mean profiles differ between hemispheres? '
              f'{not np.allclose(cmean["l"], cmean["r"])}')

    for h in ('l', 'r'):
        oth = 'r' if h == 'l' else 'l'
        for k, v in cprof[h].items():
            others = [x for kk, x in cprof[h].items() if kk != k]
            rows.append(dict(roi=roi, group=f'ctrl {h.upper()}', sub=k[0],
                             r_own=_r(v, np.mean(others, axis=0)),
                             r_other=_r(v, cmean[oth])))

    for sub, ses, intact in pt_keys:
        h = 'l' if intact == 'left' else 'r'
        oth = 'r' if h == 'l' else 'l'
        v = _profile(rsa_p_long, sub, ses, roi, h)
        if v is None:
            continue
        rows.append(dict(roi=roi, group=f'{h.upper()}H-intact pt', sub=sub,
                         r_own=_r(v, cmean[h]), r_other=_r(v, cmean[oth])))

TYP = pd.DataFrame(rows)
TYP['own_minus_other'] = TYP['r_own'] - TYP['r_other']

print('\nTypicality (Pearson): r with own- vs other-hemisphere control mean')
print(f"{'ROI':6}{'group':16}{'n':>4}{'r_own':>8}{'r_other':>9}{'diff':>8}{'p':>9}")
for roi, g in TYP.groupby('roi', sort=False):
    for grp, s in g.groupby('group', sort=False):
        d = s['own_minus_other'].dropna().values
        p = wilcoxon(d)[1] if len(d) >= 6 and np.abs(d).sum() > 0 else np.nan
        print(f"{ROI_TITLE.get(roi, roi):6}{grp:16}{len(s):>4}"
              f"{s.r_own.mean():>8.3f}{s.r_other.mean():>9.3f}"
              f"{d.mean():>8.3f}{p:>9.4f}")
    print()