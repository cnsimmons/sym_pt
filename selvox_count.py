#!/usr/bin/env python
"""
Selective-voxel count by ROI and intact hemisphere: raw and control-normalized.

Row 1  raw suprathreshold voxel count, log y-axis (total loss)
Row 2  z relative to same-hemisphere controls, on sqrt(count)
       controls leave-one-out (each control z'd against the other 35) so their
       spread stays visible; patients z'd against the full control mean/SD.
       sqrt because counts are right-skewed and strictly positive.

Source: D_liu/univariate_v1.csv, column `volume`
        = int((searchmask & (z > SEL_Z_THRESH)).sum()). Raw, unharmonized.
Cohort: OTC + controls (nonOTC excluded); age cap <= 23 -> 12 / 12 / 36.
Controls first session, patients last session, intact hemisphere only.
house_PPA_strict per marlene_grid_brief.

Run: python selvox_count_v4.py     (saves PNG to C_results/figures/, writes no CSV)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/user_data/csimmon2/git_repos/sym_pt'
UNI = os.path.join(REPO, 'D_liu', 'univariate_v1.csv')
SUB = os.path.join(REPO, 'sub_info.csv')
FIGD = os.path.join(REPO, 'C_results', 'figures')
AGE_CAP = 23.0
NPERM = 10000
RNG = np.random.default_rng(0)

ROIS = ['object_LOC', 'house_PPA_strict', 'face_FFA', 'word_VWFA']   # a priori |LI| order
SHORT = {'object_LOC': 'object_LOC', 'house_PPA_strict': 'house_PPA',
         'face_FFA': 'face_FFA', 'word_VWFA': 'word_VWFA'}
COLOR = {'object_LOC': '#6699cc', 'house_PPA_strict': '#66aa77',
         'face_FFA': '#dd6644', 'word_VWFA': '#8877cc'}


def norm_id(s):
    return s.astype(str).str.replace('^sub-?', '', regex=True).str.zfill(3)


def norm_ses(s):
    return s.astype(str).str.extract(r'(\d+)', expand=False).astype(float)


# ------------------------------------------------------------------ load
uni = pd.read_csv(UNI)
sub = pd.read_csv(SUB)
uni = uni[uni['group'].isin(['OTC', 'control'])]
uni = uni[uni['category'].isin(ROIS)].copy()

uni['_sid'] = norm_id(uni['subject_id'])
uni['_ses'] = norm_ses(uni['session'])
sub['_sid'] = norm_id(sub['sub'])
sub['_ses'] = norm_ses(sub['ses'])
uni = uni.merge(sub[['_sid', '_ses', 'age']].drop_duplicates(['_sid', '_ses']),
                on=['_sid', '_ses'], how='left')

is_ctrl = uni['status'] == 'control'
ctrl = (uni[is_ctrl].sort_values('_ses')
        .groupby(['_sid', 'hemi', 'category'], as_index=False).first())
pt = (uni[~is_ctrl].sort_values('_ses')
      .groupby(['_sid', 'hemi', 'category'], as_index=False).last())
pt = pt[pt['hemi'].str.lower().str[0] == pt['intact_hemi'].str.lower().str[0]]

d = pd.concat([ctrl.assign(grp='control'), pt.assign(grp='patient')], ignore_index=True)
d['hemi'] = d['hemi'].str.upper().str[0]
d['age'] = pd.to_numeric(d['age'], errors='coerce')
d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
d = d[d.age <= AGE_CAP]
d['sq'] = np.sqrt(d['volume'])
print(d.groupby(['grp', 'hemi'])['_sid'].nunique())

# ------------------------------------------------------------------ control-referenced z on sqrt
d['zc'] = np.nan
for hemi in ['L', 'R']:
    for roi in ROIS:
        cell = (d.hemi == hemi) & (d.category == roi)
        cv = d.loc[cell & (d.grp == 'control'), 'sq']
        mu, sd = cv.mean(), cv.std(ddof=1)
        # patients: against full control distribution
        d.loc[cell & (d.grp == 'patient'), 'zc'] = \
            (d.loc[cell & (d.grp == 'patient'), 'sq'] - mu) / sd
        # controls: leave-one-out
        n = len(cv)
        for idx in cv.index:
            o = cv.drop(idx)
            d.loc[idx, 'zc'] = (cv[idx] - o.mean()) / o.std(ddof=1)


# ------------------------------------------------------------------ test (on sqrt scale)
def perm_p(a, b, nperm=NPERM):
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    n = len(a)
    null = np.empty(nperm)
    for i in range(nperm):
        s = RNG.permutation(pool)
        null[i] = s[:n].mean() - s[n:].mean()
    return obs, float((np.abs(null) >= abs(obs)).mean())


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


rows = []
for hemi in ['L', 'R']:
    for roi in ROIS:
        cell = (d.hemi == hemi) & (d.category == roi)
        c = d.loc[cell & (d.grp == 'control'), 'sq'].dropna().values
        p_ = d.loc[cell & (d.grp == 'patient'), 'sq'].dropna().values
        cr = d.loc[cell & (d.grp == 'control'), 'volume'].dropna().values
        pr = d.loc[cell & (d.grp == 'patient'), 'volume'].dropna().values
        diff, pv = perm_p(p_, c)
        rows.append(dict(hemi=hemi, roi=SHORT[roi], n_ctrl=len(c), n_pt=len(p_),
                         ctrl_med=np.median(cr), pt_med=np.median(pr),
                         pct=100 * (np.median(pr) / np.median(cr) - 1),
                         mean_z_pt=d.loc[cell & (d.grp == 'patient'), 'zc'].mean(),
                         d=cohen_d(p_, c), p=pv))

res = pd.DataFrame(rows)
m = len(res)
prev, q = 1.0, np.empty(m)
for rank, idx in enumerate(np.argsort(res['p'].values)[::-1]):
    prev = min(prev, res['p'].values[idx] * m / (m - rank))
    q[idx] = prev
res['q'] = q

print('\n=== selective-voxel count, patients vs matched-hemisphere controls '
      '(permutation on sqrt scale, BH-FDR over 8 cells) ===')
print(res.to_string(index=False, formatters={
    'ctrl_med': '{:.0f}'.format, 'pt_med': '{:.0f}'.format, 'pct': '{:+.0f}%'.format,
    'mean_z_pt': '{:+.2f}'.format, 'd': '{:+.2f}'.format,
    'p': '{:.4f}'.format, 'q': '{:.4f}'.format}))


# ------------------------------------------------------------------ figure
def strip(ax, hemi, col, val, logy):
    for i, roi in enumerate(ROIS):
        for j, (grp, mk, fill) in enumerate([('control', 'o', True), ('patient', 'D', False)]):
            v = d[(d.grp == grp) & (d.hemi == hemi) & (d.category == roi)][val].dropna().values
            if not len(v):
                continue
            x = i + (-0.18 if j == 0 else 0.18)
            ax.scatter(x + RNG.uniform(-0.055, 0.055, len(v)), v, s=24, marker=mk,
                       facecolor=COLOR[roi] if fill else 'none',
                       edgecolor=COLOR[roi] if fill else 'k',
                       linewidth=0.8, alpha=0.85, zorder=3)
            ax.hlines(np.median(v), x - 0.13, x + 0.13, color='k', lw=2, zorder=4)
    ax.set_xticks(range(len(ROIS)))
    ax.set_xticklabels([SHORT[r] for r in ROIS], fontsize=8.5)
    if logy:
        ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
for col, hemi in enumerate(['L', 'R']):
    strip(axes[0, col], hemi, col, 'volume', True)
    n_pt = d[(d.grp == 'patient') & (d.hemi == hemi)]['_sid'].nunique()
    n_ct = d[(d.grp == 'control') & (d.hemi == hemi)]['_sid'].nunique()
    axes[0, col].set_title(f'{hemi}H-intact   (pt n={n_pt}, ctrl n={n_ct})', fontsize=10)

    strip(axes[1, col], hemi, col, 'zc', False)
    axes[1, col].axhline(0, color='grey', lw=0.8, zorder=1)
    axes[1, col].axhspan(-2, 2, color='grey', alpha=0.08, zorder=0)

zl = max(abs(np.nanmin(d.zc)), abs(np.nanmax(d.zc))) * 1.08
for col in [0, 1]:
    axes[1, col].set_ylim(-zl, zl)
ylim0 = (min(a.get_ylim()[0] for a in axes[0]), max(a.get_ylim()[1] for a in axes[0]))
for a in axes[0]:
    a.set_ylim(*ylim0)

# significance stars on the z row
for col, hemi in enumerate(['L', 'R']):
    for i, roi in enumerate(ROIS):
        r = res[(res.hemi == hemi) & (res.roi == SHORT[roi])]
        if len(r) and r['q'].iloc[0] < .05:
            qq = r['q'].iloc[0]
            axes[1, col].text(i, zl * 0.86,
                              '***' if qq < .001 else '**' if qq < .01 else '*',
                              ha='center', fontsize=13)

axes[0, 0].set_ylabel('No. selective voxels  (z > 2.326, log scale)')
axes[1, 0].set_ylabel('z vs. same-hemisphere controls  [sqrt(count)]')
fig.suptitle('Category-selective voxel count: total (top) and control-normalized (bottom)'
             f'\nraw counts, unharmonized, age \u2264 {AGE_CAP:.0f}', fontsize=11)
fig.text(0.5, 0.008, 'filled circles = controls (first session, leave-one-out z); '
                     'open diamonds = patients, intact hemisphere (last session); '
                     'bars = medians; shaded band = \u00b12 SD',
         ha='center', fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])

os.makedirs(FIGD, exist_ok=True)
out = os.path.join(FIGD, 'selvox_count_raw_and_z_cap23.png')
fig.savefig(out, dpi=200)
print('\nsaved', out)

o = d[['_sid','grp','hemi','category','age','volume','zc']].sort_values(['hemi','category','grp','age'])
print(o.to_csv(index=False, float_format='%.2f'))