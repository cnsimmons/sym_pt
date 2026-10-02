# ─── Figure: side typicality as paired same-side and opposite-side agreement ─
# Same quantity as the side-typicality cell, decomposed into its two terms.
# Each hemisphere contributes two points joined by a line: the mean r of its
# six between-category similarities against same-side controls, and against
# opposite-side controls. The SLOPE is typicality. A hemisphere that resembles
# all controls less, but resembles neither side more, gives a flat line at a
# lower level -- which is why a uniform reduction in agreement cannot produce
# the effect. Self-comparisons are excluded, and a control's two hemispheres
# are never correlated with each other.
#
# VALIDATION: the printed group means must reproduce +0.147, +0.050, +0.002,
# -0.008. If they do not, this cell is not computing the manuscript quantity.

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

RSM_CSV = 'H_RosenkeRSM/otc_rsm_persubject.csv'
OUTDIR  = '/user_data/csimmon2/git_repos/sym_pt/C_results/figures'

PAIRS = ['face-house', 'face-object', 'face-word',
         'house-object', 'house-word', 'object-word']

# live convention: blue = left hemisphere, red = right hemisphere
BLUE, RED = '#1f77b4', '#d62728'

d = pd.read_csv(RSM_CSV).dropna(subset=PAIRS).reset_index(drop=True)

is_ctrl = d['group'].str.lower().eq('control')
hemi_l  = d['hemi'].str.lower().str[0].eq('l')

pt = d[~is_ctrl]
mism = pt[pt['intact_hemi'].astype(str).str.lower().str[0] != pt['hemi'].str.lower().str[0]]
if len(mism):
    print('WARNING hemi != intact_hemi in %d patient rows:' % len(mism),
          mism['subject_id'].tolist())

d['cell'] = np.where(is_ctrl,
                     np.where(hemi_l, 'LH ctrl', 'RH ctrl'),
                     np.where(hemi_l, 'LH-intact', 'RH-intact'))

R = np.corrcoef(d[PAIRS].to_numpy(float))
np.fill_diagonal(R, np.nan)

ctrl_L = np.where(d['cell'].eq('LH ctrl'))[0]
ctrl_R = np.where(d['cell'].eq('RH ctrl'))[0]
subs   = d['subject_id'].to_numpy()

def mean_r(i, ref):
    """Mean profile agreement of hemisphere i with a reference set, excluding
    any hemisphere belonging to the same subject."""
    keep = [j for j in ref if j != i and subs[j] != subs[i]]
    return np.nanmean(R[i, keep])

d['r_LH_ctrl'] = [mean_r(i, ctrl_L) for i in range(len(d))]
d['r_RH_ctrl'] = [mean_r(i, ctrl_R) for i in range(len(d))]

# reorient onto same-side / opposite-side
d['r_same'] = np.where(hemi_l, d['r_LH_ctrl'], d['r_RH_ctrl'])
d['r_opp']  = np.where(hemi_l, d['r_RH_ctrl'], d['r_LH_ctrl'])
d['typicality'] = d['r_same'] - d['r_opp']

ORDER = ['LH ctrl', 'RH ctrl', 'LH-intact', 'RH-intact']
COL   = {'LH ctrl': BLUE, 'RH ctrl': RED, 'LH-intact': BLUE, 'RH-intact': RED}
FILL  = {'LH ctrl': False, 'RH ctrl': False, 'LH-intact': True, 'RH-intact': True}

print('recomputed typicality means  (manuscript: +0.147  +0.050  +0.002  -0.008)')
for g in ORDER:
    s = d[d['cell'].eq(g)]
    print('  %-10s n=%2d  same %+.3f  opposite %+.3f  typicality %+.3f'
          % (g, len(s), s['r_same'].mean(), s['r_opp'].mean(), s['typicality'].mean()))

# ─── plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.8))
DX = 0.19

SHOW_INDIVIDUALS = True   # set False for a means-only version

for i, g in enumerate(ORDER):
    s, c = d[d['cell'].eq(g)], COL[g]
    x = [i - DX, i + DX]

    if SHOW_INDIVIDUALS:
        for _, r in s.iterrows():
            ax.plot(x, [r['r_same'], r['r_opp']], color=c, lw=.6, alpha=.13, zorder=1)

    # fill encodes control vs patient only; position encodes same vs opposite
    face = c if FILL[g] else 'white'
    m_same, m_opp = s['r_same'].mean(), s['r_opp'].mean()
    ax.plot(x, [m_same, m_opp], color=c, lw=3.0, zorder=3,
            solid_capstyle='round')
    ax.scatter(x, [m_same, m_opp], s=[95, 95], zorder=4,
               facecolor=face, edgecolor=c, linewidth=2.2)
    ax.annotate('%+.3f' % s['typicality'].mean(), xy=(i, max(m_same, m_opp) + .035),
                ha='center', fontsize=9, color=c)

ax.axvline(1.5, color='0.88', lw=.9, zorder=0)
ax.set_xticks(range(4))
ax.set_xticklabels(['%s\n(n=%d)' % (g, (d['cell'] == g).sum()) for g in ORDER],
                   fontsize=9)
ax.set_xlim(-.55, 3.55)
ax.set_ylabel('mean r to control hemispheres')
ax.set_title('Agreement with same-side and opposite-side controls', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)

# legend as text, since the encoding is position not color
ax.annotate('left marker: same-side controls\nright marker: opposite-side controls\n'
            'slope = side typicality',
            xy=(0.985, 0.03), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=8, color='0.35')

fig.text(.5, -.06, "whole OTC parcel, no ROI \u00b7 six between-category pattern "
                   "similarities per subject \u00b7 faint lines are individual hemispheres\n"
                   "self-comparisons excluded; a control's two hemispheres are never "
                   "correlated with each other",
         ha='center', fontsize=7.5, color='0.35')

fig.savefig('%s/typicality_paired_components.png' % OUTDIR, dpi=300,
            bbox_inches='tight', facecolor='white')
fig.savefig('%s/typicality_paired_components.pdf' % OUTDIR, bbox_inches='tight')
plt.show()