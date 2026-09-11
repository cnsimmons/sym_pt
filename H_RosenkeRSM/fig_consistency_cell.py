# ─── Within-group representational consistency (Rosenke Fig 4 + 5 style) ─────
# Prereqs from cell 2: FIG_DIR, np, pd, plt, Path
#
# Source: otc_rsm_persubject.csv, written by
#     python otc_rsm_rosenke.py --csv otc_rsm.csv
# Whole OTC parcel, cat-vs-all-others copes 6-9, no ROI and no peak-finding, so
# this measure is independent of the searchmask and cope-13 definitions.
#
# Each subject contributes six between-category pattern similarities
# (face-house ... object-word). Consistency = the mean correlation of a
# subject's six values against every other member of their own group.
#
# Panel A  per-subject consistency, four columns
# Panel B  patient x patient matrices, rows ordered by mean agreement
#
# Local names are suffixed _RSM so nothing in the rest of the notebook is
# overwritten.

from matplotlib.gridspec import GridSpec

RSM_CSV      = Path('/user_data/csimmon2/git_repos/sym_pt/otc_rsm_persubject.csv')
EXCLUDE_RSM  = {'sub-091', 'sub-084'}   # 091 over the age cap; 084 no face/word response
NPERM_RSM    = 10000
INTACT_RSM   = {'l': 'left', 'r': 'right'}
CC_RSM, PC_RSM = '#5b8fbf', '#c96a3a'
rng_rsm = np.random.default_rng(42)

rsm = pd.read_csv(RSM_CSV)
rsm = rsm[~rsm['subject_id'].isin(EXCLUDE_RSM)]
PAIRS_RSM = [c for c in rsm.columns if '-' in c]
assert len(PAIRS_RSM) == 6, PAIRS_RSM
print(f'{len(rsm)} subject x hemisphere rows | pairs: {PAIRS_RSM}')


def _within_rsm(mat):
    """Each row's mean correlation to every other row."""
    n = len(mat)
    out = np.full(n, np.nan)
    for i in range(n):
        o = [j for j in range(n) if j != i]
        out[i] = np.mean([np.corrcoef(mat[i], mat[j])[0, 1] for j in o])
    return out


def _perm_rsm(a, b, n=NPERM_RSM):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b]); na = len(a); k = 0
    for _ in range(n):
        p = rng_rsm.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= abs(obs) - 1e-12:
            k += 1
    return obs, (k + 1) / (n + 1)


def _star_rsm(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


D_RSM = {}
for hemi in ('l', 'r'):
    d = rsm[rsm['hemi'] == hemi]
    c = d[d['group'] == 'control']
    p = d[(d['group'] == 'OTC') & (d['intact_hemi'] == INTACT_RSM[hemi])]
    ctrl_v = _within_rsm(c[PAIRS_RSM].to_numpy(float))
    pt_v   = _within_rsm(p[PAIRS_RSM].to_numpy(float))
    diff, pv = _perm_rsm(ctrl_v, pt_v)
    D_RSM[hemi] = dict(ctrl=ctrl_v, pt=pt_v, diff=diff, p=pv,
                       pt_ids=p['subject_id'].str[-3:].to_numpy(),
                       pt_mat=p[PAIRS_RSM].to_numpy(float),
                       n_ctrl=len(c), n_pt=len(p))
    print(f'[{hemi.upper()}H] ctrl {ctrl_v.mean():+.3f} (n={len(c)})   '
          f'pt {pt_v.mean():+.3f} (n={len(p)})   diff {diff:+.3f}  p={pv:.4f}')

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 8.8))
gs = GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.46, wspace=0.20,
              left=0.075, right=0.95, top=0.93, bottom=0.08)

axA = fig.add_subplot(gs[0, :])
cols_rsm = [('LH ctrl', D_RSM['l']['ctrl'], CC_RSM, True),
            ('LH-intact', D_RSM['l']['pt'], PC_RSM, False),
            ('RH ctrl', D_RSM['r']['ctrl'], CC_RSM, True),
            ('RH-intact', D_RSM['r']['pt'], PC_RSM, False)]
for i, (lbl, v, col, solid) in enumerate(cols_rsm):
    axA.scatter(i + rng_rsm.uniform(-.13, .13, len(v)), v, s=26,
                facecolor=col if solid else 'white', edgecolor=col,
                linewidth=0 if solid else 1.0, alpha=.6 if solid else 1, zorder=3)
    axA.hlines(np.mean(v), i - .26, i + .26, color='k', lw=2.4, zorder=4)

lo_rsm = min(np.min(c[1]) for c in cols_rsm)
hi_rsm = max(np.max(c[1]) for c in cols_rsm)
pad = (hi_rsm - lo_rsm) * 0.30
for gi, hemi in [(0, 'l'), (2, 'r')]:
    y = hi_rsm + pad * 0.35
    axA.plot([gi, gi + 1], [y, y], color='k', lw=1)
    axA.text(gi + .5, y + pad * 0.06,
             f"{_star_rsm(D_RSM[hemi]['p'])}   p = {D_RSM[hemi]['p']:.4f}",
             ha='center', fontsize=9.5)
for i, (lbl, v, col, solid) in enumerate(cols_rsm):
    axA.text(i, lo_rsm - pad * 0.55, f'{np.mean(v):+.3f}', ha='center', fontsize=9)

axA.set_xticks(range(4))
axA.set_xticklabels([f'{c[0]}\n(n={len(c[1])})' for c in cols_rsm], fontsize=9.5)
axA.set_ylabel('mean r to others in own group', fontsize=10)
axA.set_ylim(lo_rsm - pad * 0.85, hi_rsm + pad)
axA.axhline(0, color='grey', lw=.6, zorder=1)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)
axA.set_title('A · within-group agreement in between-category pattern similarity',
              loc='left', fontsize=12, pad=14)

for k, (hemi, name) in enumerate([('r', 'RH-intact'), ('l', 'LH-intact')]):
    ax = fig.add_subplot(gs[1, k])
    M = D_RSM[hemi]['pt_mat']; ids = D_RSM[hemi]['pt_ids']
    R = np.corrcoef(M); np.fill_diagonal(R, np.nan)
    order = np.argsort(-np.nanmean(R, axis=1))
    Rs = R[np.ix_(order, order)]
    im = ax.imshow(Rs, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids[order], fontsize=7.5, rotation=90)
    ax.set_yticklabels(ids[order], fontsize=7.5)
    offd = Rs[np.triu_indices(len(Rs), 1)]
    ax.set_title(f'{name} (n={len(ids)})    r < 0.2 in '
                 f'{int((offd < .2).sum())}/{len(offd)} pairs',
                 loc='left', fontsize=10.5, pad=8)
    if k == 1:
        cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
        cb.set_label('patient–patient r', fontsize=9)
        cb.ax.tick_params(labelsize=8)

fig.text(0.075, 0.50, 'B · who resembles whom (rows ordered by mean agreement)',
         fontsize=12, ha='left')
fig.text(0.075, 0.015,
         'whole OTC parcel, no ROI · six between-category pattern similarities per subject · '
         'permutation, 10,000 label shuffles · sub-091 and sub-084 excluded',
         fontsize=8.5, color='#555')

out_rsm = FIG_DIR / 'otc_rsm_consistency.png'
plt.savefig(out_rsm, dpi=300, bbox_inches='tight')
plt.show()
print('saved', out_rsm)
