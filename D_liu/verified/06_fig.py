#!/usr/bin/env python3
"""
fig_wta_territory.py — WTA territory bar plot (LH-intact / RH-intact VOTC).

Data sources (read, never recomputed):
  - bars + per-subject points : wta_percentages.csv
        filter region=='otc' & denominator=='selective', last-session
        (controls = first session, OTC patients = last session) -- matches 05.
  - significance stars        : stats_results.csv, measure=='wta',
        comparison=='patient_vs_control', column q_fdr (two-tailed BH).
        M1 -> LH panel, M2 -> RH panel.

Output: C_results/figures/fig_wta_territory.{png,pdf}
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Config ───────────────────────────────────────────────────────────────────
CATEGORIES = ['face', 'house', 'object', 'word']
CAT_COLORS = {'face': '#378ADD', 'house': '#D85A30',
              'object': '#639922', 'word': '#EF9F27'}

EXCLUDE     = ['sub-017']
EXCLUDE_SES = [('sub-108', 2)]

WTA_CSV   = '/user_data/csimmon2/sym_pt/group_results/wta_percentages.csv'
STATS_CSV = '/user_data/csimmon2/git_repos/sym_pt/D_liu/stats_results.csv'
OUT_DIR   = Path('/user_data/csimmon2/git_repos/sym_pt/C_results/figures')

# M1 = LH panel, M2 = RH panel
MODEL_FOR_HEMI = {'l': 'M1_LH_ctrl_vs_pt', 'r': 'M2_RH_ctrl_vs_pt'}


# ── Session selection (matches 05) ────────────────────────────────────────────
def apply_exclusions(df):
    df = df[~df['subject_id'].isin(EXCLUDE)].copy()
    df['ses_num'] = pd.to_numeric(df['session'], errors='coerce').astype('Int64')
    for s, se in EXCLUDE_SES:
        df = df[~((df['subject_id'] == s) & (df['ses_num'] == se))]
    return df


def select_sessions(df):
    ctrl = df[df['status'] == 'control'].copy()
    fs = ctrl.groupby('subject_id')['ses_num'].min().rename('keep').reset_index()
    ctrl = ctrl.merge(fs, on='subject_id')
    ctrl = ctrl[ctrl['ses_num'] == ctrl['keep']].drop(columns='keep')
    pt = df[df['group'] == 'OTC'].copy()
    ls = pt.groupby('subject_id')['ses_num'].max().rename('keep').reset_index()
    pt = pt.merge(ls, on='subject_id')
    pt = pt[pt['ses_num'] == pt['keep']].drop(columns='keep')
    return pd.concat([ctrl, pt], ignore_index=True)


def stars(q):
    if pd.isna(q):
        return ''
    if q < .001:
        return '***'
    if q < .01:
        return '**'
    if q < .05:
        return '*'
    return ''


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wta', default=WTA_CSV)
    ap.add_argument('--stats', default=STATS_CSV)
    ap.add_argument('--out', default=str(OUT_DIR))
    args = ap.parse_args()

    # --- WTA territory data ---
    wta = pd.read_csv(args.wta)
    wta = wta[(wta['region'] == 'otc') & (wta['denominator'] == 'selective')].copy()
    wta = select_sessions(apply_exclusions(wta))

    # --- significance q's (two-tailed BH) from stats_results ---
    stats = pd.read_csv(args.stats)
    sw = stats[(stats['measure'] == 'wta') &
               (stats['comparison'] == 'patient_vs_control')]
    qmap = {}  # (hemi, category) -> q_fdr
    for hemi, model in MODEL_FOR_HEMI.items():
        rows = sw[sw['model'] == model]
        for _, r in rows.iterrows():
            qmap[(hemi, r['level'])] = r['q_fdr']

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(CATEGORIES))
    w = 0.38
    rng = np.random.RandomState(0)  # deterministic jitter

    for ax, h, hlabel in zip(axes, ['l', 'r'], ['LH-intact', 'RH-intact']):
        sub = wta[wta['hemi'] == h]
        max_top = 0
        for offset, grp, alpha in [(-w / 2, 'control', 1.0), (+w / 2, 'OTC', 0.45)]:
            means, sems = [], []
            for c in CATEGORIES:
                vals = sub[(sub['category'] == c) & (sub['group'] == grp)]['wta_pct']
                means.append(vals.mean())
                sems.append(vals.sem())
            ax.bar(x + offset, means, w, yerr=sems,
                   color=[CAT_COLORS[c] for c in CATEGORIES],
                   alpha=alpha, edgecolor='black', linewidth=0.6, capsize=3,
                   zorder=2)
            # per-subject open circles
            for xi, c in enumerate(CATEGORIES):
                vals = sub[(sub['category'] == c) & (sub['group'] == grp)]['wta_pct'].values
                jitter = (rng.rand(len(vals)) - 0.5) * (w * 0.6)
                ax.scatter(xi + offset + jitter, vals, s=22,
                           facecolors='none',
                           edgecolors=CAT_COLORS[c], linewidths=1.0,
                           alpha=0.9, zorder=3)
                if len(vals):
                    max_top = max(max_top, vals.max())

        # significance brackets (stars from q_fdr)
        bracket_y = max_top * 1.04
        bracket_h = max_top * 0.015
        for xi, c in enumerate(CATEGORIES):
            s = stars(qmap.get((h, c), np.nan))
            if s:
                x_l, x_r = xi - w / 2, xi + w / 2
                ax.plot([x_l, x_l, x_r, x_r],
                        [bracket_y, bracket_y + bracket_h,
                         bracket_y + bracket_h, bracket_y],
                        color='black', lw=1, zorder=4)
                ax.text(xi, bracket_y + bracket_h, s, ha='center', va='bottom',
                        color='red', fontsize=14, fontweight='bold', zorder=4)

        ax.set_ylim(0, max_top * 1.22)
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES, fontsize=10)
        ax.set_title(f'{hlabel} hemisphere VOTC', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if h == 'l':
            ax.set_ylabel('% of selective voxels (WTA)', fontsize=10)

    style_handles = [
        Patch(facecolor='gray', edgecolor='black', label='Controls'),
        Patch(facecolor='gray', alpha=0.45, edgecolor='black', label='Patients'),
    ]
    fig.legend(handles=style_handles, ncol=2, frameon=False, fontsize=10,
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('WTA territory: % of selective voxels per category   '
                 '(two-tailed BH-FDR: * q<.05, ** q<.01, *** q<.001)', fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig_wta_territory.{ext}', dpi=300, bbox_inches='tight')
    print(f'Saved: {out}/fig_wta_territory.png  (+ .pdf)')


if __name__ == '__main__':
    main()