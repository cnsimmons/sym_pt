#!/usr/bin/env python3
"""
fig_cross_sectional_slide1.py
=============================
Slide 1 Figure: Cross-Sectional Univariate Selectivity Results

Visual style: Horizontal gray bars = control 95% CI (bootstrap, 10K iterations,
without replacement). Individual OTC patient values overlaid as colored circles.
Anatomical homolog (primary): L-res (intact RH) vs Ctrl RH, R-res (intact LH) vs Ctrl LH.

Panels:
  A. Sum selectivity (normalized) — log scale
  B. Number of selective voxels (mm³)
  C. Mean activation (z)

Each panel has four rows (Face, House, Object, Word). Each row has two sub-rows:
  - Top sub-row: L-resection patients (intact RH, teal) vs Control RH CI
  - Bottom sub-row: R-resection patients (intact LH, peach) vs Control LH CI

Patient markers that fall outside the control CI are the "hits" — but the overall
message of this figure is that selectivity is largely comparable to controls.

Data source: selectivity_summary.csv from the Ayzenberg-inspired pipeline.
Statistical method: Bootstrap 95% CI from controls (Ayzenberg method, 10K iterations,
sampling n-matched to patient group size, without replacement).

Output: /user_data/csimmon2/git_repos/sym_pt/C_results/figures/slide1_cross_sectional_univariate.png

Usage:
  conda activate fmri
  python fig_cross_sectional_slide1.py
"""

import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Font setup (Arial unavailable on cluster, fallback chain) ─────────────────
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 11

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE     = Path(processed_dir)
SEL_DIR  = BASE / 'group_results' / 'selectivity'
FIG_DIR  = Path('/user_data/csimmon2/git_repos/sym_pt/C_results/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

COPE_SET   = 'differential'
EXCLUDE    = ['sub-017']          # Polymicrogyria — never included
CATEGORIES = ['face', 'house', 'object', 'word']
N_ITER     = 10_000
RNG        = np.random.default_rng(42)

# Panel definitions: (metric_column, title, xlabel, log_scale)
PANELS = [
    ('sum_selec_norm', 'A  Sum Selectivity (normalized)',
     'Sum Selectivity (norm.)', True),
    ('volume', 'B  No. Selective Voxels (mm³)',
     'No. Selective Voxels (mm³)', True),
    ('mean_act', 'C  Mean Activation',
     'Mean Activation (z)', True),
]

# ── Colors & markers ──────────────────────────────────────────────────────────
# PTOC paper palette: outline at full color, fill at reduced opacity
CTRL_CI_COLOR = '#d9d9d9'         # Light gray for control CI bars
LRES_HEX      = '#4ac0c0'         # Teal — L-resection (intact RH)
RRES_HEX      = '#ff9b83'         # Peach — R-resection (intact LH)
PT_MARKER     = 'o'               # Circles for both groups
PT_SIZE       = 20                # Small enough not to overwhelm
PT_EDGE_W     = 1.2               # Thicker edge so outline color reads clearly
PT_FILL_ALPHA = 0.45              # Fill opacity (outline stays at 1.0)

# ── Layout ────────────────────────────────────────────────────────────────────
BAR_HEIGHT    = 0.25              # Height of each CI bar
ROW_SPACING   = 1.4              # More space between categories
SUBROW_OFFSET = BAR_HEIGHT / 2 + 0.06  # Offset between L-res and R-res sub-rows
FIG_WIDTH     = 7
PANEL_HEIGHT  = 3.8              # Taller panels to accommodate spacing


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print('Loading data...')

sel_file = SEL_DIR / 'selectivity_summary.csv'
if not sel_file.exists():
    print(f'ERROR: {sel_file} not found')
    sys.exit(1)

df_sel = pd.read_csv(sel_file)
df_sel = df_sel[~df_sel['sub'].isin(EXCLUDE)]
df_sel['ses_int'] = df_sel['ses'].astype(int)

# First session only → cross-sectional snapshot
first_ses = (df_sel.groupby('sub')['ses_int'].min()
             .reset_index().rename(columns={'ses_int': 'fs'}))
df_sel_cs = df_sel.merge(first_ses, on='sub')
df_sel_cs = df_sel_cs[df_sel_cs['ses_int'] == df_sel_cs['fs']].copy()

ctrl_cs = df_sel_cs[df_sel_cs['group'] == 'control']
otc_cs  = df_sel_cs[df_sel_cs['group'] == 'OTC']

# Diagnostic: confirm expected sample sizes
n_ctrl = ctrl_cs['sub'].nunique()
n_otc  = otc_cs['sub'].nunique()
n_lres = otc_cs[otc_cs['intact_hemi'] == 'right']['sub'].nunique()
n_rres = otc_cs[otc_cs['intact_hemi'] == 'left']['sub'].nunique()
print(f'Controls: {n_ctrl} | OTC: {n_otc} (L-res: {n_lres}, R-res: {n_rres})')

# Verify expected columns exist
for col in ['sum_selec_norm', 'volume', 'mean_act', 'category', 'hemi', 'intact_hemi']:
    if col not in df_sel_cs.columns:
        print(f'ERROR: Column "{col}" not found in selectivity CSV.')
        print(f'  Available columns: {list(df_sel_cs.columns)}')
        sys.exit(1)
print(f'Columns verified: {["sum_selec_norm", "volume", "mean_act"]}')


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sel_vals(df, cat, hemi):
    """Extract selectivity values for a category × hemisphere.
    
    For controls: hemi = 'left' or 'right' (actual hemisphere).
    For patients: hemi = 'intact_left' or 'intact_right' (filters by intact_hemi).
    """
    c = df[df['category'] == cat]
    if hemi in ['left', 'right']:
        return c[c['hemi'] == hemi]
    elif hemi == 'intact_left':
        return c[(c['intact_hemi'] == 'left') & (c['hemi'] == 'left')]
    elif hemi == 'intact_right':
        return c[(c['intact_hemi'] == 'right') & (c['hemi'] == 'right')]
    return c.iloc[0:0]


def bootstrap_ci(vals, n_draw=None, n_iter=N_ITER, rng=RNG):
    """Bootstrap 95% CI (Ayzenberg method: without replacement, n-matched).
    
    Args:
        vals: Control values to bootstrap from.
        n_draw: Number to draw per iteration (matched to patient group size).
        n_iter: Number of bootstrap iterations.
        rng: Random number generator.
    
    Returns:
        (lower_2.5, upper_97.5) percentiles, or (nan, nan) if insufficient data.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return np.nan, np.nan
    if n_draw is None:
        n_draw = n
    n_draw = min(n_draw, n)
    boot = np.array([
        rng.choice(v, size=n_draw, replace=False).mean()
        for _ in range(n_iter)
    ])
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def bootstrap_ci_replace(vals, n_iter=N_ITER, rng=RNG):
    """Bootstrap 95% CI WITH replacement — for patient groups (small n).
    
    Standard bootstrap: resample n values from n WITH replacement,
    compute mean each iteration. Returns 2.5th and 97.5th percentiles.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return np.nan, np.nan
    boot = np.array([
        rng.choice(v, size=n, replace=True).mean()
        for _ in range(n_iter)
    ])
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PANEL DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_panel_data(metric):
    """For one metric, compute control CIs and patient CIs per category.
    
    Returns list of dicts (one per category), each containing:
      - category: str
      - lres_ctrl_m: control RH mean
      - lres_ci: (lo, hi) — control RH CI for L-res comparison
      - lres_patients: [(sub_id, value), ...] — L-res patient values
      - lres_pt_m: patient mean (L-res)
      - lres_pt_ci: (lo, hi) — patient CI (L-res, bootstrapped from n=8)
      - lres_sig: bool — patient mean outside control CI
      (same for rres_*)
    """
    panel = []
    for cat in CATEGORIES:
        row = {'category': cat}

        # Control hemisphere distributions
        c_rh = sel_vals(ctrl_cs, cat, 'right')[metric].dropna().values
        c_lh = sel_vals(ctrl_cs, cat, 'left')[metric].dropna().values

        # ── L-resection patients (intact RH) vs Control RH ────────────────
        o_rh_df = sel_vals(otc_cs, cat, 'intact_right')
        o_rh_vals = o_rh_df[metric].dropna().values
        n_lres_cat = len(o_rh_df)
        lo_rh, hi_rh = bootstrap_ci(c_rh, n_draw=max(1, n_lres_cat))
        pt_m_rh = np.nanmean(o_rh_vals) if len(o_rh_vals) > 0 else np.nan
        pt_lo_rh, pt_hi_rh = bootstrap_ci_replace(o_rh_vals)  # Patient CI (with replacement)
        sig_rh = (pt_m_rh < lo_rh or pt_m_rh > hi_rh) if np.isfinite(pt_m_rh) else False

        row['lres_ctrl_m'] = np.nanmean(c_rh)
        row['lres_ci'] = (lo_rh, hi_rh)
        row['lres_patients'] = list(zip(o_rh_df['sub'].values, o_rh_vals))
        row['lres_pt_m'] = pt_m_rh
        row['lres_pt_ci'] = (pt_lo_rh, pt_hi_rh)
        row['lres_sig'] = sig_rh

        # ── R-resection patients (intact LH) vs Control LH ────────────────
        o_lh_df = sel_vals(otc_cs, cat, 'intact_left')
        o_lh_vals = o_lh_df[metric].dropna().values
        n_rres_cat = len(o_lh_df)
        lo_lh, hi_lh = bootstrap_ci(c_lh, n_draw=max(1, n_rres_cat))
        pt_m_lh = np.nanmean(o_lh_vals) if len(o_lh_vals) > 0 else np.nan
        pt_lo_lh, pt_hi_lh = bootstrap_ci_replace(o_lh_vals)  # Patient CI (with replacement)
        sig_lh = (pt_m_lh < lo_lh or pt_m_lh > hi_lh) if np.isfinite(pt_m_lh) else False

        row['rres_ctrl_m'] = np.nanmean(c_lh)
        row['rres_ci'] = (lo_lh, hi_lh)
        row['rres_patients'] = list(zip(o_lh_df['sub'].values, o_lh_vals))
        row['rres_pt_m'] = pt_m_lh
        row['rres_pt_ci'] = (pt_lo_lh, pt_hi_lh)
        row['rres_sig'] = sig_lh

        panel.append(row)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_panel(ax, panel_data, title, xlabel, log_scale=False):
    """Plot one panel: horizontal CI bars with patient dots overlaid.
    
    Layout per category:
      Top sub-row:    L-resection (teal) vs Control RH CI
      Bottom sub-row: R-resection (peach) vs Control LH CI
    """
    n_cats = len(panel_data)

    for i, row in enumerate(panel_data):
        cat = row['category'].capitalize()
        y_center = (n_cats - 1 - i) * ROW_SPACING
        y_top = y_center + SUBROW_OFFSET
        y_bot = y_center - SUBROW_OFFSET

        # ── L-resection sub-row (teal, top) ───────────────────────────────
        lo, hi = row['lres_ci']
        if np.isfinite(lo) and np.isfinite(hi):
            ax.barh(y_top, hi - lo, left=lo, height=BAR_HEIGHT,
                    color=CTRL_CI_COLOR, edgecolor='#bbb', linewidth=0.5,
                    zorder=1)

        for sub_id, val in row['lres_patients']:
            ax.scatter(val, y_top,
                      color=(*matplotlib.colors.to_rgb(LRES_HEX), PT_FILL_ALPHA),
                      marker=PT_MARKER, s=PT_SIZE,
                      edgecolors=LRES_HEX, linewidth=PT_EDGE_W,
                      zorder=3)

        # ── R-resection sub-row (peach, bottom) ──────────────────────────
        lo, hi = row['rres_ci']
        if np.isfinite(lo) and np.isfinite(hi):
            ax.barh(y_bot, hi - lo, left=lo, height=BAR_HEIGHT,
                    color=CTRL_CI_COLOR, edgecolor='#bbb', linewidth=0.5,
                    zorder=1)

        for sub_id, val in row['rres_patients']:
            ax.scatter(val, y_bot,
                      color=(*matplotlib.colors.to_rgb(RRES_HEX), PT_FILL_ALPHA),
                      marker=PT_MARKER, s=PT_SIZE,
                      edgecolors=RRES_HEX, linewidth=PT_EDGE_W,
                      zorder=3)

        # ── Category label (left of y-axis) ──────────────────────────────
        ax.text(-0.02, y_center, cat,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontweight='500', fontsize=11)

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if log_scale:
        ax.set_xscale('log')

    # Add some padding on y-axis
    ax.set_ylim(-0.7, (len(panel_data) - 1) * ROW_SPACING + 0.7)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def make_figure():
    """Generate the complete Slide 1 figure."""

    print('\nBuilding panel data...')

    # Build data for each panel
    panel_data = {}
    for metric, title, xlabel, log_scale in PANELS:
        panel_data[metric] = get_panel_data(metric)

    # ── Print summary table ───────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('SLIDE 1 TABLE: Cross-Sectional Univariate (Anatomical Homolog)')
    print('=' * 100)
    for metric, title, xlabel, log_scale in PANELS:
        prec = 1 if metric == 'mean_act' else 0
        print(f'\n  {title}')
        print(f'  {"":20} {"Face":>22} {"House":>22} {"Object":>22} {"Word":>22}')
        print(f'  {"-"*110}')

        # RH block: L-res (intact RH) vs Ctrl RH
        ctrl_line = '  RH: Ctrl (n=24)   '
        pt_line   = '  RH: OTC  (n=8)    '
        for row in panel_data[metric]:
            cm = row['lres_ctrl_m']
            lo, hi = row['lres_ci']
            pm = row['lres_pt_m']
            plo, phi = row['lres_pt_ci']
            sig = '*' if row['lres_sig'] else ''
            ctrl_line += f' {cm:>6.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]'
            pt_line   += f' {pm:>6.{prec}f} [{plo:.{prec}f}, {phi:.{prec}f}]{sig:>1}'
        print(ctrl_line)
        print(pt_line)

        # LH block: R-res (intact LH) vs Ctrl LH
        ctrl_line = '  LH: Ctrl (n=24)   '
        pt_line   = '  LH: OTC  (n=8)    '
        for row in panel_data[metric]:
            cm = row['rres_ctrl_m']
            lo, hi = row['rres_ci']
            pm = row['rres_pt_m']
            plo, phi = row['rres_pt_ci']
            sig = '*' if row['rres_sig'] else ''
            ctrl_line += f' {cm:>6.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]'
            pt_line   += f' {pm:>6.{prec}f} [{plo:.{prec}f}, {phi:.{prec}f}]{sig:>1}'
        print(ctrl_line)
        print(pt_line)

    print('\n* = OTC mean outside control 95% CI')
    print('RH = L-resection patients (intact RH) vs Control RH')
    print('LH = R-resection patients (intact LH) vs Control LH')

    # Create figure
    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(FIG_WIDTH, PANEL_HEIGHT * n_panels))

    for idx, (metric, title, xlabel, log_scale) in enumerate(PANELS):
        plot_panel(axes[idx], panel_data[metric], title, xlabel, log_scale=log_scale)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CTRL_CI_COLOR, edgecolor='#bbb',
                       label='Control 95% CI'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=(*matplotlib.colors.to_rgb(LRES_HEX), PT_FILL_ALPHA),
               markeredgecolor=LRES_HEX, markeredgewidth=1.2,
               markersize=7, label='L-resection (intact RH)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=(*matplotlib.colors.to_rgb(RRES_HEX), PT_FILL_ALPHA),
               markeredgecolor=RRES_HEX, markeredgewidth=1.2,
               markersize=7, label='R-resection (intact LH)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = FIG_DIR / 'slide1_cross_sectional_univariate.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {out_path}')
    plt.show()


if __name__ == '__main__':
    make_figure()