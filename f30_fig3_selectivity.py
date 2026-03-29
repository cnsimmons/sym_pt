#!/usr/bin/env python3
"""
F30 Figure 3: Univariate Selectivity (Aim 2)
Panel A: Mean activation | Panel B: Active volume
Subjects: sub-052 (TD control), sub-004 (OTC)
Grayscale palette. No control range bands. Arial, 300 DPI.
"""

import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Helvetica']

from pathlib import Path

BASE    = Path(processed_dir)
SEL_DIR = BASE / 'group_results' / 'selectivity'

# ── Load data ─────────────────────────────────────────────────────────────────
df_sel = pd.read_csv(SEL_DIR / 'selectivity_summary.csv')
df_sel['ses_int'] = df_sel['ses'].astype(int)

CATEGORIES = ['face', 'house', 'object', 'word']
TARGETS = {
    'sub-052': {'label': 'TD Control (sub-052)', 'color': '#4D4D4D', 'hatch': ''},
    'sub-004': {'label': 'OTC (sub-004)',         'color': '#BFBFBF', 'hatch': ''},
}
EDGE_COLOR = '#333333'

# ── Extract per-subject values ────────────────────────────────────────────────
data = {}

for sid, info in TARGETS.items():
    sub_data = df_sel[df_sel['sub'] == sid]
    if sub_data.empty:
        sub_data = df_sel[df_sel['sub'] == sid.replace('sub-', '')]
    if sub_data.empty:
        print(f"WARNING: {sid} not found in selectivity CSV")
        continue

    # First session
    first = sub_data['ses_int'].min()
    ses_data = sub_data[sub_data['ses_int'] == first]

    # For sub-004 (OTC): use intact hemisphere (left)
    if sid == 'sub-004':
        if 'intact_hemi' in ses_data.columns:
            intact = ses_data['intact_hemi'].iloc[0]
            if intact != 'control':
                ses_data = ses_data[ses_data['hemi'] == intact]
        else:
            ses_data = ses_data[ses_data['hemi'] == 'left']

    # For sub-052 (control): use preferred hemisphere per category
    # We'll pull each category from its preferred hemisphere
    vals_act = {}
    vals_vol = {}

    for cat in CATEGORIES:
        if sid == 'sub-052':
            # Preferred hemisphere
            pref = {'face': 'right', 'house': 'right', 'object': 'left', 'word': 'left'}
            cat_data = ses_data[(ses_data['category'] == cat) & (ses_data['hemi'] == pref[cat])]
        else:
            cat_data = ses_data[ses_data['category'] == cat]

        if len(cat_data) > 0:
            vals_act[cat] = cat_data['mean_act'].values[0]
            vals_vol[cat] = cat_data['volume'].values[0]
        else:
            vals_act[cat] = np.nan
            vals_vol[cat] = np.nan

    data[sid] = {'mean_act': vals_act, 'volume': vals_vol}

    print(f"{sid} (ses-{first:02d}):")
    print(f"  Mean act: {vals_act}")
    print(f"  Volume:   {vals_vol}")

# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))

x = np.arange(len(CATEGORIES))
width = 0.35

sids = list(TARGETS.keys())

# Panel A: Mean Activation
for i, sid in enumerate(sids):
    info = TARGETS[sid]
    vals = [data[sid]['mean_act'].get(cat, 0) for cat in CATEGORIES]
    offset = -width/2 + i * width
    ax_a.bar(x + offset, vals, width,
             color=info['color'], edgecolor=EDGE_COLOR, linewidth=1.2,
             label=info['label'], zorder=3)

ax_a.set_xticks(x)
ax_a.set_xticklabels([c.title() for c in CATEGORIES], fontsize=12)
ax_a.set_ylabel('Mean Activation (z)', fontsize=12)
ax_a.set_title('A.  Mean Activation', fontsize=13, fontweight='bold', loc='left')
ax_a.legend(fontsize=10, frameon=False)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_ylim(0, None)
ax_a.tick_params(axis='both', labelsize=11)

# Panel B: Active Volume
for i, sid in enumerate(sids):
    info = TARGETS[sid]
    vals = [data[sid]['volume'].get(cat, 0) for cat in CATEGORIES]
    offset = -width/2 + i * width
    ax_b.bar(x + offset, vals, width,
             color=info['color'], edgecolor=EDGE_COLOR, linewidth=1.2,
             label=info['label'], zorder=3)

ax_b.set_xticks(x)
ax_b.set_xticklabels([c.title() for c in CATEGORIES], fontsize=12)
ax_b.set_ylabel('Active Volume (voxels)', fontsize=12)
ax_b.set_title('B.  Active Volume', fontsize=13, fontweight='bold', loc='left')
ax_b.legend(fontsize=10, frameon=False)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.set_ylim(0, None)
ax_b.tick_params(axis='both', labelsize=11)

plt.tight_layout(w_pad=3)

out_path = BASE / 'group_results' / 'figures' / 'f30_fig3_selectivity_grayscale.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\nSaved: {out_path}")
