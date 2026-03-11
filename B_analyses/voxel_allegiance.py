#!/usr/bin/env python3
# %% [markdown]
# # Voxel Allegiance Analysis
# 
# Explores three approaches to quantifying voxel-level category reorganization in VOTC:
# 1. **Winner-take-all maps** — each voxel colored by its preferred category
# 2. **Transition matrices** — which categories trade territory T1→T2
# 3. **Profile stability** — cosine similarity of voxel response vectors across time
#
# Also computes Blauch-style summed selectivity per category per hemisphere.
#
# COPEs used:
# - Raw betas: face=15, house=16, object=17, word=18
# - Category vs all: face=6, house=7, object=8, word=9
# - Differential: face=1(>obj), house=2(>obj), object=3(>scr), word=4(>obj)

# %%
import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import (processed_dir, skip_subs, get_sessions,
                           is_patient, get_sub_info, _load_csv)

# %% [markdown]
# ## Configuration

# %%
# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = ['face', 'house', 'object', 'word']
CAT_COLORS = {
    'face':   '#E74C3C',   # red
    'house':  '#3498DB',   # blue
    'object': '#2ECC71',   # green
    'word':   '#F39C12',   # orange
    'none':   '#BDC3C7',   # gray (subthreshold)
}

# ── Cope maps ─────────────────────────────────────────────────────────────────
# Raw betas (no contrast — just activation to each category)
RAW_BETA_COPES = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

# Category vs mean of all others (selectivity)
CAT_VS_ALL_COPES = {'face': 6, 'house': 7, 'object': 8, 'word': 9}

# Differential contrasts (what pipeline already uses)
DIFFERENTIAL_COPES = {'face': 1, 'house': 2, 'object': 3, 'word': 4}

# ── Which cope set to use for allegiance ──────────────────────────────────────
# Options: 'raw_beta', 'cat_vs_all', 'differential'
# raw_beta:     most neutral — just activation magnitude per category
# cat_vs_all:   selectivity-based — already captures "preference"
# differential: what we use elsewhere, but face/house/word are all vs object
ALLEGIANCE_COPE_SET = 'raw_beta'   # <-- CHANGE THIS to try different approaches

# ── Thresholds ────────────────────────────────────────────────────────────────
# Minimum z-score for a voxel to be considered "active" in WTA map
# Set to 0 if using raw betas (they're not z-scored)
# For cat_vs_all or differential, consider z > 1.96 or 2.3
WTA_THRESHOLD = 0  # will adjust per cope set below

# ── Pre-surgery sessions to skip ──────────────────────────────────────────────
PRE_SURGERY_SESSIONS = {
    'sub-021': ['01'], 'sub-045': ['01'], 'sub-047': ['01'], 'sub-049': ['01'],
    'sub-070': ['01'], 'sub-073': ['01'], 'sub-081': ['01'], 'sub-086': ['01'],
}

# ── Exclusions ────────────────────────────────────────────────────────────────
EXCLUDE_SUBS = ['OTC108', 'control083', 'control085']


# %% [markdown]
# ## Helper Functions

# %%
def get_cope_map():
    """Return the cope mapping for the chosen allegiance set."""
    if ALLEGIANCE_COPE_SET == 'raw_beta':
        return RAW_BETA_COPES
    elif ALLEGIANCE_COPE_SET == 'cat_vs_all':
        return CAT_VS_ALL_COPES
    elif ALLEGIANCE_COPE_SET == 'differential':
        return DIFFERENTIAL_COPES
    else:
        raise ValueError(f"Unknown cope set: {ALLEGIANCE_COPE_SET}")


def load_subjects():
    """Load subject info, return dict keyed by sub ID."""
    df = _load_csv()
    subjects = {}
    for sub_clean in sorted(df['sub_clean'].unique()):
        if sub_clean in skip_subs:
            continue
        sid = f'sub-{sub_clean}'
        sessions = get_sessions(sub_clean)
        base = os.path.join(processed_dir, sid)
        if not sessions or not os.path.exists(base):
            continue
        info = get_sub_info(sub_clean, sessions[0])
        pt = is_patient(sub_clean)
        intact = info.get('intact_hemi', '')
        code = f"{info.get('group', '')}{sub_clean}"
        if code in EXCLUDE_SUBS:
            continue
        
        # Filter sessions: remove pre-surgical
        post_sessions = []
        for s in sessions:
            ses_str = f'{s:02d}'
            if sid.replace('sub-', '') in [k.replace('sub-', '') for k in PRE_SURGERY_SESSIONS]:
                key = sid
                if key in PRE_SURGERY_SESSIONS and ses_str in PRE_SURGERY_SESSIONS[key]:
                    continue
            post_sessions.append(ses_str)
        
        subjects[sid] = {
            'code': code,
            'sessions': post_sessions,
            'hemi': ('l' if intact == 'left' else 'r') if pt else None,
            'group': info.get('group', 'unknown'),
            'patient_status': 'patient' if pt else 'control',
            'intact_hemi': intact,
        }
    return subjects


def get_votc_mask(sub_id, first_ses, hemi):
    """
    Get a broad VOTC mask by combining all 4 category searchmasks.
    Alternative: use lVentral/rVentral from derivatives/rois if available.
    
    Returns: boolean mask array, affine
    """
    roi_dir = os.path.join(processed_dir, sub_id, f'ses-{first_ses}', 'ROIs')
    
    # Try combining category searchmasks (guarantees we stay in the same
    # anatomical territory as our other analyses)
    combined = None
    affine = None
    for cat in CATEGORIES:
        mask_path = os.path.join(roi_dir, f'{hemi}_{cat}_searchmask.nii.gz')
        if os.path.exists(mask_path):
            img = nib.load(mask_path)
            data = img.get_fdata() > 0
            if combined is None:
                combined = data.copy()
                affine = img.affine
            else:
                combined = combined | data
    
    if combined is None:
        # Fallback: try ventral mask
        ventral_dir = os.path.join(processed_dir, sub_id, f'ses-{first_ses}',
                                    'derivatives', 'rois')
        ventral_path = os.path.join(ventral_dir, f'{hemi}Ventral.nii.gz')
        if os.path.exists(ventral_path):
            img = nib.load(ventral_path)
            combined = img.get_fdata() > 0
            affine = img.affine
    
    return combined, affine


def load_category_betas(sub_id, session, first_ses, mask, cope_map):
    """
    Load beta/zstat values for all 4 categories within a mask.
    
    Returns: dict {category: 1D array of values within mask}, or None
    """
    feat_dir = os.path.join(processed_dir, sub_id, f'ses-{session}',
                            'derivatives', 'fsl', 'loc', 'HighLevel.gfeat')
    
    # Determine filename convention
    # First session: cope1.nii.gz / zstat1.nii.gz
    # Other sessions: cope1_ses{first_ses}.nii.gz / zstat1_ses{first_ses}.nii.gz
    if ALLEGIANCE_COPE_SET == 'raw_beta':
        stat_file = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'
        stat_dir = 'stats'
    else:
        stat_file = 'zstat1.nii.gz' if session == first_ses else f'zstat1_ses{first_ses}.nii.gz'
        stat_dir = 'stats'
    
    betas = {}
    for cat, cope_num in cope_map.items():
        fpath = os.path.join(feat_dir, f'cope{cope_num}.feat', stat_dir, stat_file)
        if not os.path.exists(fpath):
            return None
        img = nib.load(fpath)
        data = img.get_fdata()
        betas[cat] = data[mask]
    
    return betas


# %% [markdown]
# ## Load All Data

# %%
print("Loading subjects...")
subjects = load_subjects()
print(f"  {len(subjects)} subjects loaded")

# Identify longitudinal subjects (2+ post-surgical sessions)
long_subs = {sid: info for sid, info in subjects.items() 
             if len(info['sessions']) >= 2}
print(f"  {len(long_subs)} longitudinal subjects")

# Separate by group
ctrl_long = {s: i for s, i in long_subs.items() if i['patient_status'] == 'control'}
pt_long = {s: i for s, i in long_subs.items() if i['patient_status'] == 'patient'}
print(f"    Controls: {len(ctrl_long)}, Patients: {len(pt_long)}")

# Cross-sectional: everyone with at least 1 session
all_subs = subjects
ctrl_all = {s: i for s, i in all_subs.items() if i['patient_status'] == 'control'}
pt_all = {s: i for s, i in all_subs.items() if i['patient_status'] == 'patient'}
print(f"  Cross-sectional — Controls: {len(ctrl_all)}, Patients: {len(pt_all)}")


# %% [markdown]
# ## 1. Winner-Take-All Maps
# 
# For each voxel in the VOTC mask, assign the category with the highest
# response (raw beta or z-score). Optionally threshold to exclude
# low-responding voxels.

# %%
def compute_wta(betas, threshold=0):
    """
    Winner-take-all assignment for each voxel.
    
    Args:
        betas: dict {category: 1D array} — values per voxel within mask
        threshold: minimum value in ANY category for a voxel to be assigned
                   (set to 0 for raw betas, ~2.0 for z-stats)
    
    Returns:
        winners: 1D array of category indices (0-3) or -1 for subthreshold
        margin: 1D array — difference between 1st and 2nd highest
        max_vals: 1D array — value of the winning category
    """
    n_voxels = len(betas[CATEGORIES[0]])
    beta_matrix = np.column_stack([betas[cat] for cat in CATEGORIES])  # (n_voxels, 4)
    
    max_vals = np.max(beta_matrix, axis=1)
    winners = np.argmax(beta_matrix, axis=1)
    
    # Compute margin (1st - 2nd place)
    sorted_betas = np.sort(beta_matrix, axis=1)
    margin = sorted_betas[:, -1] - sorted_betas[:, -2]
    
    # Apply threshold: if max activation is below threshold, mark as unassigned
    if threshold > 0:
        winners[max_vals < threshold] = -1
    
    return winners, margin, max_vals


def wta_summary(winners):
    """Count voxels per category from WTA assignment."""
    counts = {}
    for i, cat in enumerate(CATEGORIES):
        counts[cat] = int(np.sum(winners == i))
    counts['none'] = int(np.sum(winners == -1))
    total = len(winners)
    pcts = {k: v / total * 100 for k, v in counts.items()}
    return counts, pcts


# %%
# ── Compute WTA for all subjects, first available session ─────────────────────

cope_map = get_cope_map()
threshold = WTA_THRESHOLD if ALLEGIANCE_COPE_SET == 'raw_beta' else 1.96

print(f"Cope set: {ALLEGIANCE_COPE_SET}")
print(f"Threshold: {threshold}")
print(f"Copes: {cope_map}")
print()

wta_results = {}

for sid, info in subjects.items():
    first_ses = info['sessions'][0]
    hemi = info['hemi'] if info['patient_status'] == 'patient' else 'l'  # default L for controls
    
    # For controls, do both hemispheres
    hemis = ['l', 'r'] if info['patient_status'] == 'control' else [hemi]
    
    for h in hemis:
        mask, affine = get_votc_mask(sid, first_ses, h)
        if mask is None:
            continue
        
        betas = load_category_betas(sid, first_ses, first_ses, mask, cope_map)
        if betas is None:
            continue
        
        winners, margin, max_vals = compute_wta(betas, threshold)
        counts, pcts = wta_summary(winners)
        
        wta_results[(sid, first_ses, h)] = {
            'winners': winners,
            'margin': margin,
            'max_vals': max_vals,
            'counts': counts,
            'pcts': pcts,
            'n_voxels': len(winners),
            'group': info['patient_status'],
            'info': info,
        }

print(f"Computed WTA for {len(wta_results)} subject-hemisphere entries")


# %%
# ── Summarize WTA territory: Controls vs Patients ────────────────────────────

def summarize_group_wta(wta_results, group, hemi=None):
    """Average WTA percentages across subjects in a group."""
    pct_rows = []
    for (sid, ses, h), res in wta_results.items():
        if res['group'] != group:
            continue
        if hemi is not None and h != hemi:
            continue
        pct_rows.append(res['pcts'])
    
    if not pct_rows:
        return None
    
    df = pd.DataFrame(pct_rows)
    return df.describe()

print("=== CONTROLS — LEFT HEMISPHERE ===")
ctrl_L_summary = summarize_group_wta(wta_results, 'control', 'l')
if ctrl_L_summary is not None:
    print(ctrl_L_summary.loc[['mean', 'std']].round(1))

print("\n=== CONTROLS — RIGHT HEMISPHERE ===")
ctrl_R_summary = summarize_group_wta(wta_results, 'control', 'r')
if ctrl_R_summary is not None:
    print(ctrl_R_summary.loc[['mean', 'std']].round(1))

print("\n=== PATIENTS — INTACT HEMISPHERE ===")
pt_summary = summarize_group_wta(wta_results, 'patient')
if pt_summary is not None:
    print(pt_summary.loc[['mean', 'std']].round(1))


# %%
# ── Bar chart: Average territory per category ─────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, (title, group, hemi) in zip(axes, [
    ('Controls LH', 'control', 'l'),
    ('Controls RH', 'control', 'r'),
    ('Patients (intact)', 'patient', None),
]):
    pct_rows = []
    for (sid, ses, h), res in wta_results.items():
        if res['group'] != group:
            continue
        if hemi is not None and h != hemi:
            continue
        pct_rows.append(res['pcts'])
    
    if not pct_rows:
        ax.set_title(f"{title}\n(no data)")
        continue
    
    df = pd.DataFrame(pct_rows)
    means = df[CATEGORIES].mean()
    sems = df[CATEGORIES].sem()
    
    bars = ax.bar(CATEGORIES, means, yerr=sems, capsize=5,
                  color=[CAT_COLORS[c] for c in CATEGORIES], edgecolor='black')
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('% of VOTC voxels')
    ax.set_ylim(0, 60)

plt.suptitle(f'Winner-Take-All Territory ({ALLEGIANCE_COPE_SET})', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, 'group_results', 
            f'wta_territory_{ALLEGIANCE_COPE_SET}.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved WTA territory figure")


# %% [markdown]
# ## 2. Longitudinal Transition Matrices
# 
# For longitudinal subjects, classify voxels at T1 and T2 and build a
# 4×4 transition matrix (rows=T1 category, cols=T2 category).
# Diagonal = retention, off-diagonal = switching.

# %%
def compute_transition_matrix(winners_t1, winners_t2, n_categories=4):
    """
    Build transition matrix from T1 winners to T2 winners.
    Only includes voxels assigned to a category at BOTH timepoints.
    
    Returns:
        trans_counts: (4, 4) count matrix
        trans_pct:    (4, 4) row-normalized percentage matrix
        retention:    scalar — proportion of voxels retaining same category
    """
    # Only include voxels that are assigned (not -1) at both timepoints
    valid = (winners_t1 >= 0) & (winners_t2 >= 0)
    w1 = winners_t1[valid]
    w2 = winners_t2[valid]
    
    trans = np.zeros((n_categories, n_categories), dtype=int)
    for i in range(len(w1)):
        trans[w1[i], w2[i]] += 1
    
    # Row-normalize
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    trans_pct = trans / row_sums * 100
    
    # Overall retention
    total_valid = valid.sum()
    retention = np.diag(trans).sum() / total_valid if total_valid > 0 else np.nan
    
    return trans, trans_pct, retention


# %%
# ── Compute transitions for longitudinal subjects ────────────────────────────

transition_results = {}

for sid, info in long_subs.items():
    sessions = info['sessions']
    t1_ses, t2_ses = sessions[0], sessions[-1]  # first and last session
    first_ses = sessions[0]
    
    hemi = info['hemi'] if info['patient_status'] == 'patient' else None
    hemis = [hemi] if hemi else ['l', 'r']
    
    for h in hemis:
        mask, affine = get_votc_mask(sid, first_ses, h)
        if mask is None:
            continue
        
        betas_t1 = load_category_betas(sid, t1_ses, first_ses, mask, cope_map)
        betas_t2 = load_category_betas(sid, t2_ses, first_ses, mask, cope_map)
        if betas_t1 is None or betas_t2 is None:
            continue
        
        winners_t1, margin_t1, _ = compute_wta(betas_t1, threshold)
        winners_t2, margin_t2, _ = compute_wta(betas_t2, threshold)
        
        trans, trans_pct, retention = compute_transition_matrix(winners_t1, winners_t2)
        
        transition_results[(sid, h)] = {
            'trans_counts': trans,
            'trans_pct': trans_pct,
            'retention': retention,
            'winners_t1': winners_t1,
            'winners_t2': winners_t2,
            'margin_t1': margin_t1,
            'margin_t2': margin_t2,
            'group': info['patient_status'],
            't1_ses': t1_ses,
            't2_ses': t2_ses,
            'info': info,
        }

print(f"Computed transitions for {len(transition_results)} subject-hemisphere entries")


# %%
# ── Average transition matrix: Controls vs Patients ──────────────────────────

def avg_transition_matrix(results, group, hemi=None):
    """Average the percentage transition matrices across a group."""
    matrices = []
    sids = []
    for (sid, h), res in results.items():
        if res['group'] != group:
            continue
        if hemi is not None and h != hemi:
            continue
        matrices.append(res['trans_pct'])
        sids.append(sid)
    
    if not matrices:
        return None, []
    return np.mean(matrices, axis=0), sids


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (title, group, hemi) in zip(axes, [
    ('Controls LH', 'control', 'l'),
    ('Controls RH', 'control', 'r'),
    ('Patients (intact)', 'patient', None),
]):
    avg_mat, sids = avg_transition_matrix(transition_results, group, hemi)
    if avg_mat is None:
        ax.set_title(f"{title}\n(no data)")
        continue
    
    im = ax.imshow(avg_mat, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_yticklabels(CATEGORIES, fontsize=11)
    ax.set_xlabel('T2 category')
    ax.set_ylabel('T1 category')
    ax.set_title(f'{title} (n={len(sids)})', fontsize=13)
    
    # Annotate cells
    for i in range(4):
        for j in range(4):
            color = 'white' if avg_mat[i, j] > 50 else 'black'
            ax.text(j, i, f'{avg_mat[i, j]:.1f}%', ha='center', va='center',
                    fontsize=10, color=color)

plt.colorbar(im, ax=axes, shrink=0.8, label='% of T1 voxels')
plt.suptitle(f'Average Transition Matrix T1→T2 ({ALLEGIANCE_COPE_SET})', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, 'group_results',
            f'transition_matrix_{ALLEGIANCE_COPE_SET}.png'), dpi=150, bbox_inches='tight')
plt.show()


# %%
# ── Retention rates: Controls vs Patients ─────────────────────────────────────

retention_rows = []
for (sid, h), res in transition_results.items():
    # Per-category retention (diagonal of percentage matrix)
    for i, cat in enumerate(CATEGORIES):
        row_total = res['trans_counts'][i, :].sum()
        if row_total > 0:
            cat_retention = res['trans_counts'][i, i] / row_total
        else:
            cat_retention = np.nan
        retention_rows.append({
            'sub': sid,
            'hemi': h,
            'group': res['group'],
            'category': cat,
            'retention': cat_retention,
            'overall_retention': res['retention'],
        })

ret_df = pd.DataFrame(retention_rows)

print("=== RETENTION RATES (proportion voxels keeping same category T1→T2) ===\n")
print("Overall retention:")
print(ret_df.groupby('group')['overall_retention'].describe()[['mean', 'std', 'min', 'max']].round(3))

print("\nPer-category retention:")
pivot = ret_df.pivot_table(values='retention', index='group', columns='category',
                           aggfunc=['mean', 'std']).round(3)
print(pivot)


# %% [markdown]
# ## 3. Voxel Profile Stability
# 
# Instead of hard WTA assignment, treat each voxel as a 4-element response
# vector [β_face, β_house, β_object, β_word]. Compute cosine similarity
# between T1 and T2 vectors. This captures gradual reorganization without
# threshold sensitivity.

# %%
def compute_profile_stability(betas_t1, betas_t2):
    """
    For each voxel, compute cosine similarity of its 4-category response
    profile between T1 and T2.
    
    Returns:
        cos_sims: 1D array of cosine similarities per voxel
        mean_cos: scalar mean
        median_cos: scalar median
    """
    mat_t1 = np.column_stack([betas_t1[cat] for cat in CATEGORIES])  # (n_voxels, 4)
    mat_t2 = np.column_stack([betas_t2[cat] for cat in CATEGORIES])  # (n_voxels, 4)
    
    # Cosine similarity per voxel
    # cos_sim = dot(v1, v2) / (||v1|| * ||v2||)
    dot_products = np.sum(mat_t1 * mat_t2, axis=1)
    norms_t1 = np.linalg.norm(mat_t1, axis=1)
    norms_t2 = np.linalg.norm(mat_t2, axis=1)
    
    # Avoid division by zero
    denom = norms_t1 * norms_t2
    valid = denom > 0
    cos_sims = np.full(len(dot_products), np.nan)
    cos_sims[valid] = dot_products[valid] / denom[valid]
    
    return cos_sims, np.nanmean(cos_sims), np.nanmedian(cos_sims)


# %%
# ── Compute profile stability for longitudinal subjects ──────────────────────

stability_results = {}

for sid, info in long_subs.items():
    sessions = info['sessions']
    t1_ses, t2_ses = sessions[0], sessions[-1]
    first_ses = sessions[0]
    
    hemi = info['hemi'] if info['patient_status'] == 'patient' else None
    hemis = [hemi] if hemi else ['l', 'r']
    
    for h in hemis:
        mask, affine = get_votc_mask(sid, first_ses, h)
        if mask is None:
            continue
        
        betas_t1 = load_category_betas(sid, t1_ses, first_ses, mask, cope_map)
        betas_t2 = load_category_betas(sid, t2_ses, first_ses, mask, cope_map)
        if betas_t1 is None or betas_t2 is None:
            continue
        
        cos_sims, mean_cos, median_cos = compute_profile_stability(betas_t1, betas_t2)
        
        stability_results[(sid, h)] = {
            'cos_sims': cos_sims,
            'mean_cos': mean_cos,
            'median_cos': median_cos,
            'group': info['patient_status'],
            'info': info,
        }

print(f"Computed profile stability for {len(stability_results)} entries")


# %%
# ── Profile stability: Controls vs Patients ──────────────────────────────────

stab_rows = []
for (sid, h), res in stability_results.items():
    stab_rows.append({
        'sub': sid,
        'hemi': h,
        'group': res['group'],
        'mean_cos_sim': res['mean_cos'],
        'median_cos_sim': res['median_cos'],
    })

stab_df = pd.DataFrame(stab_rows)

print("=== PROFILE STABILITY (cosine similarity T1 vs T2) ===\n")
print(stab_df.groupby('group')[['mean_cos_sim', 'median_cos_sim']].describe().round(3))


# %%
# ── Histogram of cosine similarities: Controls vs Patients ───────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (title, group) in zip(axes, [('Controls', 'control'), ('Patients', 'patient')]):
    all_cos = []
    for (sid, h), res in stability_results.items():
        if res['group'] != group:
            continue
        all_cos.append(res['cos_sims'])
    
    if not all_cos:
        ax.set_title(f"{title} (no data)")
        continue
    
    pooled = np.concatenate(all_cos)
    pooled = pooled[np.isfinite(pooled)]
    
    ax.hist(pooled, bins=50, range=(-1, 1), density=True, alpha=0.7, 
            color='steelblue', edgecolor='black')
    ax.axvline(np.mean(pooled), color='red', linestyle='--', label=f'mean={np.mean(pooled):.3f}')
    ax.set_xlabel('Cosine similarity (T1 vs T2)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} (n={len(all_cos)} hemi)', fontsize=13)
    ax.legend()

plt.suptitle(f'Voxel Profile Stability ({ALLEGIANCE_COPE_SET})', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, 'group_results',
            f'profile_stability_{ALLEGIANCE_COPE_SET}.png'), dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 4. Blauch-Style Summed Selectivity
# 
# For each category, sum the positive selectivity values across all VOTC voxels.
# This captures both territory (how many voxels) and magnitude (how strongly)
# without forcing a binary assignment.

# %%
def compute_summed_selectivity(betas, mask_size):
    """
    Blauch (2025) style: for each category, sum positive values across all
    VOTC voxels. Gives a continuous measure of territory × magnitude.
    
    Returns: dict {category: summed_selectivity}
    """
    result = {}
    for cat in CATEGORIES:
        vals = betas[cat]
        positive_vals = vals[vals > 0]
        result[cat] = {
            'summed_sel': float(np.sum(positive_vals)),
            'n_positive': int(len(positive_vals)),
            'pct_positive': float(len(positive_vals) / mask_size * 100) if mask_size > 0 else 0,
            'mean_positive': float(np.mean(positive_vals)) if len(positive_vals) > 0 else 0,
        }
    return result


# %%
# ── Compute summed selectivity: all subjects, all sessions ───────────────────
# Using cat_vs_all copes (6-9) for selectivity, regardless of ALLEGIANCE_COPE_SET

sumsel_cope_map = CAT_VS_ALL_COPES  # Always use cat-vs-all for this measure

sumsel_rows = []

for sid, info in long_subs.items():
    sessions = info['sessions']
    first_ses = sessions[0]
    hemi = info['hemi'] if info['patient_status'] == 'patient' else None
    hemis = [hemi] if hemi else ['l', 'r']
    
    for h in hemis:
        mask, affine = get_votc_mask(sid, first_ses, h)
        if mask is None:
            continue
        mask_size = int(mask.sum())
        
        for ses in sessions:
            betas = load_category_betas(sid, ses, first_ses, mask, sumsel_cope_map)
            if betas is None:
                continue
            
            ss = compute_summed_selectivity(betas, mask_size)
            
            for cat, vals in ss.items():
                sumsel_rows.append({
                    'sub': sid,
                    'ses': ses,
                    'hemi': h,
                    'group': info['patient_status'],
                    'category': cat,
                    **vals,
                })

sumsel_df = pd.DataFrame(sumsel_rows)
if not sumsel_df.empty:
    print("=== SUMMED SELECTIVITY (cat_vs_all copes) ===\n")
    print("Longitudinal subjects, T1 vs T2:")
    
    # For each subject, get T1 and T2
    for sid in sorted(long_subs.keys()):
        sub_df = sumsel_df[sumsel_df['sub'] == sid]
        if sub_df.empty:
            continue
        sessions = sorted(sub_df['ses'].unique())
        if len(sessions) < 2:
            continue
        t1_df = sub_df[sub_df['ses'] == sessions[0]].set_index('category')['summed_sel']
        t2_df = sub_df[sub_df['ses'] == sessions[-1]].set_index('category')['summed_sel']
        
        group = long_subs[sid]['patient_status']
        print(f"\n{sid} ({group}):")
        for cat in CATEGORIES:
            t1_val = t1_df.get(cat, np.nan)
            t2_val = t2_df.get(cat, np.nan)
            change = t2_val - t1_val if np.isfinite(t1_val) and np.isfinite(t2_val) else np.nan
            print(f"  {cat:8s}: T1={t1_val:8.1f}  T2={t2_val:8.1f}  Δ={change:+8.1f}")


# %% [markdown]
# ## 5. Individual Patient Transition Matrices
# 
# Show each patient's transition matrix separately — since n=5,
# individual patterns matter.

# %%
patient_trans = {k: v for k, v in transition_results.items() if v['group'] == 'patient'}

if patient_trans:
    n_patients = len(patient_trans)
    fig, axes = plt.subplots(1, n_patients, figsize=(5 * n_patients, 4.5))
    if n_patients == 1:
        axes = [axes]
    
    for ax, ((sid, h), res) in zip(axes, patient_trans.items()):
        im = ax.imshow(res['trans_pct'], cmap='Oranges', vmin=0, vmax=100)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CATEGORIES, fontsize=9)
        ax.set_yticklabels(CATEGORIES, fontsize=9)
        ax.set_xlabel('T2')
        ax.set_ylabel('T1')
        ax.set_title(f'{sid} ({h}H)\nretention={res["retention"]:.2f}', fontsize=11)
        
        for i in range(4):
            for j in range(4):
                color = 'white' if res['trans_pct'][i, j] > 50 else 'black'
                ax.text(j, i, f'{res["trans_pct"][i, j]:.0f}%', 
                        ha='center', va='center', fontsize=9, color=color)
    
    plt.suptitle('Individual Patient Transition Matrices', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, 'group_results',
                f'patient_transitions_{ALLEGIANCE_COPE_SET}.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()


# %% [markdown]
# ## 6. Margin Analysis
# 
# The "happy medium" between WTA and full profile: examine how CONFIDENT
# the WTA assignment is. Voxels with large margins are clearly committed;
# voxels with small margins are contested territory.

# %%
# ── Distribution of WTA margins: Controls vs Patients ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (title, group) in zip(axes, [('Controls', 'control'), ('Patients', 'patient')]):
    all_margins = {cat: [] for cat in CATEGORIES}
    
    for (sid, ses, h), res in wta_results.items():
        if res['group'] != group:
            continue
        for i, cat in enumerate(CATEGORIES):
            cat_mask = res['winners'] == i
            if cat_mask.sum() > 0:
                all_margins[cat].append(res['margin'][cat_mask])
    
    positions = []
    data = []
    colors = []
    for i, cat in enumerate(CATEGORIES):
        if all_margins[cat]:
            pooled = np.concatenate(all_margins[cat])
            data.append(pooled)
            positions.append(i)
            colors.append(CAT_COLORS[cat])
    
    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(CATEGORIES)
    
    ax.set_ylabel('WTA Margin (1st - 2nd)')
    ax.set_title(f'{title}', fontsize=13)

plt.suptitle(f'WTA Confidence Margins ({ALLEGIANCE_COPE_SET})', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, 'group_results',
            f'wta_margins_{ALLEGIANCE_COPE_SET}.png'), dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 7. Change in Margin (Longitudinal)
# 
# Do voxels become more or less committed over time?
# Compare average margin at T1 vs T2 per category.

# %%
margin_change_rows = []

for (sid, h), res in transition_results.items():
    for i, cat in enumerate(CATEGORIES):
        # Voxels that were this category at T1
        cat_at_t1 = res['winners_t1'] == i
        if cat_at_t1.sum() == 0:
            continue
        
        mean_margin_t1 = np.mean(res['margin_t1'][cat_at_t1])
        
        # For those same voxels, what's their margin at T2?
        # (regardless of whether they switched)
        mean_margin_t2 = np.mean(res['margin_t2'][cat_at_t1])
        
        margin_change_rows.append({
            'sub': sid,
            'hemi': h,
            'group': res['group'],
            'category': cat,
            'mean_margin_t1': mean_margin_t1,
            'mean_margin_t2': mean_margin_t2,
            'margin_change': mean_margin_t2 - mean_margin_t1,
            'n_voxels': int(cat_at_t1.sum()),
        })

margin_df = pd.DataFrame(margin_change_rows)
if not margin_df.empty:
    print("=== MARGIN CHANGE T1→T2 ===\n")
    print("Positive = voxels became MORE committed, Negative = LESS committed\n")
    pivot = margin_df.pivot_table(values='margin_change', index='group', 
                                   columns='category', aggfunc='mean').round(3)
    print(pivot)


# %% [markdown]
# ## 8. Summary Table
# 
# Compile key metrics per subject for easy export.

# %%
summary_rows = []

for sid, info in long_subs.items():
    hemi = info['hemi'] if info['patient_status'] == 'patient' else None
    hemis = [hemi] if hemi else ['l', 'r']
    
    for h in hemis:
        row = {
            'sub': sid,
            'hemi': h,
            'group': info['patient_status'],
        }
        
        # Transition retention
        if (sid, h) in transition_results:
            row['overall_retention'] = transition_results[(sid, h)]['retention']
            for i, cat in enumerate(CATEGORIES):
                tc = transition_results[(sid, h)]['trans_counts']
                row_total = tc[i, :].sum()
                row[f'{cat}_retention'] = tc[i, i] / row_total if row_total > 0 else np.nan
        
        # Profile stability
        if (sid, h) in stability_results:
            row['mean_cos_sim'] = stability_results[(sid, h)]['mean_cos']
            row['median_cos_sim'] = stability_results[(sid, h)]['median_cos']
        
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
out_path = os.path.join(processed_dir, 'group_results',
                         f'voxel_allegiance_summary_{ALLEGIANCE_COPE_SET}.csv')
summary_df.to_csv(out_path, index=False)
print(f"Saved summary table: {out_path}")
print()
print(summary_df.to_string(index=False))


# %% [markdown]
# ## Notes on Interpretation
# 
# **Cope set matters:**
# - `raw_beta`: Each voxel's raw BOLD response to each category. Most neutral,
#   but all categories will have positive values in most voxels (everything
#   activates VOTC to some degree), making margins small.
# - `cat_vs_all`: Selectivity relative to mean of other categories. Better 
#   for "allegiance" — positive means this category wins, negative means others
#   win. Naturally produces clearer winners.
# - `differential`: What the pipeline already uses, but face/house/word are all
#   relative to object, so object's "selectivity" is actually object > scramble.
#   Not ideal for direct comparison across categories.
#
# **Recommendation:** Run this notebook with all three cope sets and compare.
# `cat_vs_all` is probably the most appropriate for allegiance claims.
# `raw_beta` is the most honest but noisiest.
#
# **The problem you identified:** With `raw_beta`, a voxel responding 
# β=3.2 face, β=2.8 object has a margin of only 0.4. A small amount of noise
# flips it. The margin analysis (Section 6) quantifies exactly this — if most
# voxels have tiny margins, WTA is misleading. If margins are large, WTA is
# informative. This is the diagnostic to check.
#
# **Connection to Blauch:** The summed selectivity (Section 4) is the 
# territory-level version — how much total face/house/object/word selectivity
# exists in VOTC, without forcing each voxel to pick sides. Changes in summed
# selectivity T1→T2 tell you if categories are gaining or losing territory and
# amplitude, which is the same question the voxel-level analysis answers from
# a different angle.

# %%
print("\n=== DONE ===")
print(f"Cope set used: {ALLEGIANCE_COPE_SET}")
print(f"All outputs saved to: {processed_dir}/group_results/")
print("\nTo run with a different cope set, change ALLEGIANCE_COPE_SET at the top")
print("Options: 'raw_beta', 'cat_vs_all', 'differential'")
