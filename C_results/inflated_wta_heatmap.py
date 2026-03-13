#!/usr/bin/env python3
"""
inflated_wta_heatmap.py — Per-category vote count heatmaps on fsaverage5.

For each vertex, shows how many controls (0–N) prefer that category.
Four panels per hemisphere (face, house, object, word), each in its own colormap.

Requires FSL loaded (for flirt). Run with:
  module load fsl/6.0.3
  python inflated_wta_heatmap.py
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, _load_csv

from nilearn import datasets, surface, plotting

# ── Configuration ─────────────────────────────────────────────────────────────

EXCLUDE = ['sub-017']
CATEGORIES = ['face', 'house', 'object', 'word']
CAT_CMAPS = {'face': 'Reds', 'house': 'Blues', 'object': 'Greens', 'word': 'Oranges'}

WTA_DIR = os.path.join(processed_dir, 'group_results', 'wta_maps')

FSLDIR = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
MNI_REF = os.path.join(FSLDIR, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

OUT_DIR = os.path.join(processed_dir, 'group_results', 'figures', 'inflated_wta')
os.makedirs(OUT_DIR, exist_ok=True)

fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
N_VERTICES = 10242


# ── Load subjects ─────────────────────────────────────────────────────────────

def load_subjects():
    df = _load_csv()
    subjects = {}
    for _, row in df.iterrows():
        sid = row['sub']
        if sid in EXCLUDE:
            continue
        if row.get('pre_post', 'post') == 'pre':
            continue
        if sid not in subjects:
            intact = row.get('intact_hemi', '')
            hemi_letter = 'l' if intact == 'left' else ('r' if intact == 'right' else '')
            subjects[sid] = {
                'sessions': [],
                'patient_status': row.get('group', 'control'),
                'surgery_side': row.get('surgery_side', 'na'),
                'hemi': hemi_letter,
                'intact_hemi': intact,
                'code': row.get('code', sid),
            }
        ses = str(row.get('ses_num', '')).zfill(2)
        if ses not in subjects[sid]['sessions']:
            subjects[sid]['sessions'].append(ses)
    for sid in subjects:
        subjects[sid]['sessions'] = sorted(subjects[sid]['sessions'])
    return subjects


# ── Helpers ───────────────────────────────────────────────────────────────────

def native_to_mni(native_nifti, anat2stand_mat):
    mni_out = tempfile.NamedTemporaryFile(suffix='_mni.nii.gz', delete=False).name
    cmd = [
        'flirt', '-in', str(native_nifti), '-ref', MNI_REF,
        '-applyxfm', '-init', str(anat2stand_mat),
        '-out', mni_out, '-interp', 'nearestneighbour',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FLIRT error: {result.stderr.strip()}")
        return None
    return mni_out


def project_to_surface(mni_nifti, hemi='left'):
    mesh = fsaverage[f'pial_{hemi}']
    return surface.vol_to_surf(mni_nifti, mesh, interpolation='nearest')


def get_anat2stand(sid, first_ses):
    for ses_try in [first_ses, '01']:
        path = os.path.join(processed_dir, sid, f'ses-{ses_try}',
                             'anat', 'anat2stand.mat')
        if os.path.exists(path):
            return path
    return None


# ── Accumulate votes ──────────────────────────────────────────────────────────

def accumulate_votes(subjects, hemi_str):
    """Count per-vertex per-category votes across controls."""
    hemi_name = 'left' if hemi_str == 'lh' else 'right'

    # votes[v, c] for c in 1-4 (face=1, house=2, object=3, word=4)
    votes = np.zeros((N_VERTICES, 5), dtype=int)
    n_subs = 0

    for sid, info in sorted(subjects.items()):
        if info['patient_status'] != 'control':
            continue
        first_ses = info['sessions'][0]

        wta_path = os.path.join(WTA_DIR, f'{sid}_ses-{first_ses}_{hemi_str}_wta.nii.gz')
        if not os.path.exists(wta_path):
            continue

        anat2stand = get_anat2stand(sid, first_ses)
        if anat2stand is None:
            continue

        mni_path = native_to_mni(wta_path, anat2stand)
        if mni_path is None:
            continue

        surf = project_to_surface(mni_path, hemi=hemi_name)
        os.unlink(mni_path)

        labels = np.round(surf).astype(int)
        labels = np.clip(labels, 0, 4)

        for v in range(N_VERTICES):
            if labels[v] > 0:
                votes[v, labels[v]] += 1

        n_subs += 1
        print(f'  {sid} ({n_subs})', end='\r')

    print(f'\n  Total controls: {n_subs}')
    return votes, n_subs


# ── Plot heatmaps ─────────────────────────────────────────────────────────────

def plot_category_heatmaps(votes, n_subs, hemi_str):
    """4-panel figure: one per category, showing vote count 0–n_subs."""
    hemi_name = 'left' if hemi_str == 'lh' else 'right'
    hemi_label = 'LH' if hemi_str == 'lh' else 'RH'
    mesh_key = f'infl_{hemi_name}'
    bg_key = f'sulc_{hemi_name}'

    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                              subplot_kw={'projection': '3d'})

    for idx, cat in enumerate(CATEGORIES):
        ax = axes[idx // 2, idx % 2]
        cat_idx = idx + 1  # face=1, house=2, object=3, word=4
        count_data = votes[:, cat_idx].astype(float)

        plotting.plot_surf_stat_map(
            fsaverage[mesh_key],
            count_data,
            bg_map=fsaverage[bg_key],
            hemi=hemi_name,
            view='ventral',
            cmap=CAT_CMAPS[cat],
            threshold=1,  # need at least 1 vote to show
            vmax=n_subs,
            colorbar=True,
            axes=ax,
        )
        ax.set_title(f'{cat.capitalize()}', fontsize=14, fontweight='bold')

    plt.suptitle(f'Controls (n={n_subs}) — {hemi_label}\n'
                 f'Vote count per category per vertex',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(OUT_DIR, f'group_ctrl_{hemi_str}_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if subprocess.run(['which', 'flirt'], capture_output=True).returncode != 0:
        print("ERROR: flirt not found. Load FSL first: module load fsl/6.0.3")
        sys.exit(1)

    subjects = load_subjects()
    print(f"Subjects: {len(subjects)}")

    for hemi_str, hemi_label in [('lh', 'LH'), ('rh', 'RH')]:
        print(f"\n{'='*60}")
        print(f"{hemi_label}")
        print(f"{'='*60}")

        votes, n_subs = accumulate_votes(subjects, hemi_str)
        plot_category_heatmaps(votes, n_subs, hemi_str)

    print(f"\nDone. Figures in: {OUT_DIR}")


if __name__ == '__main__':
    main()