#!/usr/bin/env python3
"""
inflated_wta_surface.py — Project WTA category maps onto fsaverage5 inflated surface.

Requires FSL loaded (for flirt). Run with:
  module load fsl/6.0.3
  python inflated_wta_surface.py

Two-pass approach:
  Pass 1: Build group control WTA with agreement threshold → defines VOTC mask
  Pass 2: Project individual patients, masked to VOTC vertices only

Outputs:
  - Group control LH and RH ventral WTA
  - Per-patient per-session intact hemisphere ventral WTA (VOTC-masked)
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, _load_csv

from nilearn import datasets, surface, plotting

# ── Configuration ─────────────────────────────────────────────────────────────

EXCLUDE = ['sub-017']
CATEGORIES = ['face', 'house', 'object', 'word']
CAT_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# Minimum proportion of controls that must agree at a vertex to include it
MIN_AGREEMENT = 0.3

# WTA NIfTI directory
WTA_DIR = os.path.join(processed_dir, 'group_results', 'wta_maps')

# MNI reference
FSLDIR = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
MNI_REF = os.path.join(FSLDIR, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

# Output
OUT_DIR = os.path.join(processed_dir, 'group_results', 'figures', 'inflated_wta')
os.makedirs(OUT_DIR, exist_ok=True)

# fsaverage5
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
N_VERTICES = 10242  # fsaverage5

# Discrete colormap: indices 0-4 → bg, face, house, object, word
ROI_CMAP = ListedColormap(['#cccccc', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])


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
    """Transform native-space NIfTI to MNI using flirt."""
    mni_out = tempfile.NamedTemporaryFile(suffix='_mni.nii.gz', delete=False).name
    cmd = [
        'flirt',
        '-in', str(native_nifti),
        '-ref', MNI_REF,
        '-applyxfm',
        '-init', str(anat2stand_mat),
        '-out', mni_out,
        '-interp', 'nearestneighbour',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FLIRT error: {result.stderr.strip()}")
        return None
    return mni_out


def project_to_surface(mni_nifti, hemi='left'):
    """Project MNI volume onto fsaverage5 pial mesh."""
    mesh = fsaverage[f'pial_{hemi}']
    return surface.vol_to_surf(mni_nifti, mesh, interpolation='nearest')


def render_ventral(surf_data, hemi, title='', save_path=None, votc_mask=None):
    """Render ventral view using plot_surf_roi for discrete category labels.
    If votc_mask provided, zero out vertices outside VOTC."""
    mesh_key = f'infl_{hemi}'
    bg_key = f'sulc_{hemi}'

    labels = np.round(surf_data).astype(int)
    labels = np.clip(labels, 0, 4)

    # Apply VOTC mask: only show categories at vertices where controls show data
    if votc_mask is not None:
        labels[~votc_mask] = 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                            subplot_kw={'projection': '3d'})

    plotting.plot_surf_roi(
        fsaverage[mesh_key],
        roi_map=labels,
        bg_map=fsaverage[bg_key],
        hemi=hemi,
        view='ventral',
        cmap=ROI_CMAP,
        axes=ax,
    )

    legend_patches = [Patch(facecolor=c, edgecolor='black', label=cat)
                      for cat, c in zip(CATEGORIES, CAT_COLORS)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=12, frameon=True)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


def get_anat2stand(sid, first_ses):
    """Find anat2stand.mat, trying first session then ses-01."""
    for ses_try in [first_ses, '01']:
        path = os.path.join(processed_dir, sid, f'ses-{ses_try}',
                             'anat', 'anat2stand.mat')
        if os.path.exists(path):
            return path
    return None


# ── Group average WTA ─────────────────────────────────────────────────────────

def compute_group_wta(subjects, hemi_str):
    """
    For each fsaverage5 vertex, find which category wins most often
    across controls (first session). Applies MIN_AGREEMENT threshold.

    Returns:
      group_wta: (n_vertices,) int array, 0=bg, 1-4=categories
      votc_mask: (n_vertices,) bool array, True where agreement >= threshold
      n_subs: number of controls projected
    """
    hemi_name = 'left' if hemi_str == 'lh' else 'right'

    # votes[v, c] = number of controls where vertex v is category c
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
        print(f'  Control {sid} projected ({n_subs})', end='\r')

    print(f'\n  Total controls: {n_subs}')

    if n_subs == 0:
        return np.zeros(N_VERTICES, dtype=int), np.zeros(N_VERTICES, dtype=bool), 0

    # Agreement: proportion of controls voting for winning category
    total_votes = votes[:, 1:].sum(axis=1)
    max_votes = votes[:, 1:].max(axis=1)

    agreement = np.zeros(N_VERTICES)
    has_any = total_votes > 0
    agreement[has_any] = max_votes[has_any] / n_subs

    # VOTC mask: vertices where enough controls agree
    votc_mask = agreement >= MIN_AGREEMENT

    # WTA only at masked vertices
    group_wta = np.zeros(N_VERTICES, dtype=int)
    group_wta[votc_mask] = np.argmax(votes[votc_mask, 1:], axis=1) + 1

    n_votc = int(votc_mask.sum())
    n_total = int(has_any.sum())
    print(f'  VOTC vertices: {n_votc} / {n_total} with any data '
          f'(agreement >= {MIN_AGREEMENT:.0%})')

    return group_wta, votc_mask, n_subs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if subprocess.run(['which', 'flirt'], capture_output=True).returncode != 0:
        print("ERROR: flirt not found. Load FSL first: module load fsl/6.0.3")
        sys.exit(1)

    if not os.path.exists(MNI_REF):
        print(f"ERROR: MNI reference not found at {MNI_REF}")
        sys.exit(1)

    subjects = load_subjects()
    long_subs = {sid: info for sid, info in subjects.items()
                 if len(info['sessions']) >= 2}

    print(f"Subjects: {len(subjects)} total, {len(long_subs)} longitudinal")

    # ── Pass 1: Group control averages (also builds VOTC masks) ───────────
    print("\n" + "=" * 60)
    print("PASS 1: GROUP CONTROLS")
    print("=" * 60)

    votc_masks = {}  # 'left' / 'right' → bool mask

    for hemi_str, hemi_label in [('lh', 'LH'), ('rh', 'RH')]:
        print(f"\n  {hemi_label}:")
        group_wta, votc_mask, n = compute_group_wta(subjects, hemi_str)
        hemi_name = 'left' if hemi_str == 'lh' else 'right'
        votc_masks[hemi_name] = votc_mask

        render_ventral(
            group_wta.astype(float), hemi_name,
            title=f'Controls Group Average (n={n}) — {hemi_label} WTA',
            save_path=os.path.join(OUT_DIR, f'group_ctrl_{hemi_str}_wta_inflated.png'),
            votc_mask=votc_mask,
        )

    # ── Pass 2: Individual patients (masked to VOTC) ──────────────────────
    print("\n" + "=" * 60)
    print("PASS 2: INDIVIDUAL PATIENTS")
    print("=" * 60)

    for sid, info in sorted(long_subs.items()):
        if info['patient_status'] != 'OTC':
            continue

        h = info['hemi']
        surgery = info.get('surgery_side', '?')
        hemi_label = 'LH' if h == 'l' else 'RH'
        first_ses = info['sessions'][0]

        anat2stand = get_anat2stand(sid, first_ses)
        if anat2stand is None:
            print(f"SKIP {sid}: no anat2stand.mat")
            continue

        intact_side = 'left' if h == 'l' else 'right'
        mask = votc_masks.get(intact_side)

        print(f"\n{'='*60}")
        print(f"{sid} ({surgery}-resect, intact {hemi_label})")
        print(f"{'='*60}")

        for ses in info['sessions']:
            intact_hemi_str = 'lh' if h == 'l' else 'rh'
            wta_path = os.path.join(WTA_DIR,
                                     f'{sid}_ses-{ses}_{intact_hemi_str}_wta.nii.gz')

            if not os.path.exists(wta_path):
                print(f"  ses-{ses}: WTA NIfTI not found, skipping")
                continue

            print(f"  ses-{ses}: projecting intact {hemi_label}...")

            mni_path = native_to_mni(wta_path, anat2stand)
            if mni_path is None:
                continue

            surf_data = project_to_surface(mni_path, hemi=intact_side)
            os.unlink(mni_path)

            render_ventral(
                surf_data, intact_side,
                title=f'{sid} ({info["code"]}, intact {hemi_label}) — ses-{ses} WTA',
                save_path=os.path.join(OUT_DIR,
                                        f'{sid}_ses-{ses}_wta_inflated.png'),
                votc_mask=mask,
            )

        print(f"  Done: {sid}")

    print(f"\nAll done. Figures saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()