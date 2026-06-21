"""
QC for sub-108 T1/T2 registration.
Outputs to /user_data/csimmon2/sym_pt/qc_sub108/
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

BASE = Path(processed_dir)
OUT  = Path('/user_data/csimmon2/sym_pt/qc_sub108')
OUT.mkdir(parents=True, exist_ok=True)

# cope numbering: 1=face, 2=house, 3=object, 4=word (differential)
COPE_MAP = {'face': 1, 'house': 2, 'object': 3, 'word': 4}

# Longitudinal patients — zstats live under first_ses gfeat,
# filename encodes which ses the zstat represents.
SUBJECTS = {
    'sub-108': {'first_ses': '01', 'sessions': ['01', '02']},
    'sub-010': {'first_ses': '02', 'sessions': ['02', '03']},
}

def zstat_path(sub, first_ses, ses, cope):
    # Each session's gfeat names its zstat using first_ses number
    return (BASE / sub / f'ses-{ses}' / 'derivatives' / 'fsl' / 'loc' /
            'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats' / f'zstat1_ses{first_ses}.nii.gz')

def zstat_path_alt(sub, first_ses, ses, cope):
    # No fallback — zstat1.nii.gz is the group-level combined stat, not per-session
    return None

def anat_path(sub, first_ses):
    return BASE / sub / f'ses-{first_ses}' / 'anat' / 'T1w_brain_stand.nii.gz'

def resolve_zstat(sub, first_ses, ses, cope):
    p = zstat_path(sub, first_ses, ses, cope)
    return p if p.exists() else None

# ───── 1. Verify paths ─────────────────────────────────────────────────────
print('── Checking paths ──')
for sub, info in SUBJECTS.items():
    fs = info['first_ses']
    ap = anat_path(sub, fs)
    print(f'  {sub} anat: {"OK" if ap.exists() else "MISSING"}  {ap}')
    for ses in info['sessions']:
        for cat, cope in COPE_MAP.items():
            zp = resolve_zstat(sub, fs, ses, cope)
            status = f'OK ({zp.name})' if zp else 'MISSING'
            print(f'    {sub} ses-{ses} {cat}: {status}')

# ───── 2. Anatomical + functional overlay per session (face anchor) ────────
print('\n── Anat+func overlays ──')
for sub, info in SUBJECTS.items():
    fs = info['first_ses']
    ap = anat_path(sub, fs)
    if not ap.exists(): continue
    for ses in info['sessions']:
        fp = resolve_zstat(sub, fs, ses, COPE_MAP['face'])
        if not fp: continue
        out = OUT / f'{sub}_anat_func_ses-{ses}.png'
        plotting.plot_stat_map(
            str(fp), bg_img=str(ap), threshold=2.3,
            display_mode='ortho',
            title=f'{sub} ses-{ses} face z>2.3',
            output_file=str(out))
        print(f'  wrote {out.name}')

# ───── 3. T1 vs T_last superimposed per category ───────────────────────────
print('\n── T1 vs T2 overlays ──')
for sub, info in SUBJECTS.items():
    fs = info['first_ses']
    ses_a, ses_b = info['sessions']
    ap = anat_path(sub, fs)
    if not ap.exists(): continue
    for cat, cope in COPE_MAP.items():
        f1 = resolve_zstat(sub, fs, ses_a, cope)
        f2 = resolve_zstat(sub, fs, ses_b, cope)
        if not f1 or not f2: continue
        disp = plotting.plot_anat(str(ap), display_mode='ortho',
                                   title=f'{sub} {cat}: T1 red, T2 blue (z>2.3)')
        disp.add_overlay(str(f1), threshold=2.3, cmap='autumn')
        disp.add_overlay(str(f2), threshold=2.3, cmap='winter')
        out = OUT / f'{sub}_T1T2_{cat}.png'
        disp.savefig(str(out))
        disp.close()
        print(f'  wrote {out.name}')

# ───── 4. Peak coord table ─────────────────────────────────────────────────
print('\n── Peak coords ──')
rows = []
for sub, info in SUBJECTS.items():
    fs = info['first_ses']
    for ses in info['sessions']:
        for cat, cope in COPE_MAP.items():
            fp = resolve_zstat(sub, fs, ses, cope)
            if not fp:
                rows.append({'sub': sub, 'ses': ses, 'cat': cat,
                             'x_mni': np.nan, 'y_mni': np.nan,
                             'z_mni': np.nan, 'peak_z': np.nan}); continue
            img = nib.load(str(fp))
            data = img.get_fdata()
            if not (data > 2.3).any():
                rows.append({'sub': sub, 'ses': ses, 'cat': cat,
                             'x_mni': np.nan, 'y_mni': np.nan,
                             'z_mni': np.nan, 'peak_z': np.nan}); continue
            ijk = np.unravel_index(np.argmax(data), data.shape)
            mni = nib.affines.apply_affine(img.affine, ijk)
            rows.append({'sub': sub, 'ses': ses, 'cat': cat,
                         'x_mni': round(mni[0], 1), 'y_mni': round(mni[1], 1),
                         'z_mni': round(mni[2], 1), 'peak_z': round(float(data[ijk]), 2)})

df = pd.DataFrame(rows)
df.to_csv(OUT / 'peak_coords.csv', index=False)
print(df.to_string(index=False))

# ───── 5. T1→T2 drift summary ──────────────────────────────────────────────
print('\n── T1→T2 peak drift (mm) ──')
for sub, info in SUBJECTS.items():
    ses_a, ses_b = info['sessions']
    for cat in COPE_MAP:
        a = df[(df['sub']==sub) & (df['ses']==ses_a) & (df['cat']==cat)]
        b = df[(df['sub']==sub) & (df['ses']==ses_b) & (df['cat']==cat)]
        if a.empty or b.empty or a.x_mni.isna().any() or b.x_mni.isna().any():
            print(f'  {sub} {cat:<7} = skipped (missing)'); continue
        d = np.sqrt((a.x_mni.values[0]-b.x_mni.values[0])**2 +
                    (a.y_mni.values[0]-b.y_mni.values[0])**2 +
                    (a.z_mni.values[0]-b.z_mni.values[0])**2)
        print(f'  {sub} {cat:<7} = {d:6.2f} mm')

print(f'\nAll output: {OUT}')
print('tar -czf qc_sub108.tar.gz -C /user_data/csimmon2/sym_pt qc_sub108')