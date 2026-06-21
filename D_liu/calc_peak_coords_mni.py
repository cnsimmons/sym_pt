#!/usr/bin/env python3
"""
calc_peak_coords_mni.py — Convert native peak coords to MNI.

Reads liu_exact_replication_v2.csv (peak_x/y/z_native columns), converts
each peak to MNI mm via FSL img2imgcoord using each subject's anat2stand.mat,
writes peak_coords_mni.csv.

BE SURE TO LOAD FSL BEFORE RUNNING (for img2imgcoord).
"""
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir

BASE_DIR = Path(processed_dir)
INPUT_CSV  = Path('/user_data/csimmon2/git_repos/sym_pt/liu_exact_replication_v2.csv')
OUTPUT_CSV = BASE_DIR / 'group_results' / 'peak_coords' / 'peak_coords_mni.csv'
MNI_REF    = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'


def native_mm_to_mni_mm(xyz_mm, native_brain, anat2stand):
    """Native mm → MNI mm via img2imgcoord."""
    tmp = '/tmp/peak_mm.txt'
    with open(tmp, 'w') as f:
        f.write(f'{xyz_mm[0]} {xyz_mm[1]} {xyz_mm[2]}\n')
    cmd = (f'img2imgcoord -src {native_brain} -dest {MNI_REF} '
           f'-xfm {anat2stand} -mm {tmp}')
    try:
        out = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        return [float(c) for c in out.stdout.strip().split('\n')[-1].split()[:3]]
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f'    img2imgcoord failed: {e}')
        return [np.nan, np.nan, np.nan]


def first_ses_for(sub):
    """First session string ('01', '02', ...) from filesystem."""
    sub_dir = BASE_DIR / sub
    sess = sorted(d.name for d in sub_dir.iterdir()
                  if d.is_dir() and d.name.startswith('ses-'))
    return sess[0].replace('ses-', '') if sess else None


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['peak_x_native', 'peak_y_native', 'peak_z_native']).copy()

    # Cache transforms per subject
    transforms = {}
    for sub in df['subject_id'].unique():
        fs = first_ses_for(sub)
        if fs is None:
            print(f'  {sub}: no sessions found, skipping')
            continue
        anat = BASE_DIR / sub / f'ses-{fs}' / 'anat'
        anat2stand   = anat / 'anat2stand.mat'
        native_brain = anat / 'T1w_brain.nii.gz'
        if not anat2stand.exists() or not native_brain.exists():
            print(f'  {sub}: missing anat2stand or T1w_brain, skipping')
            continue
        transforms[sub] = (str(native_brain), str(anat2stand))

    print(f'Converting {len(df)} peaks across {len(transforms)} subjects...')

    mni_x, mni_y, mni_z = [], [], []
    for i, row in df.iterrows():
        sub = row['subject_id']
        if sub not in transforms:
            mni_x.append(np.nan); mni_y.append(np.nan); mni_z.append(np.nan)
            continue
        brain, mat = transforms[sub]
        xyz = native_mm_to_mni_mm(
            [row['peak_x_native'], row['peak_y_native'], row['peak_z_native']],
            brain, mat)
        mni_x.append(xyz[0]); mni_y.append(xyz[1]); mni_z.append(xyz[2])

    df['peak_x_mni'] = mni_x
    df['peak_y_mni'] = mni_y
    df['peak_z_mni'] = mni_z

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'Saved: {OUTPUT_CSV} ({len(df)} rows)')


if __name__ == '__main__':
    main()