#!/usr/bin/env python3
"""
register_1stlevel.py - Register each 1stLevel FEAT filtered_func_data to ses-01 anat
For ses-01: applies example_func2highres.mat directly
For ses-02+: concatenates example_func2highres.mat + anat2ses01.mat

Usage: python register_1stlevel.py sub-004 01
"""
import subprocess
import os
import sys
from glob import glob
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, get_sessions

sub = sys.argv[1]   # e.g., 'sub-004'
ses = sys.argv[2]   # e.g., '01'

task = 'loc'
sub_clean = sub.replace('sub-', '')

# First session's anat = reference
sessions = get_sessions(sub_clean)
first_ses = f'{sessions[0]:02d}'
ref_anat = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses}/anat/T1w_brain.nii.gz'

if not os.path.exists(ref_anat):
    print(f"ERROR: Reference anat not found: {ref_anat}")
    sys.exit(1)

# Auto-detect runs from FEAT directories
task_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses}/derivatives/fsl/{task}'
runs = []
for feat_dir in glob(f'{task_dir}/run-*/1stLevel.feat'):
    run = feat_dir.split('run-')[1].split('/')[0]
    runs.append(run)
runs = sorted(runs)

print(f"Processing sub-{sub_clean} ses-{ses}")
print(f"Reference: ses-{first_ses} anat")
print(f"Found runs: {runs}")

for run in runs:
    print(f"\n  sub-{sub_clean} {task} run-{run}")

    run_dir = f'{task_dir}/run-{run}/1stLevel.feat'
    filtered_func = f'{run_dir}/filtered_func_data.nii.gz'
    out_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
    func2highres = f'{run_dir}/reg/example_func2highres.mat'

    if not os.path.exists(filtered_func):
        print(f"    filtered_func_data.nii.gz missing")
        continue

    if not os.path.exists(func2highres):
        print(f"    example_func2highres.mat missing")
        continue

    if os.path.exists(out_func):
        print(f"    Already registered")
        continue

    # Determine transform matrix
    if ses == first_ses:
        xfm_mat = func2highres
    else:
        anat2ses01 = f'{processed_dir}/sub-{sub_clean}/ses-{ses}/anat/anat2ses{first_ses}.mat'
        if not os.path.exists(anat2ses01):
            print(f"    anat2ses{first_ses}.mat missing - run script 08 first")
            continue
        xfm_mat = f'{run_dir}/reg/func2ses{first_ses}.mat'
        if not os.path.exists(xfm_mat):
            concat_cmd = (f'convert_xfm -omat {xfm_mat} '
                          f'-concat {anat2ses01} {func2highres}')
            print(f"    Creating combined matrix: func -> ses-{first_ses}")
            subprocess.run(concat_cmd.split(), check=True)

    cmd = (f'flirt -in {filtered_func} -ref {ref_anat} -out {out_func} '
           f'-applyxfm -init {xfm_mat} -interp trilinear')
    print(f"    Running: {cmd}")

    try:
        subprocess.run(cmd.split(), check=True)
        print(f"    Done")
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e}")

print(f"\nFinished sub-{sub_clean} ses-{ses}")