#!/usr/bin/env python3
"""
07_register_1stlevel.py - Register each 1stLevel FEAT output to anat
Applies example_func2standard.mat (func -> ses-01 anat) to filtered_func_data

Usage: python 07_register_1stlevel.py sub-004 01
"""
import subprocess
import os
import sys
from glob import glob
from sym_pt_params import processed_dir, get_sessions

# Get command line arguments
sub = sys.argv[1]   # e.g., 'sub-004'
ses = sys.argv[2]   # e.g., '01'

task = 'loc'
sub_clean = sub.replace('sub-', '')

# First session's anat = reference (FEAT's "standard")
sessions = get_sessions(sub_clean)
first_ses = f'{sessions[0]:02d}'
anat = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses}/anat/T1w_brain.nii.gz'

# Auto-detect runs from FEAT directories
sub_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses}'
task_dir = f'{sub_dir}/derivatives/fsl/{task}'

runs = []
for feat_dir in glob(f'{task_dir}/run-*/1stLevel.feat'):
    run = feat_dir.split('run-')[1].split('/')[0]
    runs.append(run)
runs = sorted(runs)

print(f"Processing sub-{sub_clean} ses-{ses}")
print(f"Anat reference: ses-{first_ses}")
print(f"Found runs: {runs}")

for run in runs:
    print(f"  sub-{sub_clean} {task} run-{run}")

    run_dir = f'{task_dir}/run-{run}/1stLevel.feat'
    filtered_func = f'{run_dir}/filtered_func_data.nii.gz'
    out_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
    xfm_mat = f'{run_dir}/reg/example_func2standard.mat'

    if not os.path.exists(filtered_func):
        print(f"    filtered_func_data.nii.gz missing")
        continue

    if not os.path.exists(xfm_mat):
        print(f"    example_func2standard.mat missing")
        continue

    if os.path.exists(out_func):
        print(f"    Already registered")
        continue

    cmd = (f'flirt -in {filtered_func} -ref {anat} -out {out_func} '
           f'-applyxfm -init {xfm_mat} -interp trilinear')
    print(f"    Running: {cmd}")

    try:
        subprocess.run(cmd.split(), check=True)
        print(f"    Done")
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e}")

print(f"Finished sub-{sub_clean} ses-{ses}")