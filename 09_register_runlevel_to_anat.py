#!/usr/bin/env python3
"""
09_register_zstats.py - Register run-level zstat and cope files to ses-01 anat
Applies example_func2standard.mat (func -> ses-01 anat)

Usage: python 09_register_zstats.py sub-004 01
"""
import subprocess
import os
import sys
from glob import glob
from sym_pt_params import processed_dir, get_sessions

sub = sys.argv[1]   # e.g., 'sub-004'
ses = sys.argv[2]   # e.g., '01'

task = 'loc'
sub_clean = sub.replace('sub-', '')
zstats = list(range(1, 20))  # 1-19

# First session's anat = reference
sessions = get_sessions(sub_clean)
first_ses = f'{sessions[0]:02d}'
ref_anat = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses}/anat/T1w_brain.nii.gz'

if not os.path.exists(ref_anat):
    print(f"ERROR: Reference anat not found: {ref_anat}")
    sys.exit(1)

# Auto-detect runs
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
    print(f"\n  Run {run}:")
    run_dir = f'{task_dir}/run-{run}/1stLevel.feat'
    xfm_mat = f'{run_dir}/reg/example_func2standard.mat'
    reg_stats_dir = f'{run_dir}/reg_standard/stats'
    os.makedirs(reg_stats_dir, exist_ok=True)

    if not os.path.exists(xfm_mat):
        print(f"    SKIP: example_func2standard.mat missing")
        continue

    for z in zstats:
        for prefix in ['zstat', 'cope']:
            src = f'{run_dir}/stats/{prefix}{z}.nii.gz'
            dst = f'{reg_stats_dir}/{prefix}{z}.nii.gz'

            if not os.path.exists(src):
                continue
            if os.path.exists(dst):
                continue

            cmd = (f'flirt -in {src} -ref {ref_anat} -out {dst} '
                   f'-applyxfm -init {xfm_mat} -interp trilinear')
            try:
                subprocess.run(cmd.split(), check=True)
                print(f"    {prefix}{z} done")
            except subprocess.CalledProcessError as e:
                print(f"    {prefix}{z} ERROR: {e}")

print(f"\nFinished sub-{sub_clean} ses-{ses}")