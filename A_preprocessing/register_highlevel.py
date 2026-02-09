#!/usr/bin/env python3
"""
register_highlevel.py - Register HighLevel outputs to ses-01 space
For first session: creates symlinks (already in correct space)
For later sessions: applies anat2ses01.mat

Usage: python register_highlevel.py sub-004 01
"""
import subprocess
import os
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, get_sessions

sub = sys.argv[1]
ses = sys.argv[2]

task = 'loc'
sub_clean = sub.replace('sub-', '')
copes = list(range(1, 20))  # 1-19

# First session
sessions = get_sessions(sub_clean)
first_ses = f'{sessions[0]:02d}'
ref_anat = f'{processed_dir}/sub-{sub_clean}/ses-{first_ses}/anat/T1w_brain.nii.gz'

if not os.path.exists(ref_anat):
    print(f"ERROR: Reference anat not found: {ref_anat}")
    sys.exit(1)

# Determine if registration needed
need_reg = (ses != first_ses)
anat_xfm = f'{processed_dir}/sub-{sub_clean}/ses-{ses}/anat/anat2ses{first_ses}.mat'

if need_reg and not os.path.exists(anat_xfm):
    print(f"ERROR: anat2ses{first_ses}.mat not found for ses-{ses}")
    print("Run 08_register_anat_to_ses01.sh first")
    sys.exit(1)

highlevel_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses}/derivatives/fsl/{task}/HighLevel.gfeat'

if not os.path.exists(highlevel_dir):
    print(f"ERROR: HighLevel.gfeat not found for sub-{sub_clean} ses-{ses}")
    sys.exit(1)

print(f"Processing sub-{sub_clean} ses-{ses}")
print(f"Reference: ses-{first_ses}")
print(f"Registration needed: {need_reg}")

for cope_num in copes:
    cope_dir = f'{highlevel_dir}/cope{cope_num}.feat'

    for prefix in ['zstat', 'cope']:
        src = f'{cope_dir}/stats/{prefix}1.nii.gz'
        dst = f'{cope_dir}/stats/{prefix}1_ses{first_ses}.nii.gz'

        if not os.path.exists(src):
            continue
        if os.path.exists(dst):
            continue

        if need_reg:
            cmd = (f'flirt -in {src} -ref {ref_anat} -out {dst} '
                   f'-applyxfm -init {anat_xfm} -interp trilinear')
            try:
                subprocess.run(cmd.split(), check=True)
                print(f"  cope{cope_num} {prefix} registered")
            except subprocess.CalledProcessError as e:
                print(f"  cope{cope_num} {prefix} ERROR: {e}")
        else:
            os.symlink(os.path.abspath(src), dst)
            print(f"  cope{cope_num} {prefix} linked")

print(f"\nFinished sub-{sub_clean} ses-{ses}")