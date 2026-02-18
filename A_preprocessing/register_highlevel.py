#!/usr/bin/env python3
"""
register_highlevel.py - Create ses-01 labeled copies of HighLevel outputs

Since FEAT used the ses-01 anatomy as the highres reference for all sessions,
HighLevel outputs are ALREADY in ses-01 space. This script simply creates
symlinks with the _ses{first} naming convention that downstream scripts expect.

For first session: symlinks zstat1.nii.gz -> zstat1_ses01.nii.gz
For later sessions: same symlinks (no transform needed)

Usage: python register_highlevel.py sub-004 01
       python register_highlevel.py sub-004 05
"""
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

highlevel_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses}/derivatives/fsl/{task}/HighLevel.gfeat'

if not os.path.exists(highlevel_dir):
    print(f"ERROR: HighLevel.gfeat not found for sub-{sub_clean} ses-{ses}")
    sys.exit(1)

print(f"Processing sub-{sub_clean} ses-{ses}")
print(f"Reference: ses-{first_ses}")
print(f"Mode: symlink (HighLevel already in ses-{first_ses} space)")

for cope_num in copes:
    cope_dir = f'{highlevel_dir}/cope{cope_num}.feat'

    for prefix in ['zstat', 'cope']:
        src = f'{cope_dir}/stats/{prefix}1.nii.gz'
        dst = f'{cope_dir}/stats/{prefix}1_ses{first_ses}.nii.gz'

        if not os.path.exists(src):
            continue
        if os.path.exists(dst):
            continue

        os.symlink(os.path.abspath(src), dst)
        print(f"  cope{cope_num} {prefix} linked")

print(f"\nFinished sub-{sub_clean} ses-{ses}")