#!/usr/bin/env python3
"""Quick check: which subjects/sessions are missing HighLevel.gfeat"""
import os, sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, skip_subs, get_sessions

subjects = sorted(
    d.replace('sub-', '')
    for d in os.listdir(processed_dir)
    if d.startswith('sub-') and d.replace('sub-', '') not in skip_subs
)

missing = []
complete = []

for sub in subjects:
    sessions = get_sessions(sub)
    if not sessions:
        continue
    for ses in sessions:
        ses_str = f'{ses:02d}'
        gfeat = f'{processed_dir}/sub-{sub}/ses-{ses_str}/derivatives/fsl/loc/HighLevel.gfeat'
        cope1 = f'{gfeat}/cope1.feat/stats/zstat1.nii.gz'
        
        if not os.path.exists(gfeat):
            missing.append((sub, ses_str, 'no HighLevel.gfeat dir'))
        elif not os.path.exists(cope1):
            missing.append((sub, ses_str, 'gfeat exists but cope1/zstat1 missing'))
        else:
            complete.append((sub, ses_str))

print(f'Complete: {len(complete)}')
print(f'Missing:  {len(missing)}')
print()
if missing:
    print('MISSING:')
    for sub, ses, reason in missing:
        print(f'  sub-{sub} ses-{ses}: {reason}')