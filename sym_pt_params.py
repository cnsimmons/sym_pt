"""
sym_pt_params.py - Configuration for clean anatomical & ROI pipeline
"""
import os
import glob
import pandas as pd

# =============================================================================
# 1. DIRECTORIES
# =============================================================================
raw_dir = '/lab_data/behrmannlab/hemi/Raw'
processed_dir = '/user_data/csimmon2/sym_pt'
git_dir = '/user_data/csimmon2/git_repos/sym_pt'
csv_file = f'{git_dir}/sub_info.csv'
roi_dir = f'{processed_dir}/rois'

# =============================================================================
# 2. EXPERIMENT PARAMETERS
# =============================================================================
task = 'loc'
conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']
# fd_threshold REMOVED to match Ayzenberg pipeline

# =============================================================================
# 3. ANATOMY / TEMPLATES
# =============================================================================
mni_brain = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
mni_2mm = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm.nii.gz'

# =============================================================================
# 4. SUBJECT CONFIGURATION
# =============================================================================
skip_subs = ['004', '007', '021', '108']
session_start = {'010': 2, '018': 2, '068': 2}

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
def get_sessions(sub, df=None):
    if df is None:
        if not os.path.exists(csv_file):
            print(f"ERROR: CSV file not found at {csv_file}")
            return []
        df = pd.read_csv(csv_file)
    
    sub_clean = sub.replace('sub-', '')
    row = df[df['sub'].astype(str).str.contains(sub_clean)]
    if row.empty:
        return []
    
    row = row.iloc[0]
    valid_sessions = 0
    for i in range(1, 6):
        col = f'age_{i}'
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
            valid_sessions += 1
            
    start = session_start.get(sub_clean, 1)
    return list(range(start, start + valid_sessions))

def get_runs(sub, ses):
    sub_clean = sub.replace('sub-', '')
    func_dir = f'{raw_dir}/sub-{sub_clean}/ses-{ses:02d}/func'
    files = glob.glob(f'{func_dir}/*task-{task}_run-*_bold.nii.gz')
    runs = []
    for f in files:
        try:
            run_str = f.split('run-')[1].split('_')[0]
            runs.append(int(run_str))
        except (IndexError, ValueError):
            continue
    return sorted(runs)
