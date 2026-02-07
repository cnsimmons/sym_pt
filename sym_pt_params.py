"""
sym_pt_params.py - Configuration for clean anatomical & ROI pipeline
"""
import os
import glob
import pandas as pd

# =============================================================================
# DIRECTORIES
# =============================================================================
# 1. RAW DATA (Lab Standard)
raw_dir = '/lab_data/behrmannlab/hemi/Raw'

# 2. PROCESSED DATA (Your Scratch Folder - NOT Git)
processed_dir = '/user_data/csimmon2/sym_pt'

# 3. CODE REPOSITORY (Where this file lives)
git_dir = '/user_data/csimmon2/git_repos/sym_pt'

# 4. CSV File (Subject Info)
csv_file = f'{git_dir}/sub_info.csv'

# 5. ROI Directory (Inside Processed)
roi_dir = f'{processed_dir}/rois'

# =============================================================================
# ANATOMY / TEMPLATES
# =============================================================================
mni_brain = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
mni_2mm = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm.nii.gz'

# =============================================================================
# SUBJECT CONFIGURATION
# =============================================================================
skip_subs = ['004', '007', '021', '108']

# Offsets for specific subjects (if session 1 is missing/bad)
session_start = {'010': 2, '018': 2, '068': 2}

# =============================================================================
# TASK PARAMETERS
# =============================================================================
task = 'loc'
conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']

# =============================================================================
# MOTION PARAMETERS
# =============================================================================
fd_threshold = 0.5  # Threshold for spike detection (framewise displacement)

# =============================================================================
# HELPER FUNCTIONS (Needed for scripts to run)
# =============================================================================
def get_sessions(sub, df=None):
    """Get session numbers for subject based on CSV info"""
    if df is None:
        df = pd.read_csv(csv_file)
    
    sub_clean = sub.replace('sub-', '')
    
    # Filter for subject
    row = df[df['sub'].str.contains(sub_clean)]
    if row.empty:
        return []
    
    row = row.iloc[0]
    
    # Count sessions based on filled 'age' columns
    n_sessions = sum(1 for c in ['age_1','age_2','age_3','age_4','age_5'] 
            if c in row and pd.notna(row[c]) and str(row[c]).strip())
    
    # Determine start session
    start = session_start.get(sub_clean, 1)
    return list(range(start, start + n_sessions))


def get_runs(sub, ses):
    """Get run numbers for subject/session from Raw dir"""
    sub_clean = sub.replace('sub-', '')
    func_dir = f'{raw_dir}/sub-{sub_clean}/ses-{ses:02d}/func'
    
    # Find all loc task runs
    files = glob.glob(f'{func_dir}/*task-loc_run-*_bold.nii.gz')
    
    # Extract run numbers
    runs = []
    for f in files:
        try:
            # Assumes format: ...run-01_bold.nii.gz
            run_str = f.split('run-')[1].split('_')[0]
            runs.append(int(run_str))
        except IndexError:
            continue
            
    return sorted(runs)
