"""
sym_pt_params.py - Configuration for clean anatomical & ROI pipeline
"""
import os
import glob
import pandas as pd

# =============================================================================
# 1. DIRECTORIES
# =============================================================================
# Where the raw data lives (Standard Lab Path)
raw_dir = '/lab_data/behrmannlab/hemi/Raw'

# Where the PROCESSED DATA goes (The Scratch Folder)
processed_dir = '/user_data/csimmon2/sym_pt'

# Where the SCRIPTS live (The Git Repo)
git_dir = '/user_data/csimmon2/git_repos/sym_pt'

# The Subject Info CSV (Must be inside the git_dir)
csv_file = f'{git_dir}/sub_info.csv'

# ROI Directory (Inside processed_dir)
roi_dir = f'{processed_dir}/rois'

# =============================================================================
# 2. EXPERIMENT PARAMETERS
# =============================================================================
task = 'loc'
conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']
fd_threshold = 0.5  # Motion threshold (mm) for scrubbing

# =============================================================================
# 3. ANATOMY / TEMPLATES
# =============================================================================
mni_brain = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
mni_2mm = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm.nii.gz'

# =============================================================================
# 4. SUBJECT CONFIGURATION
# =============================================================================
# Subjects to completely ignore
skip_subs = ['004', '007', '021', '108']

# Session offsets (if session 1 is missing/bad, start at 2)
session_start = {'010': 2, '018': 2, '068': 2}

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
def get_sessions(sub, df=None):
    """
    Get session numbers for a subject based on the CSV info.
    Checks 'age_1' through 'age_5' to count valid sessions.
    """
    if df is None:
        if not os.path.exists(csv_file):
            print(f"ERROR: CSV file not found at {csv_file}")
            return []
        df = pd.read_csv(csv_file)
    
    sub_clean = sub.replace('sub-', '')
    
    # Filter for the specific subject
    row = df[df['sub'].astype(str).str.contains(sub_clean)]
    if row.empty:
        print(f"Warning: Subject {sub} not found in CSV.")
        return []
    
    row = row.iloc[0]
    
    # Count sessions based on filled 'age' columns
    # We check if the column exists and if the value is not empty/NaN
    valid_sessions = 0
    for i in range(1, 6):
        col = f'age_{i}'
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
            valid_sessions += 1
    
    # Determine start session
    start = session_start.get(sub_clean, 1)
    
    # Return list of sessions (e.g., [1, 2, 3] or [2, 3])
    return list(range(start, start + valid_sessions))


def get_runs(sub, ses):
    """
    Get run numbers for subject/session by looking at the Raw directory.
    Returns a sorted list of integers (e.g., [1, 2, 3]).
    """
    sub_clean = sub.replace('sub-', '')
    func_dir = f'{raw_dir}/sub-{sub_clean}/ses-{ses:02d}/func'
    
    # Find all loc task runs
    files = glob.glob(f'{func_dir}/*task-{task}_run-*_bold.nii.gz')
    
    runs = []
    for f in files:
        try:
            # Assumes BIDS format: ...run-01_bold.nii.gz
            # Split by 'run-' and take the next part
            run_str = f.split('run-')[1].split('_')[0]
            runs.append(int(run_str))
        except (IndexError, ValueError):
            continue
            
    return sorted(runs)
