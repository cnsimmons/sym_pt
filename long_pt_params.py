"""
long_pt_params.py - Centralized configuration for longitudinal patient analysis
Following Ayzenberg lab conventions
"""
import numpy as np

# =============================================================================
# DIRECTORIES
# =============================================================================
raw_dir = '/lab_data/behrmannlab/hemi/Raw'
processed_dir = '/user_data/csimmon2/long_pt'
git_dir = '/user_data/csimmon2/git_repos/long_pt'
csv_file = f'{git_dir}/long_pt_sub_info.csv'

mni_brain = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
mni_2mm = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm.nii.gz'

# =============================================================================
# TASK PARAMETERS
# =============================================================================
task = 'loc'
tr = 2.0
block_duration = 16.0

# =============================================================================
# CONDITIONS
# =============================================================================
conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']
categories = ['face', 'house', 'object', 'word']  # RSA categories (no scramble)
n_conditions = len(conditions)

# =============================================================================
# COPE MAPS
# =============================================================================
# EVs: 1=Face, 2=House, 3=Object, 4=Word, 5=Scramble

# Identity contrasts for RSA (per Kriegeskorte 2008, Liu 2025)
cope_identity = {
    'face': 15, 'house': 16, 'object': 17, 'word': 18, 'scramble': 19
}

# Category > All Others for ROI definition (Liu 2025)
cope_selective = {
    'face': 6, 'house': 7, 'object': 8, 'word': 9
}

# Category > Scramble (alternative ROI definition)
cope_scramble = {
    'face': 10, 'house': 11, 'object': 3, 'word': 12
}

# =============================================================================
# RSA PARAMETERS (Liu 2025)
# =============================================================================
roi_threshold = 90  # percentile (top 10%)
roi_threshold_type = 'percentile'
sphere_radius = 6  # mm
dissimilarity = 'correlation'

# =============================================================================
# SUBJECTS
# =============================================================================
skip_subs = ['004', '007', '021', '108']

session_start = {'010': 2, '018': 2, '068': 2}

# =============================================================================
# MOTION
# =============================================================================
fd_threshold = 0.5
spike_pct_exclude = 20

# =============================================================================
# CONTRAST WEIGHTS (for FSF)
# =============================================================================
contrast_weights = {
    1:  [1, 0, -1, 0, 0],           # Face > Object
    2:  [0, 1, -1, 0, 0],           # House > Object
    3:  [0, 0, 1, 0, -1],           # Object > Scramble
    4:  [0, 0, -1, 1, 0],           # Word > Object
    5:  [-0.25, -0.25, -0.25, -0.25, 1],
    6:  [4, -1, -1, -1, -1],        # Face > all
    7:  [-1, 4, -1, -1, -1],        # House > all
    8:  [-1, -1, 4, -1, -1],        # Object > all
    9:  [-1, -1, -1, 4, -1],        # Word > all
    10: [1, 0, 0, 0, -1],           # Face > Scramble
    11: [0, 1, 0, 0, -1],           # House > Scramble
    12: [0, 0, 0, 1, -1],           # Word > Scramble
    13: [1, 0, 0, -1, 0],           # Face > Word
    14: [0, -1, 1, 0, 0],           # Object > House
    # Identity contrasts for RSA
    15: [1, 0, 0, 0, 0],            # Face_raw
    16: [0, 1, 0, 0, 0],            # House_raw
    17: [0, 0, 1, 0, 0],            # Object_raw
    18: [0, 0, 0, 1, 0],            # Word_raw
    19: [0, 0, 0, 0, 1],            # Scramble_raw
}

contrast_names = {
    1: 'Face', 2: 'House', 3: 'Object', 4: 'Word', 5: 'Scramble',
    6: 'Face-all', 7: 'House-all', 8: 'Object-all', 9: 'Word-all',
    10: 'Face-scramble', 11: 'House-scramble', 12: 'Word-scramble',
    13: 'Face-Word', 14: 'Object-House',
    15: 'Face_raw', 16: 'House_raw', 17: 'Object_raw', 
    18: 'Word_raw', 19: 'Scramble_raw'
}

# =============================================================================
# HELPERS
# =============================================================================
def get_sessions(sub, df=None):
    """Get session numbers for subject"""
    import pandas as pd
    if df is None:
        df = pd.read_csv(csv_file)
    
    sub_clean = sub.replace('sub-', '')
    row = df[df['sub'].str.contains(sub_clean)]
    if row.empty:
        return []
    
    row = row.iloc[0]
    n = sum(1 for c in ['age_1','age_2','age_3','age_4','age_5'] 
            if pd.notna(row[c]) and str(row[c]).strip())
    start = session_start.get(sub_clean, 1)
    return list(range(start, start + n))


def get_runs(sub, ses):
    """Get run numbers for subject/session"""
    import glob
    sub_clean = sub.replace('sub-', '')
    func = f'{raw_dir}/sub-{sub_clean}/ses-{ses:02d}/func'
    files = glob.glob(f'{func}/*task-{task}_run-*_bold.nii.gz')
    return sorted([int(f.split('run-')[1].split('_')[0]) for f in files])


def get_feat_dir(sub, ses, run):
    """Get FEAT directory path"""
    sub_clean = sub.replace('sub-', '')
    return f'{processed_dir}/sub-{sub_clean}/ses-{ses:02d}/derivatives/fsl/{task}/run-{run:02d}/1stLevel.feat'


def get_cope(sub, ses, run, cope, space='standard'):
    """Get cope file path"""
    feat = get_feat_dir(sub, ses, run)
    subdir = 'reg_standard/stats' if space == 'standard' else 'stats'
    return f'{feat}/{subdir}/cope{cope}.nii.gz'
