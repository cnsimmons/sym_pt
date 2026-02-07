#!/usr/bin/env python3
"""
long_pt_params.py - Centralized configuration for longitudinal patient analysis

Usage:
    from long_pt_params import PARAMS
    # or
    from long_pt_params import *
"""

from pathlib import Path

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================
PATHS = {
    'raw': Path('/lab_data/behrmannlab/hemi/Raw'),
    'processed': Path('/user_data/csimmon2/long_pt'),
    'git_repo': Path('/user_data/csimmon2/git_repos/long_pt'),
    'mni_brain': Path('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'),
    'mni_2mm': Path('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm.nii.gz'),
}
PATHS['csv'] = PATHS['git_repo'] / 'long_pt_sub_info.csv'

# =============================================================================
# TASK CONFIGURATION
# =============================================================================
TASK = 'loc'
TR = 2.0  # seconds
BLOCK_DURATION = 16.0  # seconds

# =============================================================================
# CONDITIONS & CONTRASTS
# =============================================================================
CONDITIONS = ['Face', 'House', 'Object', 'Word', 'Scramble']

# Identity contrasts for RSA (raw condition estimates vs implicit baseline)
# These are what you need for proper RSA
COPE_MAP_IDENTITY = {
    'face': 15,
    'house': 16,
    'object': 17,
    'word': 18,
    'scramble': 19
}

# Category > Scramble contrasts for ROI definition
COPE_MAP_ROI = {
    'face': 10,    # Face > Scramble
    'house': 11,   # House > Scramble
    'object': 3,   # Object > Scramble
    'word': 12     # Word > Scramble
}

# Category > All Others contrasts (Liu 2018 style ROI definition)
COPE_MAP_SELECTIVE = {
    'face': 6,     # Face > mean(others)
    'house': 7,    # House > mean(others)
    'object': 8,   # Object > mean(others)
    'word': 9      # Word > mean(others)
}

# =============================================================================
# RSA CONFIGURATION
# =============================================================================
RSA_CONFIG = {
    'cope_map': COPE_MAP_IDENTITY,  # Use raw condition estimates
    'roi_cope_map': COPE_MAP_SELECTIVE,  # Liu 2018 style for ROI definition
    'roi_threshold_percentile': 90,  # Top 10% of voxels
    'roi_threshold_type': 'percentile',  # 'percentile' or 'zscore'
    'sphere_radius_mm': 6,
    'dissimilarity_metric': 'correlation',  # 1 - Pearson r
    'categories': ['face', 'house', 'object', 'word'],  # Exclude scramble for RDM
}

# =============================================================================
# SUBJECT CONFIGURATION
# =============================================================================
# Subjects to exclude from processing
EXCLUDE_SUBS = ['sub-004', 'sub-007', 'sub-021', 'sub-108']

# Subjects with non-standard session numbering
SESSION_START = {
    'sub-010': 2,
    'sub-018': 2,
    'sub-068': 2
}

# =============================================================================
# FSL FIRST-LEVEL CONTRAST DEFINITIONS
# =============================================================================
# For reference when creating/updating FSF files
# EVs: 1=Face, 2=House, 3=Object, 4=Word, 5=Scramble

CONTRAST_DEFINITIONS = {
    # Existing differential contrasts (keep for backward compatibility)
    1:  {'name': 'Face',           'weights': [1, 0, -1, 0, 0]},      # Face > Object
    2:  {'name': 'House',          'weights': [0, 1, -1, 0, 0]},      # House > Object
    3:  {'name': 'Object',         'weights': [0, 0, 1, 0, -1]},      # Object > Scramble
    4:  {'name': 'Word',           'weights': [0, 0, -1, 1, 0]},      # Word > Object
    5:  {'name': 'Scramble',       'weights': [-0.25, -0.25, -0.25, -0.25, 1]},
    6:  {'name': 'Face-all',       'weights': [4, -1, -1, -1, -1]},   # Face > others
    7:  {'name': 'House-all',      'weights': [-1, 4, -1, -1, -1]},   # House > others
    8:  {'name': 'Object-all',     'weights': [-1, -1, 4, -1, -1]},   # Object > others
    9:  {'name': 'Word-all',       'weights': [-1, -1, -1, 4, -1]},   # Word > others
    10: {'name': 'Face-scramble',  'weights': [1, 0, 0, 0, -1]},      # Face > Scramble
    11: {'name': 'House-scramble', 'weights': [0, 1, 0, 0, -1]},      # House > Scramble
    12: {'name': 'Word-scramble',  'weights': [0, 0, 0, 1, -1]},      # Word > Scramble
    13: {'name': 'Face-Word',      'weights': [1, 0, 0, -1, 0]},      # Face > Word
    14: {'name': 'Object-House',   'weights': [0, -1, 1, 0, 0]},      # Object > House
    
    # NEW: Identity contrasts for RSA (raw condition estimates)
    15: {'name': 'Face_raw',       'weights': [1, 0, 0, 0, 0]},
    16: {'name': 'House_raw',      'weights': [0, 1, 0, 0, 0]},
    17: {'name': 'Object_raw',     'weights': [0, 0, 1, 0, 0]},
    18: {'name': 'Word_raw',       'weights': [0, 0, 0, 1, 0]},
    19: {'name': 'Scramble_raw',   'weights': [0, 0, 0, 0, 1]},
}

# =============================================================================
# MOTION/CONFOUND THRESHOLDS
# =============================================================================
MOTION_CONFIG = {
    'fd_threshold': 0.5,  # mm, for spike detection
    'dvars_threshold': None,  # not currently used
    'exclude_if_spikes_pct': 20,  # exclude run if >20% volumes are spikes
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_subject_sessions(subject_id: str, csv_df=None) -> list:
    """Get session numbers for a subject based on CSV data"""
    import pandas as pd
    
    if csv_df is None:
        csv_df = pd.read_csv(PATHS['csv'])
    
    # Normalize subject_id format
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'
    
    row = csv_df[csv_df['sub'] == subject_id]
    if row.empty:
        return []
    
    row = row.iloc[0]
    
    # Count non-empty age columns
    age_cols = ['age_1', 'age_2', 'age_3', 'age_4', 'age_5']
    session_count = sum(1 for col in age_cols 
                        if pd.notna(row[col]) and str(row[col]).strip() != '')
    
    # Get starting session
    start_ses = SESSION_START.get(subject_id, 1)
    
    return list(range(start_ses, start_ses + session_count))


def get_subject_info(subject_id: str, csv_df=None) -> dict:
    """Get all info for a subject from CSV"""
    import pandas as pd
    
    if csv_df is None:
        csv_df = pd.read_csv(PATHS['csv'])
    
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'
    
    row = csv_df[csv_df['sub'] == subject_id]
    if row.empty:
        return None
    
    row = row.iloc[0]
    
    return {
        'subject_id': subject_id,
        'is_patient': row['patient'] == 1,
        'intact_hemi': row['intact_hemi'],
        'sessions': get_subject_sessions(subject_id, csv_df),
        'dob': row.get('dob', None),
    }


def get_runs_for_session(subject_id: str, session: int) -> list:
    """Auto-detect runs from filesystem"""
    import glob
    
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'
    
    func_dir = PATHS['raw'] / subject_id / f'ses-{session:02d}' / 'func'
    
    if not func_dir.exists():
        return []
    
    bold_files = glob.glob(
        str(func_dir / f'{subject_id}_ses-{session:02d}_task-{TASK}_run-*_bold.nii.gz')
    )
    
    runs = []
    for f in bold_files:
        run_str = f.split('run-')[1].split('_')[0]
        runs.append(int(run_str))
    
    return sorted(runs)


def get_feat_dir(subject_id: str, session: int, run: int) -> Path:
    """Get path to FEAT output directory"""
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'
    
    return (PATHS['processed'] / subject_id / f'ses-{session:02d}' / 
            'derivatives' / 'fsl' / TASK / f'run-{run:02d}' / '1stLevel.feat')


def get_cope_path(subject_id: str, session: int, run: int, cope_num: int, 
                  space: str = 'native') -> Path:
    """Get path to a specific cope file
    
    Args:
        space: 'native' for run space, 'standard' for MNI space
    """
    feat_dir = get_feat_dir(subject_id, session, run)
    
    if space == 'standard':
        return feat_dir / 'reg_standard' / 'stats' / f'cope{cope_num}.nii.gz'
    else:
        return feat_dir / 'stats' / f'cope{cope_num}.nii.gz'


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================
PARAMS = {
    'paths': PATHS,
    'task': TASK,
    'tr': TR,
    'conditions': CONDITIONS,
    'cope_identity': COPE_MAP_IDENTITY,
    'cope_roi': COPE_MAP_ROI,
    'cope_selective': COPE_MAP_SELECTIVE,
    'contrasts': CONTRAST_DEFINITIONS,
    'rsa': RSA_CONFIG,
    'motion': MOTION_CONFIG,
    'exclude_subs': EXCLUDE_SUBS,
    'session_start': SESSION_START,
}

if __name__ == '__main__':
    # Print configuration summary
    print("long_pt_params.py - Configuration Summary")
    print("=" * 50)
    print(f"\nPaths:")
    for name, path in PATHS.items():
        print(f"  {name}: {path}")
    
    print(f"\nTask: {TASK}")
    print(f"Conditions: {CONDITIONS}")
    
    print(f"\nIdentity contrasts for RSA:")
    for cat, cope in COPE_MAP_IDENTITY.items():
        print(f"  {cat}: cope{cope}")
    
    print(f"\nROI definition contrasts (Category > Scramble):")
    for cat, cope in COPE_MAP_ROI.items():
        print(f"  {cat}: cope{cope}")
    
    print(f"\nExcluded subjects: {EXCLUDE_SUBS}")
    print(f"Special session starts: {SESSION_START}")
