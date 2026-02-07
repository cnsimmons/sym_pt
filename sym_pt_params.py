"""
sym_pt_params.py - Configuration for sym_pt pipeline
Uses unified long-format CSV: one row per subject-session.
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

# ROI paths
roi_source_lib = '/user_data/csimmon2/git_repos/long_pt/roiParcels'
roi_dir = f'{processed_dir}/rois'

# =============================================================================
# 2. EXPERIMENT PARAMETERS
# =============================================================================
task = 'loc'
conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']

# =============================================================================
# 3. ANATOMY / TEMPLATES
# =============================================================================
mni_brain = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'

# =============================================================================
# 4. SUBJECT CONFIGURATION
# =============================================================================
skip_subs = ['108']

# Note: session_start offsets (sub-010, sub-018, sub-068) are no longer needed.
# The unified CSV has explicit session numbers per row.

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
_df_cache = None

def _load_csv():
    """Load and cache the unified subject info CSV."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV not found at {csv_file}")
        return pd.DataFrame()
    _df_cache = pd.read_csv(csv_file)
    # Normalize sub column (strip 'sub-' prefix for matching)
    _df_cache['sub_clean'] = _df_cache['sub'].str.replace('sub-', '', regex=False)
    # Extract session number from 'ses-XX'
    _df_cache['ses_num'] = _df_cache['ses'].str.replace('ses-', '', regex=False).astype(int)
    return _df_cache


def is_patient(sub):
    """Check if subject is a patient based on group column."""
    sub_clean = sub.replace('sub-', '')
    df = _load_csv()
    if df.empty:
        return False
    row = df[df['sub_clean'] == sub_clean]
    if row.empty:
        return False
    return row.iloc[0]['group'] == 'patient'


def get_sessions(sub, df=None):
    """Get list of session numbers for a subject."""
    if df is None:
        df = _load_csv()
    if df.empty:
        return []

    sub_clean = sub.replace('sub-', '')
    rows = df[df['sub_clean'] == sub_clean]

    return sorted(rows['ses_num'].tolist())


def get_runs(sub, ses):
    """Get run numbers for subject/session from Raw directory."""
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


def get_sub_info(sub, ses=None):
    """Get full info dict for a subject (optionally at a specific session)."""
    sub_clean = sub.replace('sub-', '')
    df = _load_csv()
    if df.empty:
        return {}

    rows = df[df['sub_clean'] == sub_clean]
    if ses is not None:
        rows = rows[rows['ses_num'] == ses]

    if rows.empty:
        return {}

    row = rows.iloc[0]
    return {
        'sub': row['sub'],
        'group': row['group'],
        'sex': row['sex'],
        'surgery_side': row.get('surgery_side', ''),
        'intact_hemi': row.get('intact_hemi', ''),
        'code': row.get('code', ''),
        'age': row.get('age', None),
    }