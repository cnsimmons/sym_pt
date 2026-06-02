"""
sym_pt_params.py - Configuration for sym_pt pipeline
Uses unified long-format CSV: one row per subject-session.
## NEW AS OF 06/02/26
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
# Skipped by clean sub id (no 'sub-' prefix):
#   017 - polymicrogyria
skip_subs = ['017']

# Skipped by code (group + clean id), e.g. control083:
skip_codes = {'control083', 'control085'}

# Pre-surgical sessions to exclude from post-surgery analyses.
# Keyed by full subject id; values are session NUMBERS (ints), not 'ses-XX'.
# The first POST session is the anatomical/registration anchor.
pre_surgery_sessions = {
    'sub-021': {1}, 'sub-045': {1}, 'sub-047': {1}, 'sub-049': {1},
    'sub-070': {1}, 'sub-073': {1}, 'sub-081': {1}, 'sub-086': {1},
    'sub-108': {2},
}

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
    sub_clean = sub.replace('sub-', '')
    df = _load_csv()
    if df.empty:
        return False
    row = df[df['sub_clean'] == sub_clean]
    if row.empty:
        return False
    return row.iloc[0]['group'] in ('OTC', 'nonOTC')


def should_skip(sub):
    """True if subject is excluded by clean id or by code (group+id)."""
    sub_clean = sub.replace('sub-', '')
    if sub_clean in skip_subs:
        return True
    info = get_sub_info(sub)
    code = f"{info.get('group', '')}{sub_clean}"
    return code in skip_codes


def get_sessions(sub, df=None):
    """All session numbers for a subject (pre + post)."""
    if df is None:
        df = _load_csv()
    if df.empty:
        return []
    sub_clean = sub.replace('sub-', '')
    rows = df[df['sub_clean'] == sub_clean]
    return sorted(rows['ses_num'].tolist())


def get_post_sessions(sub):
    """Session numbers with pre-surgical sessions removed.

    post_sessions[0] is the canonical anatomy/registration anchor.
    Use this (not get_sessions) for all post-surgery analyses.
    """
    sid = f"sub-{sub.replace('sub-', '')}"
    pre = pre_surgery_sessions.get(sid, set())
    return [s for s in get_sessions(sub) if s not in pre]


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