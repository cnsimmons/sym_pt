"""
Exclude sub-083 and sub-085 from RSA CSVs.
Reason: pathological RSA beta values (|β| > 100 in >15% of house sphere voxels),
causing artificially inflated Fisher-z correlations (>3.0).
These are controls — exclusion does not affect patient analyses.
"""

from pathlib import Path
import pandas as pd
from datetime import date

EXCLUDE     = ['sub-083', 'sub-085']
LIU_DIR     = Path(processed_dir) / 'group_results' / 'liu_distinctiveness'
COPE_SETS   = ['differential', 'cat_vs_scramble', 'hybrid']
TODAY       = date.today().isoformat()

log_lines = [
    f'# RSA Exclusion Log — {TODAY}',
    f'# Excluded: {EXCLUDE}',
    f'# Reason: pathological RSA beta values (|β| > 100 in >15% of house sphere voxels)',
    f'# Effect: artificially inflated Fisher-z (sub-083 max=3.71, sub-085 max=3.12)',
    f'# These are controls — patient analyses unaffected.',
    '',
]

for cope_set in COPE_SETS:
    for fname in [f'liu_distinctiveness_{cope_set}.csv',
                  f'pairwise_correlations_{cope_set}.csv']:
        fpath = LIU_DIR / fname
        if not fpath.exists():
            log_lines.append(f'SKIP (not found): {fname}')
            continue

        df      = pd.read_csv(fpath)
        id_col  = 'subject_id' if 'subject_id' in df.columns else 'subject'
        before  = len(df)
        df      = df[~df[id_col].isin(EXCLUDE)]
        removed = before - len(df)

        df.to_csv(fpath, index=False)
        msg = f'UPDATED {fname}: removed {removed} rows'
        log_lines.append(msg)
        print(msg)

# Write log
log_path = LIU_DIR / 'exclusion_log.txt'
with open(log_path, 'w') as f:
    f.write('\n'.join(log_lines))
print(f'\nLog saved: {log_path}')