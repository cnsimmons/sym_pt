#!/usr/bin/env python3
"""
rerun_arrays.py - capped submission for the rerun prepared by rerun_prepare_v3.py.

Reads sandbox/rerun/manifest.csv and the session scripts that v3 wrote, and adds
five NEW files to sandbox/rerun/ (it never changes the existing ones):
    first_levels.txt      one first-level design per line (202)
    sessions.txt          one session script per line (65)
    first_level_task.sh   SLURM array task: runs one first level; skips it if done
    session_task.sh       SLURM array task: runs one session's HighLevel step; skips if done
    submit_capped.sh      submits both arrays, at most MAXRUN jobs running at a time
The HighLevel array starts once every first level has ended; each session step
still checks its own runs before it does anything. Use submit_capped.sh, not
v3's submit_all.sh.

Resources, from the test session (sub-008 ses-02, sacct 2026-11-14):
    first level  8.1-8.6 min, 1.13 GB peak  ->  request 3 GB, 30 min
    HighLevel    20.9 min,    1.63 GB peak  ->  request 4 GB, 60 min

Usage (fmri env), from /user_data/csimmon2/git_repos/sym_pt/sandbox:
    python rerun_arrays.py           # writes the five files (stops if any exists)
    bash rerun/submit_capped.sh      # submits
"""
import sys
from pathlib import Path

import pandas as pd

OUT     = Path('/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun')
MAXRUN  = 20                       # jobs running at once, per array
FL_MEM, FL_TIME = '3G', '00:30:00'
HL_MEM, HL_TIME = '4G', '01:00:00'
FSL_ENV = ('export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; '
           'export PATH=$FSLDIR/bin:$PATH')
COMMON  = '--nodes=1 --ntasks=1 --cpus-per-task=1'

man = pd.read_csv(OUT / 'manifest.csv')
fsfs = [Path(p) for p in man['fsf']]
sess = []
for sub, ses in man[['sub', 'ses']].drop_duplicates().itertuples(index=False):
    sess.append(OUT / 'jobs' / f'{sub[4:]}_{int(ses):02d}_session.sh')

missing = [p for p in fsfs + sess if not p.exists()]
if missing:
    sys.exit(f'STOP: {len(missing)} expected files are missing, e.g. {missing[:3]}')
new = {n: OUT / n for n in ['first_levels.txt', 'sessions.txt', 'first_level_task.sh',
                            'session_task.sh', 'submit_capped.sh']}
clash = [p for p in new.values() if p.exists()]
if clash:
    sys.exit(f'STOP, would overwrite: {clash}')

new['first_levels.txt'].write_text(''.join(f'{p}\n' for p in fsfs))
new['sessions.txt'].write_text(''.join(f'{p}\n' for p in sess))
new['first_level_task.sh'].write_text(f"""#!/bin/bash
# One first-level FEAT run: line $SLURM_ARRAY_TASK_ID of first_levels.txt.
{FSL_ENV}
set -euo pipefail
fsf=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {new['first_levels.txt']})
if [ -e "${{fsf%.fsf}}.feat" ]; then echo "skip, already run: ${{fsf%.fsf}}.feat"; exit 0; fi
feat "$fsf"
""")
new['session_task.sh'].write_text(f"""#!/bin/bash
# One session's HighLevel step: line $SLURM_ARRAY_TASK_ID of sessions.txt.
set -euo pipefail
s=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {new['sessions.txt']})
d=$(grep -m1 '^D=' "$s" | cut -d= -f2)
if [ -e "$d/RERUN_DONE" ]; then echo "skip, already done: $d"; exit 0; fi
bash "$s"
""")
L = OUT / 'logs'
new['submit_capped.sh'].write_text(f"""#!/bin/bash
# Submits the rerun as two capped arrays. Finished work is skipped by the tasks.
set -uo pipefail
L={L}; mkdir -p $L
if squeue -u "${{USER:-$(whoami)}}" -h -o %j | grep -qE '^rr[12]$'; then
  echo "STOP: rerun arrays are already queued (squeue)"; exit 1
fi
A=$(sbatch --parsable --array=1-{len(fsfs)}%{MAXRUN} --job-name=rr1 {COMMON} \\
      --mem={FL_MEM} --time={FL_TIME} --output=$L/rr1_%A_%a.out --error=$L/rr1_%A_%a.err \\
      {new['first_level_task.sh']}) || exit 1
A=${{A%%;*}}
B=$(sbatch --parsable --array=1-{len(sess)}%{MAXRUN} --dependency=afterany:$A --job-name=rr2 {COMMON} \\
      --mem={HL_MEM} --time={HL_TIME} --output=$L/rr2_%A_%a.out --error=$L/rr2_%A_%a.err \\
      {new['session_task.sh']}) || exit 1
echo "first levels: array $A, {len(fsfs)} tasks, at most {MAXRUN} running at once"
echo "HighLevels:   array ${{B%%;*}}, {len(sess)} tasks, starts when every first level has ended"
""")
for n in ['first_level_task.sh', 'session_task.sh', 'submit_capped.sh']:
    new[n].chmod(0o755)
print(f'wrote {", ".join(new)} in {OUT}')
print(f'{len(fsfs)} first levels, {len(sess)} sessions, at most {MAXRUN} running at once; '
      f'requests {FL_MEM}/{FL_TIME} and {HL_MEM}/{HL_TIME}')
