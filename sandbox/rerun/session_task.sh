#!/bin/bash
# One session's HighLevel step: line $SLURM_ARRAY_TASK_ID of sessions.txt.
set -euo pipefail
s=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/sessions.txt)
d=$(grep -m1 '^D=' "$s" | cut -d= -f2)
if [ -e "$d/RERUN_DONE" ]; then echo "skip, already done: $d"; exit 0; fi
bash "$s"
