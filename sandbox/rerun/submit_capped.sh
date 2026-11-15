#!/bin/bash
# Submits the rerun as two capped arrays. Finished work is skipped by the tasks.
set -uo pipefail
L=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/logs; mkdir -p $L
if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qE '^rr[12]$'; then
  echo "STOP: rerun arrays are already queued (squeue)"; exit 1
fi
A=$(sbatch --parsable --array=1-202%20 --job-name=rr1 --nodes=1 --ntasks=1 --cpus-per-task=1 \
      --mem=3G --time=00:30:00 --output=$L/rr1_%A_%a.out --error=$L/rr1_%A_%a.err \
      /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/first_level_task.sh) || exit 1
A=${A%%;*}
B=$(sbatch --parsable --array=1-65%20 --dependency=afterany:$A --job-name=rr2 --nodes=1 --ntasks=1 --cpus-per-task=1 \
      --mem=4G --time=01:00:00 --output=$L/rr2_%A_%a.out --error=$L/rr2_%A_%a.err \
      /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/session_task.sh) || exit 1
echo "first levels: array $A, 202 tasks, at most 20 running at once"
echo "HighLevels:   array ${B%%;*}, 65 tasks, starts when every first level has ended"
