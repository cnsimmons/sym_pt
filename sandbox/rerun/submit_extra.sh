#!/bin/bash
set -uo pipefail
L=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/logs; mkdir -p $L
if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qE '^rr[45]$'; then echo "STOP: extra arrays already queued"; exit 1; fi
A=$(sbatch --parsable --array=1-18%20 --job-name=rr4 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:30:00 \
      --output=$L/rr4_%A_%a.out --error=$L/rr4_%A_%a.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/extra_first_level_task.sh) || exit 1
A=${A%%;*}
B=$(sbatch --parsable --array=1-5%20 --dependency=afterany:$A --job-name=rr5 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 \
      --output=$L/rr5_%A_%a.out --error=$L/rr5_%A_%a.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/extra_session_task.sh) || exit 1
echo "extra first levels: array $A, 18 tasks; extra HighLevels: array ${B%%;*}, 5 tasks"
