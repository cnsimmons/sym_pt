#!/bin/bash
# One HighLevel job per session with a failed first level.
set -uo pipefail
L=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/logs
if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qx rr3; then echo "STOP: rr3 jobs already queued"; exit 1; fi
sbatch --parsable --job-name=rr3 --output=$L/rr3_004_06_%j.out --error=$L/rr3_004_06_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/004_06_session_fix.sh || exit 1
sbatch --parsable --job-name=rr3 --output=$L/rr3_087_01_%j.out --error=$L/rr3_087_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/087_01_session_fix.sh || exit 1
sbatch --parsable --job-name=rr3 --output=$L/rr3_095_01_%j.out --error=$L/rr3_095_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/095_01_session_fix.sh || exit 1
sbatch --parsable --job-name=rr3 --output=$L/rr3_097_01_%j.out --error=$L/rr3_097_01_%j.err --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun/jobs/097_01_session_fix.sh || exit 1
echo "4 sessions submitted"
