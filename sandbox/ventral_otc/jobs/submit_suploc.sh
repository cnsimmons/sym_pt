#!/bin/bash
set -uo pipefail
if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qx "vx_wotc_suploc"; then echo "STOP: vx_wotc_suploc already queued"; exit 1; fi
sbatch --parsable --job-name=vx_wotc_suploc --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=03:00:00 --output=/user_data/csimmon2/git_repos/sym_pt/sandbox/ventral_otc/logs/vx_wotc_suploc_%j.out --error=/user_data/csimmon2/git_repos/sym_pt/sandbox/ventral_otc/logs/vx_wotc_suploc_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/ventral_otc/jobs/wotc_suploc.sh || exit 1
echo "submitted 1 job; logs in /user_data/csimmon2/git_repos/sym_pt/sandbox/ventral_otc/logs"
