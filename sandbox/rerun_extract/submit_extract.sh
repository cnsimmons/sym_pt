#!/bin/bash
# Extraction stage for the rerun. Outputs: sandbox/rerun_extract/<job>/
set -uo pipefail
L=/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/logs; mkdir -p $L
if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qE "^rx_"; then echo "STOP: rx_ jobs already queued"; exit 1; fi
M=$(sbatch --parsable --array=1-60%20 --job-name=rx_mni --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:30:00 --output=$L/rx_mni_%A_%a.out --error=$L/rx_mni_%A_%a.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/mni_task.sh) || exit 1; M=${M%%;*}
sbatch --parsable --job-name=rx_uni_rec13 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=04:00:00 --output=$L/rx_uni_rec13_%j.out --error=$L/rx_uni_rec13_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/uni_rec13.sh || exit 1
sbatch --parsable --job-name=rx_uni_new13 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=04:00:00 --output=$L/rx_uni_new13_%j.out --error=$L/rx_uni_new13_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/uni_new13.sh || exit 1
sbatch --parsable --job-name=rx_uni_new20 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=04:00:00 --output=$L/rx_uni_new20_%j.out --error=$L/rx_uni_new20_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/uni_new20.sh || exit 1
sbatch --parsable --job-name=rx_rsa_rec13 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=06:00:00 --output=$L/rx_rsa_rec13_%j.out --error=$L/rx_rsa_rec13_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/rsa_rec13.sh || exit 1
sbatch --parsable --job-name=rx_rsa_new13 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=06:00:00 --output=$L/rx_rsa_new13_%j.out --error=$L/rx_rsa_new13_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/rsa_new13.sh || exit 1
sbatch --parsable --job-name=rx_rsa_new20 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=06:00:00 --output=$L/rx_rsa_new20_%j.out --error=$L/rx_rsa_new20_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/rsa_new20.sh || exit 1
sbatch --parsable --job-name=rx_wotc_rec --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=03:00:00 --output=$L/rx_wotc_rec_%j.out --error=$L/rx_wotc_rec_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/wotc_rec.sh || exit 1
sbatch --parsable --dependency=afterok:$M --job-name=rx_wotc_new --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=03:00:00 --output=$L/rx_wotc_new_%j.out --error=$L/rx_wotc_new_%j.err /user_data/csimmon2/git_repos/sym_pt/sandbox/rerun_extract/wotc_new.sh || exit 1
echo "submitted: MNI array (60 sessions, 20 at a time), 8 jobs; wotc_new waits for the MNI array"
