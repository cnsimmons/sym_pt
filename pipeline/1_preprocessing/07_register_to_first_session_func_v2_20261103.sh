#!/usr/bin/env bash
# 07_register_to_first_session_func.sh            added 2026-11-01, v2 2026-11-03
#
# Registers each run of a LATER session to the first-session T1 in two steps:
#   1. start:  later-session example_func -> first-session example_func
#              (6 dof, full search), then the first session's FEAT
#              example_func2highres.mat (route of Bonelli et al. 2012)
#   2. finish: FEAT's own BBR (FLIRT -cost bbr, bbr.sch, FEAT's white-matter
#              segmentation of the first-session T1), started from step 1
#              (Greve & Fischl 2009)
# v1 (2026-11-01) stopped after step 1. The final space is unchanged
# (first-session T1w_brain), so HighLevel outputs stay in the space every
# downstream script expects.
# v1 validation: deviation_log_2026-11-02_v6.txt. v2 validation: to be added.
#
# QC, per run: mm BBR moved the start; BBR cost of the result and of FEAT's
# original (lower = better); mm from FEAT's original. Then run-to-run agreement
# (validated range 0.06-0.30 mm). Nothing is decided automatically.
# FEAT's original reg/ = the earliest reg.bak_*, or reg/ itself if none exists.
#
# --dry-run  compute and print the QC only; nothing in reg/ is changed.
# A session with HighLevel.gfeat.longitudinal is shared with the cross-sectional
# study; it is refused unless --dry-run.
#
# Run AFTER first-level FEAT (step 06) and BEFORE the HighLevel steps (10-11).
# The old steps 07-09 were removed on 2026-11-01 (see the deviation log).
#
# Usage (from pipeline/, fmri env, FSL loaded):
#   bash 1_preprocessing/07_register_to_first_session_func.sh [--dry-run] sub-092 02 [sub-108 02 ...]
set -euo pipefail

S=/user_data/csimmon2/sym_pt
SUB_INFO=/user_data/csimmon2/git_repos/sym_pt/pipeline/sub_info.csv
REF_RUN=02                      # first-session run used as reference (validated choice)
STAMP=$(date +%Y%m%d)
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi

if [ $# -lt 2 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "Usage: $0 [--dry-run] sub-XXX NN [sub-YYY NN ...]"; exit 1
fi
command -v flirt >/dev/null || { echo "FSL not loaded"; exit 1; }
if [ -z "${FSLDIR:-}" ] || [ ! -f "$FSLDIR/etc/flirtsch/bbr.sch" ]; then
    echo "FSLDIR not set (needed for bbr.sch)"; exit 1
fi
trap 'rm -rf /tmp/s07_$$_* /tmp/hd_$$* /tmp/f_$$.mat /tmp/p_$$.mat' EXIT

first_ses() {   # lowest session number for this subject in sub_info.csv
    awk -F, -v s="$1" '$1==s { sub(/^ses-/,"",$2); n=$2+0; if (m=="" || n<m) m=n }
                       END   { if (m!="") print m }' "$SUB_INFO"
}
bbr_cost() {    # FEAT's BBR cost of transform $2, for the run whose FEAT reg/ is $1
    local out
    out=$(flirt -ref "$1/highres_head" -in "$1/example_func" -init "$2" -cost bbr \
                -wmseg "$1/example_func2highres_fast_wmseg" \
                -schedule "$FSLDIR/etc/flirtsch/measurecost1.sch" -omat "$W/cost.mat")
    echo "$out" | awk 'NR==1 {print $1}'
}

while [ $# -gt 0 ]; do
    sub=$1; ses=$(printf "%02d" $((10#$2))); shift 2
    f=$(first_ses "$sub")
    if [ -z "$f" ]; then echo "$sub: not in sub_info.csv - skipped"; continue; fi
    f=$(printf "%02d" "$f")
    if [ "$ses" = "$f" ]; then echo "$sub ses-$ses is the first session - nothing to do"; continue; fi

    loc1=$S/$sub/ses-$f/derivatives/fsl/loc
    loc2=$S/$sub/ses-$ses/derivatives/fsl/loc
    if [ "$DRY" -eq 0 ] && [ -d "$loc2/HighLevel.gfeat.longitudinal" ]; then
        echo "$sub ses-$ses is shared with the cross-sectional study - refused (use --dry-run)"; continue
    fi
    REF=$loc1/run-$REF_RUN/1stLevel.feat/reg
    [ -d "$REF" ] || REF=$( (ls -d $loc1/run-0?/1stLevel.feat/reg 2>/dev/null || true) | head -1)
    if [ -z "$REF" ] || [ ! -f "$REF/example_func2highres.mat" ]; then
        echo "$sub: no first-session reference reg/ - skipped"; continue
    fi
    T1=$S/$sub/ses-$f/anat/T1w_brain.nii.gz
    W=/tmp/s07_$$_${sub}_ses-$ses; mkdir -p "$W"
    mode="install"; [ "$DRY" -eq 1 ] && mode="dry run, nothing installed"
    echo "=== $sub ses-$ses -> ses-$f  (reference: ${REF#$S/}; $mode)"

    done_runs=()
    for R in $loc2/run-0?/1stLevel.feat/reg; do
        [ -d "$R" ] || continue
        run=$(echo "$R" | grep -oP 'run-\K[0-9]+')

        # FEAT must have used the first-session T1, or the concatenation below is invalid
        if ! fslmaths "$R/highres" -sub "$T1" -abs /tmp/hd_$$ 2>/dev/null; then
            echo "  run-$run: FEAT highres and ses-$f T1 differ in size - skipped"; continue
        fi
        md=$(fslstats /tmp/hd_$$ -R | awk '{print $2}')
        if [ "$(echo "$md > 0" | bc)" -eq 1 ]; then
            echo "  run-$run: FEAT highres is not the ses-$f T1 (maxdiff $md) - skipped"; continue
        fi
        O=$( (ls -d "$R".bak_* 2>/dev/null || true) | sort | awk 'NR==1')
        [ -n "$O" ] || O=$R
        if [ ! -f "$O/highres_head.nii.gz" ] || [ ! -f "$O/example_func2highres_fast_wmseg.nii.gz" ]; then
            echo "  run-$run: no FEAT BBR files (highres_head, wmseg) in ${O#$S/} - skipped"; continue
        fi

        w=$W/run-$run; mkdir -p "$w"
        # step 1: start
        flirt -in "$R/example_func" -ref "$REF/example_func" -dof 6 \
              -searchrx -180 180 -searchry -180 180 -searchrz -180 180 \
              -omat "$w/func2firstsesfunc.mat"
        convert_xfm -omat "$w/start.mat" -concat "$REF/example_func2highres.mat" "$w/func2firstsesfunc.mat"
        # step 2: FEAT's BBR, from that start
        flirt -in "$O/example_func" -ref "$O/highres_head" -init "$w/start.mat" -dof 6 -cost bbr \
              -wmseg "$O/example_func2highres_fast_wmseg" \
              -schedule "$FSLDIR/etc/flirtsch/bbr.sch" -omat "$w/example_func2highres.mat" >/dev/null

        moved=$(rmsdiff "$w/start.mat" "$w/example_func2highres.mat" "$O/example_func")
        c_new=$(bbr_cost "$O" "$w/example_func2highres.mat")
        c_feat=$(bbr_cost "$O" "$O/example_func2highres.mat")
        d_feat=$(rmsdiff "$O/example_func2highres.mat" "$w/example_func2highres.mat" "$O/example_func")
        awk -v r="$run" -v m="$moved" -v n="$c_new" -v o="$c_feat" -v d="$d_feat" 'BEGIN {
            printf "  run-%s: BBR moved the start %.2f mm | cost %.3f (FEAT %.3f) | %.2f mm from FEAT%s\n",
                   r, m, n, o, d, (n > o ? "   <-- worse fit than FEAT" : "") }'
        done_runs+=("$run")
    done
    if [ ${#done_runs[@]} -eq 0 ]; then echo "  no runs registered"; continue; fi

    # QC: run-to-run agreement of the new transforms (validated range 0.06-0.30 mm)
    for ((i=0; i<${#done_runs[@]}; i++)); do
        for ((j=i+1; j<${#done_runs[@]}; j++)); do
            a=${done_runs[i]}; b=${done_runs[j]}
            A=$loc2/run-$a/1stLevel.feat/reg; B=$loc2/run-$b/1stLevel.feat/reg
            flirt -in "$A/example_func" -ref "$B/example_func" -dof 6 -omat /tmp/f_$$.mat \
                  -searchrx -180 180 -searchry -180 180 -searchrz -180 180
            convert_xfm -omat /tmp/p_$$.mat -concat "$W/run-$b/example_func2highres.mat" /tmp/f_$$.mat
            mm=$(rmsdiff "$W/run-$a/example_func2highres.mat" /tmp/p_$$.mat "$A/example_func")
            flag=""
            if [ "$(echo "$mm > 1" | bc)" -eq 1 ]; then flag="   <-- over 1 mm, check"; fi
            echo "  QC run-$a vs run-$b: $mm mm$flag"
        done
    done

    [ "$DRY" -eq 1 ] && continue

    for run in "${done_runs[@]}"; do
        R=$loc2/run-$run/1stLevel.feat/reg; w=$W/run-$run
        if [ -d "$(dirname "$R")/reg_standard" ]; then
            echo "  run-$run: WARNING reg_standard exists - rerun steps 10-11 after this"
        fi
        [ -e "$R.bak_$STAMP" ] || cp -a "$R" "$R.bak_$STAMP"
        cp "$w/func2firstsesfunc.mat" "$w/example_func2highres.mat" "$R/"
        cp "$w/start.mat" "$R/example_func2highres_start.mat"
        convert_xfm -omat "$R/highres2example_func.mat"  -inverse "$R/example_func2highres.mat"
        convert_xfm -omat "$R/example_func2standard.mat" -concat "$R/highres2standard.mat" "$R/example_func2highres.mat"
        convert_xfm -omat "$R/standard2example_func.mat" -inverse "$R/example_func2standard.mat"
        flirt -in "$R/example_func" -ref "$R/highres"  -applyxfm -init "$R/example_func2highres.mat"  -out "$R/example_func2highres"
        flirt -in "$R/example_func" -ref "$R/standard" -applyxfm -init "$R/example_func2standard.mat" -out "$R/example_func2standard"
        # FEAT's own reg/*.png still show the ORIGINAL registration; this one shows the new one
        slicer "$R/example_func2highres" "$R/highres" -a "$R/example_func2highres_06b.png"
        echo "  run-$run: installed (previous reg/ kept in reg.bak_$STAMP)"
    done
done
