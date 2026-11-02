#!/usr/bin/env bash
# 07_register_to_first_session_func.sh                         added 2026-11-01
#
# Replaces FEAT's func->T1 registration for a LATER session with:
#   later-session example_func -> first-session example_func (6 dof, full search)
#   then the first session's FEAT example_func2highres.mat.
# The final space is unchanged (first-session T1w_brain), so HighLevel outputs
# stay in the space every downstream script expects.
# Why and how it was validated: deviation_log_2026-11-01.txt
#
# Run AFTER first-level FEAT (step 06) and BEFORE the HighLevel steps (10-11).
# The old steps 07-09 were removed on 2026-11-01 (see the deviation log).
#
# Usage (from pipeline/, fmri env, FSL loaded):
#   bash 1_preprocessing/07_register_to_first_session_func.sh sub-092 02 [sub-108 02 ...]
set -euo pipefail

S=/user_data/csimmon2/sym_pt
SUB_INFO=/user_data/csimmon2/git_repos/sym_pt/pipeline/sub_info.csv
REF_RUN=02                      # first-session run used as reference (validated choice)
STAMP=$(date +%Y%m%d)

if [ $# -lt 2 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "Usage: $0 sub-XXX NN [sub-YYY NN ...]"; exit 1
fi
command -v flirt >/dev/null || { echo "FSL not loaded"; exit 1; }

first_ses() {   # lowest session number for this subject in sub_info.csv
    awk -F, -v s="$1" '$1==s { sub(/^ses-/,"",$2); n=$2+0; if (m=="" || n<m) m=n }
                       END   { if (m!="") print m }' "$SUB_INFO"
}

while [ $# -gt 0 ]; do
    sub=$1; ses=$(printf "%02d" $((10#$2))); shift 2
    f=$(first_ses "$sub")
    if [ -z "$f" ]; then echo "$sub: not in sub_info.csv - skipped"; continue; fi
    f=$(printf "%02d" "$f")
    if [ "$ses" = "$f" ]; then echo "$sub ses-$ses is the first session - nothing to do"; continue; fi

    loc1=$S/$sub/ses-$f/derivatives/fsl/loc
    loc2=$S/$sub/ses-$ses/derivatives/fsl/loc
    REF=$loc1/run-$REF_RUN/1stLevel.feat/reg
    [ -d "$REF" ] || REF=$( (ls -d $loc1/run-0?/1stLevel.feat/reg 2>/dev/null || true) | head -1)
    if [ -z "$REF" ] || [ ! -f "$REF/example_func2highres.mat" ]; then
        echo "$sub: no first-session reference reg/ - skipped"; continue
    fi
    T1=$S/$sub/ses-$f/anat/T1w_brain.nii.gz
    echo "=== $sub ses-$ses -> ses-$f  (reference: ${REF#$S/})"

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
        if [ -d "$(dirname "$R")/reg_standard" ]; then
            echo "  run-$run: WARNING reg_standard exists - rerun steps 10-11 after this"
        fi

        [ -e "$R.bak_$STAMP" ] || cp -a "$R" "$R.bak_$STAMP"
        flirt -in "$R/example_func" -ref "$REF/example_func" -dof 6 \
              -searchrx -180 180 -searchry -180 180 -searchrz -180 180 \
              -omat "$R/func2firstsesfunc.mat"
        convert_xfm -omat "$R/example_func2highres.mat"  -concat "$REF/example_func2highres.mat" "$R/func2firstsesfunc.mat"
        convert_xfm -omat "$R/highres2example_func.mat"  -inverse "$R/example_func2highres.mat"
        convert_xfm -omat "$R/example_func2standard.mat" -concat "$R/highres2standard.mat" "$R/example_func2highres.mat"
        convert_xfm -omat "$R/standard2example_func.mat" -inverse "$R/example_func2standard.mat"
        flirt -in "$R/example_func" -ref "$R/highres"  -applyxfm -init "$R/example_func2highres.mat"  -out "$R/example_func2highres"
        flirt -in "$R/example_func" -ref "$R/standard" -applyxfm -init "$R/example_func2standard.mat" -out "$R/example_func2standard"
        # FEAT's own reg/*.png still show the ORIGINAL registration; this one shows the new one
        slicer "$R/example_func2highres" "$R/highres" -a "$R/example_func2highres_06b.png"
        echo "  run-$run: installed (original kept in reg.bak_$STAMP)"
    done

    # QC: run-to-run agreement of the installed transforms (validated range 0.06-0.30 mm)
    runs=( $( (ls -d $loc2/run-0?/1stLevel.feat/reg 2>/dev/null || true) ) )
    for ((i=0; i<${#runs[@]}; i++)); do
        for ((j=i+1; j<${#runs[@]}; j++)); do
            A=${runs[i]}; B=${runs[j]}
            flirt -in "$A/example_func" -ref "$B/example_func" -dof 6 -omat /tmp/f_$$.mat \
                  -searchrx -180 180 -searchry -180 180 -searchrz -180 180
            convert_xfm -omat /tmp/p_$$.mat -concat "$B/example_func2highres.mat" /tmp/f_$$.mat
            mm=$(rmsdiff "$A/example_func2highres.mat" /tmp/p_$$.mat "$A/example_func")
            flag=""
            if [ "$(echo "$mm > 1" | bc)" -eq 1 ]; then flag="   <-- over 1 mm, check"; fi
            echo "  QC $(echo "$A" | grep -oP 'run-[0-9]+') vs $(echo "$B" | grep -oP 'run-[0-9]+'): $mm mm$flag"
        done
    done
done
rm -f /tmp/hd_$$* /tmp/f_$$.mat /tmp/p_$$.mat
