#!/bin/bash
#
# 10_create_highlevel_fsf.sh - Create HighLevel (fixed effects) FSF files
# Combines runs within each session
#
# Usage: bash 10_create_highlevel_fsf.sh

dataDir='/user_data/csimmon2/sym_pt'
gitDir='/user_data/csimmon2/git_repos/sym_pt'
CSV_FILE="${gitDir}/sub_info.csv"
templateFSF="${gitDir}/template_HighLevel.fsf"

SKIP_SUBS=("108")

echo "Using template: $templateFSF"
echo ""

if [ ! -f "$templateFSF" ]; then
    echo "ERROR: Template HighLevel FSF not found at $templateFSF"
    echo "Copy a working HighLevel design.fsf to that path first."
    exit 1
fi

should_skip() {
    for skip in "${SKIP_SUBS[@]}"; do
        [[ "$1" == "$skip" ]] && return 0
    done
    return 1
}

# ── Build first-session lookup from CSV ──
declare -A FIRST_SES

while IFS=',' read -r sub ses rest; do
    [[ "$sub" == "sub" ]] && continue
    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')
    if [[ -z "${FIRST_SES[$sub_clean]}" ]] || (( 10#$ses_clean < 10#${FIRST_SES[$sub_clean]} )); then
        FIRST_SES[$sub_clean]=$ses_clean
    fi
done < "$CSV_FILE"

# ── Main loop ──
while IFS=',' read -r sub ses rest; do
    [[ "$sub" == "sub" ]] && continue
    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')

    should_skip "$sub_clean" && continue

    first_ses="${FIRST_SES[$sub_clean]}"
    session_dir="$dataDir/sub-${sub_clean}/ses-${ses_clean}"

    # Find completed FEAT runs
    runs=()
    for feat_dir in "$session_dir"/derivatives/fsl/loc/run-*/1stLevel.feat; do
        [ -d "$feat_dir" ] || continue
        # Check FEAT actually completed
        [ -f "$feat_dir/stats/cope1.nii.gz" ] || continue
        run=$(basename "$(dirname "$feat_dir")" | sed 's/run-//')
        runs+=("$run")
    done

    [ ${#runs[@]} -eq 0 ] && continue

    echo "=== sub-${sub_clean} ses-${ses_clean}: ${#runs[@]} runs ==="

    fsf_file="$session_dir/derivatives/fsl/loc/HighLevel.fsf"
    mkdir -p "$(dirname "$fsf_file")"
    [ -f "$fsf_file" ] && rm "$fsf_file"
    cp "$templateFSF" "$fsf_file"

    # Update paths
    sed -i "s|/user_data/csimmon2/long_pt|$dataDir|g" "$fsf_file"
    sed -i "s|/lab_data/behrmannlab/vlad/ptoc|$dataDir|g" "$fsf_file"
    sed -i "s/sub-004/sub-${sub_clean}/g" "$fsf_file"
    sed -i "s/ses-01/ses-${ses_clean}/g" "$fsf_file"

    # Standard = first session's anat
    first_ses_anat="$dataDir/sub-${sub_clean}/ses-${first_ses}/anat/T1w_brain.nii.gz"
    sed -i "s|set fmri(regstandard) \".*\"|set fmri(regstandard) \"$first_ses_anat\"|g" "$fsf_file"

    # Output directory
    sed -i "s|set fmri(outputdir) \".*\"|set fmri(outputdir) \"$session_dir/derivatives/fsl/loc/HighLevel\"|g" "$fsf_file"

    # Number of inputs
    sed -i "s/set fmri(multiple) [0-9]*/set fmri(multiple) ${#runs[@]}/g" "$fsf_file"
    sed -i "s/set fmri(npts) [0-9]*/set fmri(npts) ${#runs[@]}/g" "$fsf_file"

    # Set feat_files for each run
    for i in "${!runs[@]}"; do
        n=$((i + 1))
        feat_dir="$session_dir/derivatives/fsl/loc/run-${runs[i]}/1stLevel.feat"

        if grep -q "set feat_files($n)" "$fsf_file"; then
            sed -i "s|set feat_files($n) \".*\"|set feat_files($n) \"$feat_dir\"|g" "$fsf_file"
        else
            echo "set feat_files($n) \"$feat_dir\"" >> "$fsf_file"
        fi

        if ! grep -q "set fmri(groupmem.$n)" "$fsf_file"; then
            echo "set fmri(groupmem.$n) 1" >> "$fsf_file"
        fi

        if ! grep -q "set fmri(evg${n}.1)" "$fsf_file"; then
            echo "set fmri(evg${n}.1) 1" >> "$fsf_file"
        fi
    done
    
    for ((j=${#runs[@]}+1; j<=20; j++)); do
        sed -i "/set feat_files($j) /d" "$fsf_file"
        sed -i "/set fmri(groupmem.$j) /d" "$fsf_file"
        sed -i "/set fmri(evg${j}.1) /d" "$fsf_file"
    done

    echo "  Created: $fsf_file"

done < "$CSV_FILE"

echo ""
echo "HighLevel FSF creation complete!"
echo "Created FSF files:"
find "$dataDir" -name "HighLevel.fsf" -path "*/derivatives/fsl/loc/*" | wc -l