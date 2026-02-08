#!/bin/bash
#
# 06_create_fsf.sh - Create FEAT .fsf files for sym_pt project
# Reads unified long-format CSV (one row per subject-session)
# Standard image = first session's T1w_brain for each subject
#
# Usage: bash 04_create_fsf.sh

# Configuration
dataDir='/user_data/csimmon2/sym_pt'
rawDataDir='/lab_data/behrmannlab/hemi/Raw'
gitDir='/user_data/csimmon2/git_repos/sym_pt'
CSV_FILE="${gitDir}/sub_info.csv"

# Template FSF - update this path to your template
templateFSF="${gitDir}/template_1stLevel.fsf"

# Template values to replace
templateSub="004"
templateSes="01"
templateRun="01"

# Skip list
SKIP_SUBS=("108")

echo "Using template: $templateFSF"
echo "Reading CSV: $CSV_FILE"
echo ""

if [ ! -f "$templateFSF" ]; then
    echo "ERROR: Template FSF not found at $templateFSF"
    echo "Copy your working design.fsf to that path first."
    exit 1
fi

# Function to check if subject should be skipped
should_skip() {
    local sub="$1"
    for skip in "${SKIP_SUBS[@]}"; do
        if [[ "$sub" == "$skip" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if required files exist
check_files_exist() {
    local sub="$1"
    local ses="$2"
    local run="$3"
    local first_ses="$4"

    # Check functional data (in Raw)
    local funcData="$rawDataDir/sub-${sub}/ses-${ses}/func/sub-${sub}_ses-${ses}_task-loc_run-${run}_bold.nii.gz"
    if [ ! -f "$funcData" ]; then
        echo "      Missing: functional data"
        return 1
    fi

    # Check timing files
    local covsDir="$dataDir/sub-${sub}/ses-${ses}/timing"
    for condition in Face House Object Word Scramble; do
        local timing_file="$covsDir/catloc_${sub}_run-${run}_${condition}.txt"
        if [ ! -f "$timing_file" ]; then
            echo "      Missing: timing file for ${condition}"
            return 1
        fi
    done

    # Check structural image (first session's anat)
    local structImage="$dataDir/sub-${sub}/ses-${first_ses}/anat/T1w_brain.nii.gz"
    if [ ! -f "$structImage" ]; then
        echo "      Missing: structural image (ses-${first_ses})"
        return 1
    fi

    return 0
}

# Function to create FSF file
create_fsf() {
    local sub="$1"
    local ses="$2"
    local run="$3"
    local first_ses="$4"

    local outputDir="$dataDir/sub-${sub}/ses-${ses}/derivatives/fsl/loc/run-${run}"
    local fsfFile="$outputDir/1stLevel.fsf"

    mkdir -p "$outputDir"
    [ -f "$fsfFile" ] && rm "$fsfFile"

    cp "$templateFSF" "$fsfFile"

    # Replace subject/session/run
    sed -i "s/sub-${templateSub}/sub-${sub}/g" "$fsfFile"
    sed -i "s/${templateSub}/${sub}/g" "$fsfFile"
    sed -i "s/ses-${templateSes}/ses-${ses}/g" "$fsfFile"
    sed -i "s/run-${templateRun}/run-${run}/g" "$fsfFile"

    # Functional data (from Raw)
    local funcData="$rawDataDir/sub-${sub}/ses-${ses}/func/sub-${sub}_ses-${ses}_task-loc_run-${run}_bold.nii.gz"
    sed -i "s|set feat_files(1) \".*\"|set feat_files(1) \"$funcData\"|g" "$fsfFile"

    # Timing files
    local covsDir="$dataDir/sub-${sub}/ses-${ses}/timing"
    sed -i "s|set fmri(custom1) \".*\"|set fmri(custom1) \"$covsDir/catloc_${sub}_run-${run}_Face.txt\"|g" "$fsfFile"
    sed -i "s|set fmri(custom2) \".*\"|set fmri(custom2) \"$covsDir/catloc_${sub}_run-${run}_House.txt\"|g" "$fsfFile"
    sed -i "s|set fmri(custom3) \".*\"|set fmri(custom3) \"$covsDir/catloc_${sub}_run-${run}_Object.txt\"|g" "$fsfFile"
    sed -i "s|set fmri(custom4) \".*\"|set fmri(custom4) \"$covsDir/catloc_${sub}_run-${run}_Word.txt\"|g" "$fsfFile"
    sed -i "s|set fmri(custom5) \".*\"|set fmri(custom5) \"$covsDir/catloc_${sub}_run-${run}_Scramble.txt\"|g" "$fsfFile"

    # Structural = first session's T1w_brain (serves as "standard" in FEAT)
    local structImage="$dataDir/sub-${sub}/ses-${first_ses}/anat/T1w_brain.nii.gz"
    sed -i "s|set highres_files(1) \".*\"|set highres_files(1) \"$structImage\"|g" "$fsfFile"

    # Output directory
    sed -i "s|set fmri(outputdir) \".*\"|set fmri(outputdir) \"$outputDir/1stLevel\"|g" "$fsfFile"

    # Update any leftover long_pt or hemispace paths
    sed -i "s|/user_data/csimmon2/long_pt|$dataDir|g" "$fsfFile"
    sed -i "s|/lab_data/behrmannlab/vlad/hemispace|$dataDir|g" "$fsfFile"

    echo "      Created: $fsfFile"
}

# ── Main Loop ────────────────────────────────────────────────────────────────
# Build first-session lookup from CSV
declare -A FIRST_SES

tail -n +2 "$CSV_FILE" | while IFS=',' read -r sub ses rest; do
    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')

    # Track first session per subject (CSV is not sorted, so compare)
    if [[ -z "${FIRST_SES[$sub_clean]}" ]] || (( 10#$ses_clean < 10#${FIRST_SES[$sub_clean]} )); then
        FIRST_SES[$sub_clean]=$ses_clean
    fi
done

# Second pass: create FSFs
echo "Creating FSF files..."
echo ""

# Re-read to get first sessions (associative arrays don't survive subshells)
declare -A FIRST_SES
while IFS=',' read -r sub ses rest; do
    [[ "$sub" == "sub" ]] && continue  # skip header
    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')
    if [[ -z "${FIRST_SES[$sub_clean]}" ]] || (( 10#$ses_clean < 10#${FIRST_SES[$sub_clean]} )); then
        FIRST_SES[$sub_clean]=$ses_clean
    fi
done < "$CSV_FILE"

while IFS=',' read -r sub ses rest; do
    [[ "$sub" == "sub" ]] && continue  # skip header
    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')

    should_skip "$sub_clean" && continue

    first_ses="${FIRST_SES[$sub_clean]}"

    echo "=== sub-${sub_clean} ses-${ses_clean} (first session: ${first_ses}) ==="

    # Auto-detect runs from Raw
    func_dir="$rawDataDir/sub-${sub_clean}/ses-${ses_clean}/func"

    if [ ! -d "$func_dir" ]; then
        echo "  SKIP: No func directory"
        continue
    fi

    for bold_file in "$func_dir"/sub-${sub_clean}_ses-${ses_clean}_task-loc_run-*_bold.nii.gz; do
        [ ! -f "$bold_file" ] && continue

        run=$(basename "$bold_file" | sed -n 's/.*run-\([0-9]*\)_bold.nii.gz/\1/p')

        echo "    Run ${run}:"
        if check_files_exist "$sub_clean" "$ses_clean" "$run" "$first_ses"; then
            create_fsf "$sub_clean" "$ses_clean" "$run" "$first_ses"
        else
            echo "      SKIPPING - missing required files"
        fi
    done

done < "$CSV_FILE"

echo ""
echo "FSF creation complete!"
echo "Created FSF files:"
find "$dataDir" -name "1stLevel.fsf" -path "*/derivatives/fsl/loc/run-*" | wc -l