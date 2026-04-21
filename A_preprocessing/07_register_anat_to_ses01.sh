#!/bin/bash
#
# 07_register_anat_to_ses01.sh - Create inter-session anat registration matrices
# For each subject with multiple sessions, registers later session anats
# to the first session's anat using FLIRT (dof 6, rigid body)

dataDir='/user_data/csimmon2/sym_pt'
CSV_FILE='/user_data/csimmon2/git_repos/sym_pt/sub_info.csv'

SKIP_SUBS=()

should_skip() {
    local sub="$1"
    for skip in "${SKIP_SUBS[@]}"; do
        [[ "$sub" == "$skip" ]] && return 0
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
for sub_clean in $(echo "${!FIRST_SES[@]}" | tr ' ' '\n' | sort); do
    should_skip "$sub_clean" && continue

    first_ses="${FIRST_SES[$sub_clean]}"
    ref_anat="$dataDir/sub-${sub_clean}/ses-${first_ses}/anat/T1w_brain.nii.gz"

    [ ! -f "$ref_anat" ] && continue

    echo "=== sub-${sub_clean} (reference: ses-${first_ses}) ==="

    for ses_dir in "$dataDir"/sub-${sub_clean}/ses-*/; do
        [ ! -d "$ses_dir" ] && continue
        ses=$(basename "$ses_dir" | sed 's/ses-//')

        # Skip first session
        [ "$ses" == "$first_ses" ] && continue

        input_anat="$ses_dir/anat/T1w_brain.nii.gz"
        output_mat="$ses_dir/anat/anat2ses${first_ses}.mat"

        if [ -f "$input_anat" ] && [ ! -f "$output_mat" ]; then
            echo "  Creating ses-${ses} -> ses-${first_ses} matrix"
            flirt -in "$input_anat" -ref "$ref_anat" -omat "$output_mat" \
                  -dof 6 -cost corratio
        elif [ -f "$output_mat" ]; then
            echo "  ses-${ses} -> ses-${first_ses} already exists"
        else
            echo "  ses-${ses}: anat not found"
        fi
    done
done

echo ""
echo "Done!"