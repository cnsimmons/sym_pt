#!/bin/bash
#
# 00_extract_confounds.sh - Extract motion spike confound regressors
# Uses fsl_motion_outliers to identify high-motion timepoints
# Must run BEFORE 1stLevel FEAT (spike files are included as confound EVs)
#
# Output: sub-{sub}_ses-{ses}_task-loc_run-{run}_bold_spikes.txt
#   in: sym_pt/sub-{sub}/ses-{ses}/derivatives/fsl/loc/run-{run}/
#
# Usage: bash 00_extract_confounds.sh
#

RAW_DIR='/lab_data/behrmannlab/hemi/Raw'
DATA_DIR='/user_data/csimmon2/sym_pt'
GIT_DIR='/user_data/csimmon2/git_repos/sym_pt'
CSV_FILE="${GIT_DIR}/sub_info.csv"
TASK='loc'

# Match original long_pt parameters
FD_THRESH=0.5
DUMMY=0

# Skip list (from sym_pt_params)
SKIP_SUBS=("108")

echo "============================================="
echo "MOTION SPIKE EXTRACTION (fsl_motion_outliers)"
echo "============================================="
echo "Raw data: ${RAW_DIR}"
echo "Output:   ${DATA_DIR}"
echo "FD threshold: ${FD_THRESH}"
echo ""

should_skip() {
    local sub="$1"
    for skip in "${SKIP_SUBS[@]}"; do
        [[ "$sub" == "$skip" ]] && return 0
    done
    return 1
}

# Read CSV and process
while IFS=',' read -r sub ses rest; do
    [[ "$sub" == "sub" ]] && continue  # skip header

    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')

    should_skip "$sub_clean" && continue

    # Find all runs for this session
    func_dir="${RAW_DIR}/sub-${sub_clean}/ses-${ses_clean}/func"

    if [ ! -d "$func_dir" ]; then
        echo "SKIP: sub-${sub_clean} ses-${ses_clean} (no func directory)"
        continue
    fi

    for bold_file in "${func_dir}"/sub-${sub_clean}_ses-${ses_clean}_task-${TASK}_run-*_bold.nii.gz; do
        [ ! -f "$bold_file" ] && continue

        run=$(basename "$bold_file" | sed -n 's/.*run-\([0-9]*\)_bold.nii.gz/\1/p')

        # Output directory and file
        out_dir="${DATA_DIR}/sub-${sub_clean}/ses-${ses_clean}/derivatives/fsl/${TASK}/run-${run}"
        out_file="${out_dir}/sub-${sub_clean}_ses-${ses_clean}_task-${TASK}_run-${run}_bold_spikes.txt"

        # Skip if already exists
        if [ -f "$out_file" ]; then
            echo "EXISTS: sub-${sub_clean} ses-${ses_clean} run-${run}"
            continue
        fi

        echo "Processing: sub-${sub_clean} ses-${ses_clean} run-${run}"
        mkdir -p "$out_dir"

        # Run fsl_motion_outliers
        fsl_motion_outliers \
            -i "$bold_file" \
            -o "$out_file" \
            --fd \
            --thresh=${FD_THRESH} \
            --dummy=${DUMMY} \
            2>&1

        # Check result
        if [ -f "$out_file" ] && [ -s "$out_file" ]; then
            n_spikes=$(head -1 "$out_file" | awk '{print NF}')
            n_timepoints=$(fslnvols "$bold_file")
            pct=$(echo "scale=1; $n_spikes * 100 / $n_timepoints" | bc)
            echo "  -> ${n_spikes} spikes / ${n_timepoints} timepoints (${pct}%)"

            # Warn if excessive motion
            if [ "$n_spikes" -gt "$((n_timepoints / 2))" ]; then
                echo "  WARNING: >50% timepoints flagged!"
            fi
        else
            echo "  -> No spikes detected (clean run)"
            # Create empty marker file so we don't re-process
            touch "$out_file"
        fi
    done

done < "$CSV_FILE"

echo ""
echo "============================================="
echo "Confound extraction complete!"
echo "============================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Verify spike files exist in derivatives/fsl/loc/run-*/  "
echo "  2. Ensure 1stLevel FSF confoundevs point to correct paths"
echo "  3. Re-run 1stLevel FEAT (bash 06_submit_1stLevel_feat.sh)"
echo "  4. Then re-run HighLevel and registration steps"