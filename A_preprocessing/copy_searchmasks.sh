#!/bin/bash
#
# copy_searchmasks.sh - Copy FreeSurfer-derived searchmasks from long_pt to sym_pt
#
# These searchmasks were created in long_pt using FreeSurfer recon-all parcellation,
# resampled to FSL anat space, combined per category, and dilated 1x.
# Only copies for subjects that exist in BOTH long_pt and sym_pt.
#
# Usage: bash copy_searchmasks.sh [--dry-run]

LONG_PT="/user_data/csimmon2/long_pt"
SYM_PT="/user_data/csimmon2/sym_pt"
CSV_FILE="/user_data/csimmon2/git_repos/sym_pt/sub_info.csv"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

SKIP_SUBS=("108")

should_skip() {
    for skip in "${SKIP_SUBS[@]}"; do
        [[ "$1" == "$skip" ]] && return 0
    done
    return 1
}

# Build first-session lookup from sym_pt CSV
declare -A FIRST_SES
while IFS=',' read -r sub ses rest; do
    [[ "$sub" == "sub" ]] && continue
    sub_clean=$(echo "$sub" | sed 's/sub-//')
    ses_clean=$(echo "$ses" | sed 's/ses-//')
    if [[ -z "${FIRST_SES[$sub_clean]}" ]] || (( 10#$ses_clean < 10#${FIRST_SES[$sub_clean]} )); then
        FIRST_SES[$sub_clean]=$ses_clean
    fi
done < "$CSV_FILE"

# Files to copy
MASKS=(
    "l_face_searchmask.nii.gz"  "r_face_searchmask.nii.gz"
    "l_word_searchmask.nii.gz"  "r_word_searchmask.nii.gz"
    "l_object_searchmask.nii.gz" "r_object_searchmask.nii.gz"
    "l_house_searchmask.nii.gz" "r_house_searchmask.nii.gz"
)

PARCEL_MASKS=(
    "l_fusiform_mask.nii.gz"         "r_fusiform_mask.nii.gz"
    "l_lateraloccipital_mask.nii.gz" "r_lateraloccipital_mask.nii.gz"
    "l_parahippocampal_mask.nii.gz"  "r_parahippocampal_mask.nii.gz"
    "l_lingual_mask.nii.gz"          "r_lingual_mask.nii.gz"
    "l_isthmuscingulate_mask.nii.gz" "r_isthmuscingulate_mask.nii.gz"
    "l_inferiortemporal_mask.nii.gz" "r_inferiortemporal_mask.nii.gz"
    "l_middletemporal_mask.nii.gz"   "r_middletemporal_mask.nii.gz"
)

ALL_FILES=("${MASKS[@]}" "${PARCEL_MASKS[@]}")

copied=0
skipped=0
no_long_pt=0

echo "Copying searchmasks: long_pt -> sym_pt"
echo "========================================"
$DRY_RUN && echo "(DRY RUN)"
echo ""

for sub_clean in $(echo "${!FIRST_SES[@]}" | tr ' ' '\n' | sort); do
    should_skip "$sub_clean" && continue

    first_ses="${FIRST_SES[$sub_clean]}"

    # Find ROIs dir in long_pt (check all sessions)
    long_roi_dir=""
    for long_ses_dir in "$LONG_PT"/sub-${sub_clean}/ses-*/ROIs; do
        if [ -d "$long_ses_dir" ]; then
            long_roi_dir="$long_ses_dir"
            break
        fi
    done

    if [ -z "$long_roi_dir" ]; then
        echo "  sub-${sub_clean}: NOT IN long_pt (needs Harvard-Oxford searchmasks)"
        ((no_long_pt++))
        continue
    fi

    sym_roi_dir="$SYM_PT/sub-${sub_clean}/ses-${first_ses}/ROIs"

    echo "=== sub-${sub_clean} ==="

    if ! $DRY_RUN; then
        mkdir -p "$sym_roi_dir"
    fi

    sub_copied=0
    for mask_file in "${ALL_FILES[@]}"; do
        src="$long_roi_dir/$mask_file"
        dst="$sym_roi_dir/$mask_file"

        if [ -f "$dst" ]; then
            ((skipped++))
            continue
        fi

        if [ -f "$src" ]; then
            if $DRY_RUN; then
                echo "  WOULD COPY: $mask_file"
            else
                cp "$src" "$dst"
            fi
            ((copied++))
            ((sub_copied++))
        fi
    done

    if ! $DRY_RUN; then
        echo "  Copied $sub_copied files -> $sym_roi_dir"
    fi
done

echo ""
echo "========================================"
echo "Copied: $copied files"
echo "Already existed: $skipped files"
echo "Subjects not in long_pt: $no_long_pt (need Harvard-Oxford searchmasks)"