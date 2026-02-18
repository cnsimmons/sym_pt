#!/bin/bash
# Check all non-first sessions for registration quality
# Tests whether zstat1_ses{first} has any signal in the face searchmask

echo "=== REGISTRATION CHECK: non-first sessions ==="

for sub_dir in /user_data/csimmon2/sym_pt/sub-*/; do
    sub=$(basename $sub_dir)
    sub_clean=${sub/sub-/}
    
    # Find first session
    first_ses=$(ls -d ${sub_dir}ses-* 2>/dev/null | head -1 | grep -oP 'ses-\K[0-9]+')
    [ -z "$first_ses" ] && continue
    
    # Face searchmask from first session
    MASK="${sub_dir}ses-${first_ses}/ROIs/l_face_searchmask.nii.gz"
    [ ! -f "$MASK" ] && MASK="${sub_dir}ses-${first_ses}/ROIs/r_face_searchmask.nii.gz"
    [ ! -f "$MASK" ] && continue
    
    for ses_dir in ${sub_dir}ses-*/; do
        ses=$(basename $ses_dir | sed 's/ses-//')
        [ "$ses" = "$first_ses" ] && continue
        
        zstat="${ses_dir}derivatives/fsl/loc/HighLevel.gfeat/cope1.feat/stats/zstat1_ses${first_ses}.nii.gz"
        [ ! -f "$zstat" ] && continue
        
        vals=$(fslstats $zstat -k $MASK -m -R 2>/dev/null)
        mean=$(echo $vals | awk '{print $1}')
        max=$(echo $vals | awk '{print $3}')
        
        if [ "$(echo "$max == 0" | bc -l)" = "1" ]; then
            echo "FAIL  ${sub} ses-${ses}: ALL ZEROS (registration failure)"
        elif [ "$(echo "$max < 1.0" | bc -l)" = "1" ]; then
            echo "WARN  ${sub} ses-${ses}: max=${max} (possible partial misalignment)"
        else
            echo "OK    ${sub} ses-${ses}: mean=${mean} max=${max}"
        fi
    done
done