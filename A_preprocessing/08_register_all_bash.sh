#!/bin/bash
# run_all_register.sh - Run registration for all subjects and sessions

# be sure to load fsl before running this script, e.g.:
# module load fsl/6.0.4

# Full path to your script
SCRIPT_PATH="A_preprocessing/08_register_1stLevel.py"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Get subject list from sym_pt_params
SUBJECTS=($(python3 << EOF
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir
from glob import glob
import os

# Data should be in /user_data/csimmon2/sym_pt/sub-0X
subject_dirs = glob(f'{processed_dir}/sub-*')
subjects = sorted([os.path.basename(d).replace('sub-', '') for d in subject_dirs if os.path.isdir(d)])
print(' '.join(subjects))
EOF
))

echo "Found ${#SUBJECTS[@]} subjects: ${SUBJECTS[@]}"
echo "Using script: $SCRIPT_PATH"
echo ""

for sub_num in "${SUBJECTS[@]}"; do
    sub="sub-${sub_num}"
    echo "================================================"
    echo "Processing $sub"
    echo "================================================"
    
    SESSIONS=($(python3 << EOF
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import get_sessions

sessions = get_sessions('$sub_num')
print(' '.join([f'{s:02d}' for s in sessions]))
EOF
))
    
    echo "Sessions for $sub: ${SESSIONS[@]}"
    
    for ses in "${SESSIONS[@]}"; do
        echo ""
        echo "Running: python3 $SCRIPT_PATH $sub $ses"
        python3 "$SCRIPT_PATH" "$sub" "$ses"
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed $sub ses-$ses"
        else
            echo "✗ Error processing $sub ses-$ses"
        fi
    done
done

echo ""
echo "All subjects processed!"