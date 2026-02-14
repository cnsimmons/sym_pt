# 1. How many spike files exist total?
find /user_data/csimmon2/sym_pt -name "*spikes.txt" | wc -l

# 2. How many FSF files expect spike files?
find /user_data/csimmon2/sym_pt -name "1stLevel.fsf" | wc -l

# 3. Quick count: files with spikes vs empty (clean)
echo "With spikes:"
find /user_data/csimmon2/sym_pt -name "*spikes.txt" -size +0c | wc -l
echo "Clean (empty):"
find /user_data/csimmon2/sym_pt -name "*spikes.txt" -empty | wc -l

# 4. Any missing? (FSF exists but no spike file)
for fsf in /user_data/csimmon2/sym_pt/sub-*/ses-*/derivatives/fsl/loc/run-*/1stLevel.fsf; do
    run_dir=$(dirname "$fsf")
    sub=$(echo "$fsf" | grep -oP 'sub-\K[^/]+')
    ses=$(echo "$fsf" | grep -oP 'ses-\K[^/]+')
    run=$(echo "$fsf" | grep -oP 'run-\K[^/]+')
    spike="${run_dir}/sub-${sub}_ses-${ses}_task-loc_run-${run}_bold_spikes.txt"
    [ ! -f "$spike" ] && echo "MISSING: $sub $ses $run"
done