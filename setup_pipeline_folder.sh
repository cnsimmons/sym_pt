#!/bin/bash
# setup_pipeline_folder.sh
# Copies the scripts that produce the v3 and longitudinal results into
# sym_pt/pipeline/, grouped by stage. Copies only: nothing in the repo is moved
# or edited, and data/config files (sub_info.csv, params.py, sym_pt_params.py,
# sub_info_scanner.csv) are NOT copied, so there is still one copy of each.
#
# The copies still reach templates, worker scripts and imported modules at their
# original full paths, so the originals must stay where they are.
#
# Run once on the cluster:  bash setup_pipeline_folder.sh
set -euo pipefail

G=/user_data/csimmon2/git_repos/sym_pt
P=$G/pipeline
if [ -e "$P" ]; then echo "$P already exists - stopping so nothing is overwritten"; exit 1; fi
mkdir -p "$P"

copy() {   # copy <stage folder> <path relative to repo>
  mkdir -p "$P/$1"
  cp -p "$G/$2" "$P/$1/"
  echo "$1/$(basename "$2")  <-  $2" >> "$P/SOURCES.txt"
}

# 1 - preprocessing, in A_preprocessing's own order (04 is already applied, so it is not copied)
for f in 00_mat2tsv_timing.py 00_prt2tsv_timing.py 01_organize.py 1.5_extract_confounds.sh \
         02_convert_timing.py 03_submit_anatomy_new.sh register_mirror.py check_anatomy_03_register.ipynb \
         05_create_fsf.sh template_1stLevel.fsf 06_submit_1stLevel_feat.sh 00_spike_verification.sh \
         07_register_anat_to_ses01.sh 08_submit_1stLevel_register.py register_1stLevel.py \
         09_submit_zstat_register.py register_zstats.py 10_create_highlevel_fsf.sh template_HighLevel.fsf \
         11_submit_highlevel_feat.sh 12_submit_highlevel_register.py register_highlevel.py \
         check_registration.sh 13_register2mni.py; do
  copy 1_preprocessing "A_preprocessing/$f"
done

# 2 - searchmasks
copy 2_searchmasks B_analyses/01_create_searchmasks.py
copy 2_searchmasks B_analyses/add_ppa_strict_searchmasks.py

# 3 - extraction
copy 3_extraction D_liu/verified/01_univariate_analyses.py
copy 3_extraction D_liu/verified/04_multivariate_analyses.py
copy 3_extraction D_liu/liu_recreation_csv_v2.py
copy 3_extraction D_liu/calc_peak_coords_mni.py
copy 3_extraction extract_selective_voxel_counts.py
copy 3_extraction H_RosenkeRSM/otc_rsm_rosenke.py
copy 3_extraction D_liu/verified/02_tfce_analyses_dontuse_useharmony.py   # imported by TFCE and whole-OTC

# 4 - harmonization (05_stats.py is copied because 05b and 06 import its cohort and session rules)
for f in add_scanner.py combat_05b_harmonize_univariate_sqrt.py combat_06_harmonize_rsa.py \
         combat_01_build_matrix.py combat_02_harmonize.py combat_03_tfce_harmonized.py submit_combat.sh; do
  copy 4_harmonization "F_harmonization/$f"
done
copy 4_harmonization D_liu/verified/05_stats.py

# 5 - analyses (07_grid imports grid_cohort from its own folder, so they travel together)
copy 5_analyses D_liu/verified/03_wta_analyses.py
copy 5_analyses D_liu/verified/05_stats_harmony.py
copy 5_analyses control_li.py
copy 5_analyses marlene_grid.py
copy 5_analyses marlene_lmm.py
copy 5_analyses one_last_8.27/marlene_roi.py
copy 5_analyses D_liu/verified/07_grid.py
copy 5_analyses D_liu/verified/grid_cohort.py
copy 5_analyses bf_patient_vs_control.py
copy 5_analyses I_bayes/whole_otc_bayes.ipynb
copy 5_analyses E_longitudinal/longitudinal_final_2026-10-02.ipynb

# 6 - figures
copy 6_figures final_figures.ipynb

cat > "$P/README.txt" <<'EOF'
pipeline/ - the scripts behind the v3 and longitudinal results, grouped by stage.
Exact copies; SOURCES.txt lists where each came from. Run order: sym_pt_run_order docx.
The copies still read templates, worker scripts, params files and sub_info.csv at
their original full paths. Leave the originals in place.
EOF

echo "done: $(wc -l < "$P/SOURCES.txt") files copied into $P"
