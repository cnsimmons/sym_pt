#!/bin/bash
# build_pipeline.sh - step 1: make pipeline/ self-sufficient by changing paths only.
#
# Builds pipeline/ from copies of the 55 files behind the v3 and longitudinal results
# (scripts, templates, both params files). Each copy keeps its path relative to the
# repo, so the hard-coded repo paths inside the scripts can all be pointed into
# pipeline/ with one replacement. Three replacements are made, nothing else:
#   /user_data/csimmon2/git_repos/sym_pt          -> .../git_repos/sym_pt/pipeline
#   /user_data/csimmon2/git_repos/long_pt/roiParcels -> pipeline/resources/roiParcels
#   group_results                                  -> group_results_pipeline
# The last one sends group-level outputs to a new folder in /user_data/csimmon2/sym_pt,
# so the validation run cannot overwrite the committed results.
#
# Also moves sub_info.csv and F_harmonization/sub_info_scanner.csv into pipeline/ and
# leaves a link at each old location, so there is one physical copy of each and the
# old scripts keep working. Raw and per-subject imaging data are not touched.
#
# Run once on the cluster:  bash build_pipeline.sh
set -euo pipefail

G=/user_data/csimmon2/git_repos/sym_pt
P=$G/pipeline
L=/user_data/csimmon2/git_repos/long_pt/roiParcels

if [ -L "$G/sub_info.csv" ]; then
  echo "sub_info.csv is already a link into pipeline/: this was built before. Stopping."; exit 1
fi
if [ -e "$P" ]; then
  mv "$P" "$G/pipeline_stages_$(date +%Y%m%d_%H%M)"
  echo "moved the earlier pipeline/ (stage-folder copies) aside"
fi
mkdir -p "$P/resources"

FILES=(
  A_preprocessing/00_mat2tsv_timing.py A_preprocessing/00_prt2tsv_timing.py
  A_preprocessing/01_organize.py A_preprocessing/1.5_extract_confounds.sh
  A_preprocessing/02_convert_timing.py A_preprocessing/03_submit_anatomy_new.sh
  A_preprocessing/register_mirror.py A_preprocessing/check_anatomy_03_register.ipynb
  A_preprocessing/05_create_fsf.sh A_preprocessing/template_1stLevel.fsf
  A_preprocessing/06_submit_1stLevel_feat.sh A_preprocessing/00_spike_verification.sh
  A_preprocessing/07_register_anat_to_ses01.sh A_preprocessing/08_submit_1stLevel_register.py
  A_preprocessing/register_1stLevel.py A_preprocessing/09_submit_zstat_register.py
  A_preprocessing/register_zstats.py A_preprocessing/10_create_highlevel_fsf.sh
  A_preprocessing/template_HighLevel.fsf A_preprocessing/11_submit_highlevel_feat.sh
  A_preprocessing/12_submit_highlevel_register.py A_preprocessing/register_highlevel.py
  A_preprocessing/check_registration.sh A_preprocessing/13_register2mni.py
  B_analyses/01_create_searchmasks.py B_analyses/add_ppa_strict_searchmasks.py
  D_liu/verified/01_univariate_analyses.py D_liu/verified/04_multivariate_analyses.py
  D_liu/liu_recreation_csv_v2.py D_liu/calc_peak_coords_mni.py
  extract_selective_voxel_counts.py H_RosenkeRSM/otc_rsm_rosenke.py
  D_liu/verified/02_tfce_analyses_dontuse_useharmony.py
  F_harmonization/add_scanner.py F_harmonization/combat_05b_harmonize_univariate_sqrt.py
  F_harmonization/combat_06_harmonize_rsa.py F_harmonization/combat_01_build_matrix.py
  F_harmonization/combat_02_harmonize.py F_harmonization/combat_03_tfce_harmonized.py
  F_harmonization/submit_combat.sh D_liu/verified/05_stats.py
  D_liu/verified/03_wta_analyses.py D_liu/verified/05_stats_harmony.py
  control_li.py marlene_grid.py marlene_lmm.py one_last_8.27/marlene_roi.py
  D_liu/verified/07_grid.py D_liu/verified/grid_cohort.py bf_patient_vs_control.py
  I_bayes/whole_otc_bayes.ipynb E_longitudinal/longitudinal_final_2026-10-02.ipynb
  final_figures.ipynb params.py sym_pt_params.py
)

# 1. copies, keeping each file's path relative to the repo
cd "$G"
for f in "${FILES[@]}"; do
  cp -p --parents "$f" "$P/"
  echo "$f" >> "$P/SOURCES.txt"
done

# 2. the MNI ROI parcels register_mirror.py copies in (until now read from long_pt)
cp -rp "$L" "$P/resources/roiParcels"

# 3. one physical copy of each subject table, inside pipeline/, linked from its old place
mv "$G/sub_info.csv" "$P/sub_info.csv" && ln -s "$P/sub_info.csv" "$G/sub_info.csv"
mkdir -p "$P/F_harmonization"
mv "$G/F_harmonization/sub_info_scanner.csv" "$P/F_harmonization/sub_info_scanner.csv" \
  && ln -s "$P/F_harmonization/sub_info_scanner.csv" "$G/F_harmonization/sub_info_scanner.csv"

# 4. the three path replacements, in code files only
find "$P" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.ipynb' -o -name '*.fsf' \) \
     -not -path "$P/resources/*" -print0 | xargs -0 sed -i \
  -e 's#/user_data/csimmon2/git_repos/sym_pt#/user_data/csimmon2/git_repos/sym_pt/pipeline#g' \
  -e 's#/user_data/csimmon2/git_repos/long_pt/roiParcels#/user_data/csimmon2/git_repos/sym_pt/pipeline/resources/roiParcels#g' \
  -e 's#group_results#group_results_pipeline#g'

# 5. checks
changed=0; other=0
while read -r f; do
  n=$(diff "$G/$f" "$P/$f" | grep -c '^>' || true)
  o=$(diff "$G/$f" "$P/$f" | grep '^>' | grep -cvE 'sym_pt/pipeline|group_results_pipeline|resources/roiParcels' || true)
  changed=$((changed + n)); other=$((other + o))
done < "$P/SOURCES.txt"
left=$(grep -rhoE '/user_data/csimmon2/git_repos/[A-Za-z0-9_]+(/[A-Za-z0-9_.-]+)?' "$P" \
         --include='*.py' --include='*.sh' --include='*.ipynb' | grep -v 'git_repos/sym_pt/pipeline' | sort -u || true)

cat > "$P/README.txt" <<'EOF'
pipeline/ - the scripts behind the v3 and longitudinal results.
Copies of the originals (SOURCES.txt), with repo paths pointed inside pipeline/,
the long_pt ROI parcels copied to resources/, and group-level outputs sent to
/user_data/csimmon2/sym_pt/group_results_pipeline. sub_info.csv and
F_harmonization/sub_info_scanner.csv live here; the old locations are links.
EOF

echo "files copied:              $(wc -l < "$P/SOURCES.txt")"
echo "lines changed:             $changed"
echo "changed lines not a path:  $other    (should be 0)"
echo "references outside pipeline/: ${left:-none}"
