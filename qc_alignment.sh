#!/bin/bash
# qc_alignment.sh - pictures for checking alignment by eye, one web page per batch.
#
# Four pictures per session:
#   1  functional on anatomy   FEAT's own reg/example_func2highres.png (run-01). For later
#                              sessions this is the session's functional on the FIRST-session
#                              anatomy, which is where cross-session misalignment shows up.
#   2  anatomy on MNI          the first-session T1 moved to MNI with anat2stand.mat, with the
#                              MNI template's outline in red
#   3  searchmasks on anatomy  the first-session T1 with the outlines of the four searchmasks
#                              (FFA, VWFA, PPA strict, LO; both hemispheres), axial slices
#   4  z-map in MNI            the MNI template with the outline of this session's face-vs-others
#                              map (cope 6, z > 3.1), as used by TFCE and whole-OTC
#
# Usage (on the cluster, FSL on your PATH):
#   bash qc_alignment.sh sub-079 sub-diff            # every session of these subjects
#   bash qc_alignment.sh sub-004/ses-07 sub-038/ses-01
# Writes pipeline/outputs/qc/alignment_<date_time>/index.html. Copy that folder to your
# laptop and open index.html in a browser.
set -uo pipefail

P=/user_data/csimmon2/git_repos/sym_pt/pipeline
S=/user_data/csimmon2/sym_pt
MNI=/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz
MASKS="face_FFA word_VWFA house_PPA_strict object_LOC"
[ $# -eq 0 ] && { sed -n 2,22p "$0"; exit 1; }

Q=$P/outputs/qc/alignment_$(date +%Y%m%d_%H%M)
mkdir -p "$Q"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

first_session() {   # lowest session for a subject in sub_info.csv (the registration anchor)
  awk -F, -v s="$1" 'NR > 1 && $1 == s {print $2}' "$P/sub_info.csv" | sort | head -1
}

cell() {            # <png or empty> <missing message>
  if [ -n "$1" ] && [ -f "$Q/$1" ]; then
    echo "<td><a href=\"$1\"><img src=\"$1\" width=\"100%\"></a></td>"
  else
    echo "<td class=\"miss\">$2</td>"
  fi
}

rows=""
for arg in "$@"; do
  sub=${arg%%/*}
  if [[ "$arg" == */* ]]; then sessions=${arg#*/}
  else sessions=$(awk -F, -v s="$sub" 'NR > 1 && $1 == s {print $2}' "$P/sub_info.csv" | sort); fi
  anchor=$(first_session "$sub")
  if [ -z "$anchor" ]; then echo "$sub: not in sub_info.csv - skipped"; continue; fi
  A=$S/$sub/$anchor/anat

  # 2 and 3 depend only on the first session; draw them once per subject
  p2=""; p3=""
  if [ -f "$A/T1w_brain.nii.gz" ] && [ -f "$A/anat2stand.mat" ]; then
    flirt -in "$A/T1w_brain" -ref "$MNI" -applyxfm -init "$A/anat2stand.mat" -out "$TMP/t1mni" \
      && slicer "$TMP/t1mni" "$MNI" -a "$Q/${sub}_anat2mni.png" && p2=${sub}_anat2mni.png
  fi
  masks=()
  for m in $MASKS; do for h in l r; do
    f=$S/$sub/$anchor/ROIs/${h}_${m}_searchmask.nii.gz; [ -f "$f" ] && masks+=("$f")
  done; done
  if [ -f "$A/T1w_brain.nii.gz" ] && [ ${#masks[@]} -gt 0 ]; then
    cmd=(fslmaths "${masks[0]}"); for f in "${masks[@]:1}"; do cmd+=(-add "$f"); done
    "${cmd[@]}" -bin "$TMP/masks" \
      && slicer "$A/T1w_brain" "$TMP/masks" -S 8 1400 "$Q/${sub}_searchmasks.png" && p3=${sub}_searchmasks.png
  fi

  for ses in $sessions; do
    D=$S/$sub/$ses/derivatives/fsl/loc
    p1=""; f=$D/run-01/1stLevel.feat/reg/example_func2highres.png
    [ -f "$f" ] && cp "$f" "$Q/${sub}_${ses}_func2anat.png" && p1=${sub}_${ses}_func2anat.png
    p4=""; z=$D/HighLevel.gfeat/cope6.feat/stats/zstat1_mni.nii.gz
    if [ -f "$z" ]; then
      fslmaths "$z" -thr 3.1 -bin "$TMP/blob" \
        && slicer "$MNI" "$TMP/blob" -a "$Q/${sub}_${ses}_zmap_mni.png" && p4=${sub}_${ses}_zmap_mni.png
    fi
    rows+="<tr><th>$sub<br>$ses<br><small>first session: $anchor</small></th>"
    rows+="$(cell "$p1" "no FEAT registration picture for run-01")"
    rows+="$(cell "$p2" "no anat2stand.mat / T1w_brain in $anchor")"
    rows+="$(cell "$p3" "no searchmasks in $anchor/ROIs")"
    rows+="$(cell "$p4" "no zstat1_mni for cope 6")</tr>"
    echo "$sub $ses done"
  done
done

cat > "$Q/index.html" <<EOF
<html><head><meta charset="utf-8"><title>alignment QC</title>
<style>body{font-family:Arial;font-size:13px} table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:4px;vertical-align:top} th{background:#eef;width:8%}
td{width:23%} .miss{color:#b00}</style></head><body>
<h2>Alignment QC - $(date '+%Y-%m-%d %H:%M')</h2>
<p>1: functional (grey) with the anatomy's outline, from FEAT. For later sessions the anatomy is
the first session's. &nbsp; 2: first-session anatomy moved to MNI, red = MNI template outline.
&nbsp; 3: first-session anatomy, red = searchmask outlines (FFA, VWFA, PPA strict, LO).
&nbsp; 4: MNI template, red = this session's face-vs-others map at z &gt; 3.1.
Click a picture to enlarge.</p>
<table><tr><th></th><th>1 functional on anatomy</th><th>2 anatomy on MNI</th>
<th>3 searchmasks on anatomy</th><th>4 z-map in MNI</th></tr>
$rows
</table></body></html>
EOF
echo "open: $Q/index.html"
