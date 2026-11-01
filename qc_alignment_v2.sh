#!/bin/bash
# qc_alignment_v2.sh - pictures for checking alignment by eye, one web page per batch.
#
# Focus: the moves between subject space and MNI. Pictures are 3x larger than v1, use
# slices through ventral occipitotemporal cortex instead of mid-slices, and show masks
# and z-maps as solid colour (the way FEAT draws its activation pictures).
#
#   1  functional on anatomy  FEAT's reg/example_func2highres.png (run-01). For later
#                             sessions the anatomy is the first session's.
#   2  anatomy -> MNI         first-session T1 moved to MNI with anat2stand.mat; red = MNI
#                             template outline. Slices: x = +40 / -40, y = -54, z = -12 / -18 mm.
#   3  MNI -> subject         first-session T1 with the four searchmasks (FFA, VWFA, PPA strict,
#                             LO), which were moved in with mni2anat.mat, the exact inverse of
#                             anat2stand.mat. Slices through the masks' centre.
#   4  z-map in MNI           MNI template with this session's face-vs-others map (cope 6),
#                             z > 3.1 in colour, at the same slices as picture 2.
#
# Usage (on the cluster, FSL on your PATH):
#   bash qc_alignment_v2.sh sub-079 sub-diff          # every session of these subjects
#   bash qc_alignment_v2.sh sub-004/ses-07
# Writes pipeline/outputs/qc/alignment_<date_time>/index.html; copy that folder to your
# laptop and open index.html.
set -uo pipefail

P=/user_data/csimmon2/git_repos/sym_pt/pipeline
S=/user_data/csimmon2/sym_pt
MNI=/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz
MASKS="face_FFA word_VWFA house_PPA_strict object_LOC"
# MNI 2 mm voxel indices: x = +40 -> 25, x = -40 -> 65, y = -54 -> 36, z = -12 -> 30, z = -18 -> 27
MNI_SLICES=(-x -25 -x -65 -y -36 -z -30 -z -27)
[ $# -eq 0 ] && { sed -n 2,26p "$0"; exit 1; }

Q=$P/outputs/qc/alignment_$(date +%Y%m%d_%H%M)
mkdir -p "$Q"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

first_session() {   # lowest session in sub_info.csv: the registration anchor
  awk -F, -v s="$1" 'NR > 1 && $1 == s {print $2}' "$P/sub_info.csv" | sort | head -1
}

views() {           # views <image> <out.png> <slice args...>: one strip of slices, 3x scale
  local img=$1 out=$2; shift 2
  local parts=() i=0
  while [ $# -gt 0 ]; do
    slicer "$img" -s 3 "$1" "$2" "$TMP/v$i.png" || return 1
    parts+=("$TMP/v$i.png"); i=$((i + 1)); shift 2
  done
  local cmd=(pngappend "${parts[0]}")
  for p in "${parts[@]:1}"; do cmd+=(+ "$p"); done
  "${cmd[@]}" "$out"
}

views_edges() {     # same, with the edges of a second image in red
  local img=$1 edge=$2 out=$3; shift 3
  local parts=() i=0
  while [ $# -gt 0 ]; do
    slicer "$img" "$edge" -s 3 "$1" "$2" "$TMP/e$i.png" || return 1
    parts+=("$TMP/e$i.png"); i=$((i + 1)); shift 2
  done
  local cmd=(pngappend "${parts[0]}")
  for p in "${parts[@]:1}"; do cmd+=(+ "$p"); done
  "${cmd[@]}" "$out"
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

  # 2: anatomy -> MNI (first session only; every session's maps use this transform)
  p2=""
  if [ -f "$A/T1w_brain.nii.gz" ] && [ -f "$A/anat2stand.mat" ]; then
    flirt -in "$A/T1w_brain" -ref "$MNI" -applyxfm -init "$A/anat2stand.mat" -out "$TMP/t1mni" \
      && views_edges "$TMP/t1mni" "$MNI" "$Q/${sub}_anat2mni.png" "${MNI_SLICES[@]}" \
      && p2=${sub}_anat2mni.png
  fi

  # 3: MNI -> subject: searchmasks in colour on the first-session T1, at the masks' centre
  p3=""
  masks=()
  for m in $MASKS; do for h in l r; do
    f=$S/$sub/$anchor/ROIs/${h}_${m}_searchmask.nii.gz; [ -f "$f" ] && masks+=("$f")
  done; done
  if [ -f "$A/T1w_brain.nii.gz" ] && [ ${#masks[@]} -gt 0 ]; then
    cmd=(fslmaths "${masks[0]}"); for f in "${masks[@]:1}"; do cmd+=(-add "$f"); done
    if "${cmd[@]}" -bin "$TMP/masks"; then
      read -r cx cy cz <<< "$(fslstats "$TMP/masks" -C)"
      cy=${cy%.*}; cz=${cz%.*}
      overlay 1 0 "$A/T1w_brain" -a "$TMP/masks" 0.5 1.5 "$TMP/masks_on_t1" \
        && views "$TMP/masks_on_t1" "$Q/${sub}_searchmasks.png" \
             -z -"$cz" -z -$((cz - 8)) -z -$((cz + 8)) -y -"$cy" -y -$((cy - 15)) \
        && p3=${sub}_searchmasks.png
    fi
  fi

  for ses in $sessions; do
    D=$S/$sub/$ses/derivatives/fsl/loc
    p1=""; f=$D/run-01/1stLevel.feat/reg/example_func2highres.png
    [ -f "$f" ] && cp "$f" "$Q/${sub}_${ses}_func2anat.png" && p1=${sub}_${ses}_func2anat.png
    # 4: this session's MNI z-map in colour on the template
    p4=""; z=$D/HighLevel.gfeat/cope6.feat/stats/zstat1_mni.nii.gz
    if [ -f "$z" ]; then
      overlay 1 0 "$MNI" -a "$z" 3.1 8 "$TMP/z_on_mni" \
        && views "$TMP/z_on_mni" "$Q/${sub}_${ses}_zmap_mni.png" "${MNI_SLICES[@]}" \
        && p4=${sub}_${ses}_zmap_mni.png
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
td,th{border:1px solid #ccc;padding:4px;vertical-align:top} th{background:#eef;width:7%}
td{width:23%} .miss{color:#b00}</style></head><body>
<h2>Alignment QC - $(date '+%Y-%m-%d %H:%M')</h2>
<p><b>1</b> functional (grey) with the anatomy's outline (red), from FEAT; for later sessions the
anatomy is the first session's. <b>2</b> anatomy moved to MNI, red = template outline; slices
x = +40 / -40, y = -54, z = -12 / -18 mm. <b>3</b> searchmasks (colour) moved from MNI onto the
first-session anatomy; axial slices through the masks' centre and 8 mm above / below, then two
coronal slices. <b>4</b> this session's face-vs-others map (z &gt; 3.1, colour) in MNI, same
slices as 2. Click a picture to enlarge.</p>
<table><tr><th></th><th>1 functional on anatomy</th><th>2 anatomy &rarr; MNI</th>
<th>3 MNI &rarr; subject (searchmasks)</th><th>4 z-map in MNI</th></tr>
$rows
</table></body></html>
EOF
echo "open: $Q/index.html"
