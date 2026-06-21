#!/bin/bash
#
# patch_template_fsf.sh - Add raw category contrasts (COPEs 15-19) to template
# Run once, then verify with: grep -E "conname|ncon" template_1stLevel.fsf
#

TEMPLATE="/user_data/csimmon2/git_repos/sym_pt/A_preprocessing/template_1stLevel.fsf"

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template not found at $TEMPLATE"
    exit 1
fi

echo "Patching $TEMPLATE..."

# 1. Update contrast counts from 14 to 19
sed -i 's/set fmri(ncon_orig) 14/set fmri(ncon_orig) 19/' "$TEMPLATE"
sed -i 's/set fmri(ncon_real) 14/set fmri(ncon_real) 19/' "$TEMPLATE"

# 2. Append new contrast definitions
cat >> "$TEMPLATE" << 'EOF'

# ── Raw category contrasts (COPEs 15-19) ────────────────────────────────

# COPE 15: Face (raw)
set fmri(conpic_orig.15) 1
set fmri(conpic_real.15) 1
set fmri(conname_orig.15) "Face_raw"
set fmri(conname_real.15) "Face_raw"
# orig: Face House Object Word Scramble
set fmri(con_orig15.1) 1
set fmri(con_orig15.2) 0
set fmri(con_orig15.3) 0
set fmri(con_orig15.4) 0
set fmri(con_orig15.5) 0
# real: Face Face_td House House_td Object Object_td Word Word_td Scramble Scramble_td
set fmri(con_real15.1) 1
set fmri(con_real15.2) 0
set fmri(con_real15.3) 0
set fmri(con_real15.4) 0
set fmri(con_real15.5) 0
set fmri(con_real15.6) 0
set fmri(con_real15.7) 0
set fmri(con_real15.8) 0
set fmri(con_real15.9) 0
set fmri(con_real15.10) 0

# COPE 16: House (raw)
set fmri(conpic_orig.16) 1
set fmri(conpic_real.16) 1
set fmri(conname_orig.16) "House_raw"
set fmri(conname_real.16) "House_raw"
set fmri(con_orig16.1) 0
set fmri(con_orig16.2) 1
set fmri(con_orig16.3) 0
set fmri(con_orig16.4) 0
set fmri(con_orig16.5) 0
set fmri(con_real16.1) 0
set fmri(con_real16.2) 0
set fmri(con_real16.3) 1
set fmri(con_real16.4) 0
set fmri(con_real16.5) 0
set fmri(con_real16.6) 0
set fmri(con_real16.7) 0
set fmri(con_real16.8) 0
set fmri(con_real16.9) 0
set fmri(con_real16.10) 0

# COPE 17: Object (raw)
set fmri(conpic_orig.17) 1
set fmri(conpic_real.17) 1
set fmri(conname_orig.17) "Object_raw"
set fmri(conname_real.17) "Object_raw"
set fmri(con_orig17.1) 0
set fmri(con_orig17.2) 0
set fmri(con_orig17.3) 1
set fmri(con_orig17.4) 0
set fmri(con_orig17.5) 0
set fmri(con_real17.1) 0
set fmri(con_real17.2) 0
set fmri(con_real17.3) 0
set fmri(con_real17.4) 0
set fmri(con_real17.5) 1
set fmri(con_real17.6) 0
set fmri(con_real17.7) 0
set fmri(con_real17.8) 0
set fmri(con_real17.9) 0
set fmri(con_real17.10) 0

# COPE 18: Word (raw)
set fmri(conpic_orig.18) 1
set fmri(conpic_real.18) 1
set fmri(conname_orig.18) "Word_raw"
set fmri(conname_real.18) "Word_raw"
set fmri(con_orig18.1) 0
set fmri(con_orig18.2) 0
set fmri(con_orig18.3) 0
set fmri(con_orig18.4) 1
set fmri(con_orig18.5) 0
set fmri(con_real18.1) 0
set fmri(con_real18.2) 0
set fmri(con_real18.3) 0
set fmri(con_real18.4) 0
set fmri(con_real18.5) 0
set fmri(con_real18.6) 0
set fmri(con_real18.7) 1
set fmri(con_real18.8) 0
set fmri(con_real18.9) 0
set fmri(con_real18.10) 0

# COPE 19: Scramble (raw)
set fmri(conpic_orig.19) 1
set fmri(conpic_real.19) 1
set fmri(conname_orig.19) "Scramble_raw"
set fmri(conname_real.19) "Scramble_raw"
set fmri(con_orig19.1) 0
set fmri(con_orig19.2) 0
set fmri(con_orig19.3) 0
set fmri(con_orig19.4) 0
set fmri(con_orig19.5) 1
set fmri(con_real19.1) 0
set fmri(con_real19.2) 0
set fmri(con_real19.3) 0
set fmri(con_real19.4) 0
set fmri(con_real19.5) 0
set fmri(con_real19.6) 0
set fmri(con_real19.7) 0
set fmri(con_real19.8) 0
set fmri(con_real19.9) 1
set fmri(con_real19.10) 0
EOF

echo "Done. Verify with:"
echo "  grep 'ncon' $TEMPLATE"
echo "  grep 'conname.*1[5-9]' $TEMPLATE"