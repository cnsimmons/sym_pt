#!/usr/bin/env python3
"""
grid_cohort.py — the cohort, defined once.

Imported by 07_grid.py and by the three grid scripts (marlene_grid.py,
marlene_lmm.py, marlene_roi.py). It exists because those three each load the
harmonized CSVs themselves, so a driver cannot filter for them; the exclusion
has to be applied before their group statistics are computed.

Must match EXCLUDE in D_liu/verified/05_stats_harmony.py. When these two
drifted apart in Sept 2026 the grid reported 38 controls while the stats file
reported 36, and the manuscript carried numbers from both.

sub-091, sub-095 and sub-096 are not named here — the age cap removes them.
"""

EXCLUDE = {
    'sub-017',   # polymicrogyria
    'sub-027',   # control exclusion
    'sub-084',   # control exclusion: no face/word response in mid-fusiform
}

AGE_CAP = 23.0
