#!/usr/bin/env python3
"""
set_committed_cohorts.py - the two cohort lines that let pipeline/ reproduce the
committed results exactly.

  pipeline/params.py          skip_subs -> ['017'] only
      Extraction keeps the five later-excluded subjects (027, 084, 091, 095, 096),
      as it did when the committed univariate_v1.csv and rsa_v1.csv were made.
      Their univariate and RSA values then enter the committed 65-subject ComBat fit.
  pipeline/2_extraction/lib_tfce.py   load_subjects() also skips those five
      TFCE and whole-OTC stay on the committed 60 subjects.

The stats, grid and figure scripts keep their own EXCLUDE lists, so the analysed
cohort (36 / 12 / 12) does not change.

Run once on the cluster:   python set_committed_cohorts.py
"""
import os
import sys
from pathlib import Path

P = Path(os.environ.get('PIPE_ROOT', '/user_data/csimmon2/git_repos/sym_pt/pipeline'))

PARAMS_OLD = """# Single cohort definition for the whole pipeline. Scripts that load subjects
# through should_skip() inherit this; scripts with their own EXCLUDE list must
# match it.
skip_subs = ['017',              # polymicrogyria
             '027', '084',       # control exclusions
             '091', '095', '096']  # age cap <= 23
"""
PARAMS_NEW = """# Subjects never analysed. Extraction keeps 027, 084, 091, 095 and 096 so the
# univariate and RSA ComBat fit matches the committed one (65 subjects); they are
# excluded from TFCE and whole-OTC in 2_extraction/lib_tfce.py and from the stats
# by each script's EXCLUDE list.
skip_subs = ['017']              # polymicrogyria
"""

TFCE_DEF_OLD = "def load_subjects():\n"
TFCE_DEF_NEW = """# TFCE and whole-OTC cohort: the five subjects excluded from the analyses also stay
# out of these, as in the committed runs (60 subjects). sub-017 is in params.skip_subs.
TFCE_EXCLUDE = {'sub-027', 'sub-084', 'sub-091', 'sub-095', 'sub-096'}

def load_subjects():
"""
TFCE_SKIP_OLD = "        if should_skip(sid):\n            continue\n"
TFCE_SKIP_NEW = "        if should_skip(sid) or sid in TFCE_EXCLUDE:\n            continue\n"


def edit(path, pairs):
    text = path.read_text()
    for old, new in pairs:
        n = text.count(old)
        if new in text:
            sys.exit(f'STOPPED: {path.name} already has this change. Nothing written.')
        if n != 1:
            sys.exit(f'STOPPED: expected the original lines once in {path.name}, found {n}. Nothing written.')
        text = text.replace(old, new)
    return text


params = P / 'params.py'
tfce = P / '2_extraction' / 'lib_tfce.py'
new_params = edit(params, [(PARAMS_OLD, PARAMS_NEW)])
new_tfce = edit(tfce, [(TFCE_DEF_OLD, TFCE_DEF_NEW), (TFCE_SKIP_OLD, TFCE_SKIP_NEW)])
params.write_text(new_params)
tfce.write_text(new_tfce)

sys.path.insert(0, str(P))
import params as prm  # noqa: E402
print('params.skip_subs now:', prm.skip_subs)
print('lib_tfce.py: TFCE_EXCLUDE added and used in load_subjects()')
