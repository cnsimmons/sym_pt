#!/usr/bin/env python3
"""
rerun_wholeotc_extra.py - add the sessions that whole OTC uses but the ROI rerun did not.

Whole OTC (otc_rsm_rosenke.py through 02_tfce...load_subjects) takes each subject's
FIRST post-surgery session (params.get_post_sessions); the ROI measures take the
most recent. Nothing about session selection changes: this script finds, for every
subject in the record's whole-OTC file, its first post-surgery session, and sets up
the same rerun (no smoothing, MELODIC off, record registration, cope 20) for any of
those sessions that is not already in the rerun tree.

Same safeguards as rerun_prepare_v3.py: plan by default, writes only new files,
stops if anything it would write exists. Big outputs go under --root; the lists,
task scripts and the submit script go to sandbox/rerun/ with an extra_ prefix.

Usage (fmri env), from /user_data/csimmon2/git_repos/sym_pt/sandbox:
    python rerun_wholeotc_extra.py --root /lab_data/behrmannlab/claire          # plan
    python rerun_wholeotc_extra.py --root /lab_data/behrmannlab/claire write    # write
    bash rerun/submit_extra.sh                                                  # submit
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
RSM_CSV = REPO / 'H_RosenkeRSM' / 'otc_rsm_persubject.csv'

spec = importlib.util.spec_from_file_location('rp3', HERE / 'rerun_prepare_v3.py')
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)                      # same edits, checks and scripts as v3
sys.path.insert(0, str(REPO))
from params import get_post_sessions             # the rule load_subjects uses

ap = argparse.ArgumentParser()
ap.add_argument('mode', nargs='?', default='plan', choices=['plan', 'write'])
ap.add_argument('--root', required=True)
args = ap.parse_args()
rp.NEW = Path(args.root) / 'sym_pt_rerun'
if not rp.NEW.is_dir():
    sys.exit(f'STOP: {rp.NEW} does not exist; the main rerun should be there')

subs = sorted(pd.read_csv(RSM_CSV)['subject_id'].unique())
wanted, already = [], 0
for sid in subs:
    post = get_post_sessions(sid.replace('sub-', ''))
    if not post:
        sys.exit(f'STOP: {sid} has no post-surgery session in sub_info')
    ses = int(post[0])
    if (rp.NEW / sid / f'ses-{ses:02d}' / 'derivatives' / 'fsl' / rp.TREE).exists():
        already += 1
    else:
        wanted.append((sid, ses))
print(f'whole-OTC subjects: {len(subs)}   first post-op session already rerun: {already}   '
      f'to add: {len(wanted)} {[f"{s} ses-{n:02d}" for s, n in wanted]}')
if not wanted:
    sys.exit(0)

rp.sessions_of_record = lambda: wanted           # plan only these sessions
todo, problems = rp.plan()
print(f'sessions ready: {len(todo)}   runs: {sum(len(t["runs"]) for t in todo)}   problems: {len(problems)}')
for p in problems:
    print('  PROBLEM', p)
if args.mode != 'write':
    print('plan only; nothing written. Run with "write" to create the files.')
    sys.exit(0)
if problems:
    sys.exit('STOP: fix the problems above first; nothing written')

OUT, jobs = rp.OUT, rp.OUT / 'jobs'
tag = lambda t: f'{t["sub"][4:]}_{t["ses"]:02d}'
new = {n: OUT / n for n in ['extra_first_levels.txt', 'extra_sessions.txt', 'extra_first_level_task.sh',
                            'extra_session_task.sh', 'submit_extra.sh', 'extra_manifest.csv']}
targets = list(new.values()) + [jobs / f'{tag(t)}_session.sh' for t in todo] + \
          [t['hl_fsf'] for t in todo] + [r['fsf'] for t in todo for r in t['runs']]
clash = [p for p in targets if p.exists()]
if clash:
    sys.exit(f'STOP, would overwrite: {clash[:5]}')
for t in todo:
    for r in t['runs']:
        r['fsf'].parent.mkdir(parents=True)
        r['fsf'].write_text(r['text'])
    t['hl_fsf'].write_text(t['hl_text'])
    p = jobs / f'{tag(t)}_session.sh'
    p.write_text(rp.session_script(t)); p.chmod(0o755)
fsfs = [r['fsf'] for t in todo for r in t['runs']]
sess = [jobs / f'{tag(t)}_session.sh' for t in todo]
new['extra_first_levels.txt'].write_text(''.join(f'{p}\n' for p in fsfs))
new['extra_sessions.txt'].write_text(''.join(f'{p}\n' for p in sess))
new['extra_first_level_task.sh'].write_text(f"""#!/bin/bash
{rp.FSL_ENV}
set -euo pipefail
fsf=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {new['extra_first_levels.txt']})
if [ -e "${{fsf%.fsf}}.feat" ]; then echo "skip, already run: ${{fsf%.fsf}}.feat"; exit 0; fi
feat "$fsf"
""")
new['extra_session_task.sh'].write_text(f"""#!/bin/bash
set -euo pipefail
s=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {new['extra_sessions.txt']})
d=$(grep -m1 '^D=' "$s" | cut -d= -f2)
if [ -e "$d/RERUN_DONE" ]; then echo "skip, already done: $d"; exit 0; fi
bash "$s"
""")
L = OUT / 'logs'
common = '--nodes=1 --ntasks=1 --cpus-per-task=1'
new['submit_extra.sh'].write_text(f"""#!/bin/bash
set -uo pipefail
L={L}; mkdir -p $L
if squeue -u "${{USER:-$(whoami)}}" -h -o %j | grep -qE '^rr[45]$'; then echo "STOP: extra arrays already queued"; exit 1; fi
A=$(sbatch --parsable --array=1-{len(fsfs)}%20 --job-name=rr4 {common} --mem=3G --time=00:30:00 \\
      --output=$L/rr4_%A_%a.out --error=$L/rr4_%A_%a.err {new['extra_first_level_task.sh']}) || exit 1
A=${{A%%;*}}
B=$(sbatch --parsable --array=1-{len(sess)}%20 --dependency=afterany:$A --job-name=rr5 {common} --mem=4G --time=01:00:00 \\
      --output=$L/rr5_%A_%a.out --error=$L/rr5_%A_%a.err {new['extra_session_task.sh']}) || exit 1
echo "extra first levels: array $A, {len(fsfs)} tasks; extra HighLevels: array ${{B%%;*}}, {len(sess)} tasks"
""")
for n in ['extra_first_level_task.sh', 'extra_session_task.sh', 'submit_extra.sh']:
    new[n].chmod(0o755)
pd.DataFrame([{'sub': t['sub'], 'ses': t['ses'], 'run': r['run'], 'fsf': r['fsf'], 'reg_source': r['reg_src'],
               'anchor': t['anchor'], 'links': t['links']} for t in todo for r in t['runs']]
             ).to_csv(new['extra_manifest.csv'], index=False)
print(f'wrote {len(fsfs)} first-level and {len(todo)} HighLevel designs, {len(todo)} session scripts, '
      f'and the extra_ lists, task scripts, submit_extra.sh and extra_manifest.csv in {OUT}')
