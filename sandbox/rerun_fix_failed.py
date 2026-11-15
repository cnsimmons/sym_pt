#!/usr/bin/env python3
"""
rerun_fix_failed.py - finish the sessions whose rerun stopped because one first
level failed (FILM, "pinv(): svd failed", unsmoothed data).

Rule (C.S., 2026-11-14): if FILM cannot fit a run unsmoothed, that session's
HighLevel is built from its remaining runs. Applied to every session it affects.

For each such session it adds NEW files only:
    loc_rerun/HighLevel_fix.fsf      HighLevel.fsf with the failed run removed
                                     (inputs renumbered, npts and multiple updated)
    loc_rerun/RERUN_EXCEPTION.txt    which run was left out, and why
    sandbox/rerun/jobs/XXX_YY_session_fix.sh
    sandbox/rerun/submit_fix.sh      one job per session (4 GB, 1 h)
    sandbox/rerun/exceptions.txt     the same record for all sessions together
The failed run's partial folder is left in place and nothing reads it. A run's
registration that the first attempt already copied is kept only if identical to
the record's. Stops before writing if any of these files already exists.

Usage (fmri env), from /user_data/csimmon2/git_repos/sym_pt/sandbox:
    python rerun_fix_failed.py       # writes the files above, prints what it found
    bash rerun/submit_fix.sh         # submits
"""
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path('/user_data/csimmon2/git_repos/sym_pt/sandbox/rerun')
FSL_ENV = ('export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; '
           'export PATH=$FSLDIR/bin:$PATH')
PER_INPUT = [r'^set feat_files\(\d+\) .*\n', r'^set fmri\(evg\d+\.1\) .*\n',
             r'^set fmri\(groupmem\.\d+\) .*\n']

man = pd.read_csv(OUT / 'manifest.csv')
man['feat'] = [Path(p[:-4] + '.feat') for p in man['fsf']]
man['ok'] = [(f / 'stats' / 'cope20.nii.gz').exists() for f in man['feat']]
bad = man.groupby(['sub', 'ses']).filter(lambda g: not g['ok'].all())
if bad.empty:
    sys.exit('no session has a failed first level; nothing to do')

sessions, problems = [], []
for (sub, ses), g in bad.groupby(['sub', 'ses']):
    D = g['feat'].iloc[0].parent.parent
    keep, drop = g[g['ok']], g[~g['ok']]
    targets = [D / 'HighLevel_fix.fsf', D / 'RERUN_EXCEPTION.txt',
               OUT / 'jobs' / f'{sub[4:]}_{int(ses):02d}_session_fix.sh']
    if len(keep) < 2:
        problems.append(f'{sub} ses-{int(ses):02d}: fewer than 2 runs left')
    if (D / 'HighLevel.gfeat').exists() or (D / 'RERUN_DONE').exists():
        problems.append(f'{sub} ses-{int(ses):02d}: HighLevel already exists')
    problems += [f'would overwrite {p}' for p in targets if p.exists()]
    sessions.append({'sub': sub, 'ses': int(ses), 'D': D, 'keep': keep, 'drop': drop,
                     'anchor': int(g['anchor'].iloc[0]), 'links': bool(g['links'].iloc[0])})
for p in [OUT / 'submit_fix.sh', OUT / 'exceptions.txt']:
    if p.exists():
        problems.append(f'would overwrite {p}')
for s in sessions:
    print(f"{s['sub']} ses-{s['ses']:02d}: leave out {', '.join(s['drop']['run'])}; "
          f"HighLevel from {', '.join(s['keep']['run'])}")
if problems:
    sys.exit('STOP, nothing written:\n  ' + '\n  '.join(problems))

notes = []
for s in sessions:
    D, keep = s['D'], s['keep']
    t = (D / 'HighLevel.fsf').read_text()
    for pat in PER_INPUT:
        t = re.sub(pat, '', t, flags=re.M)
    n = len(keep)
    for key in ['npts', 'multiple']:
        t, k = re.subn(r'^(set fmri\(' + key + r'\) ).*$', rf'\g<1>{n}', t, flags=re.M)
        assert k >= 1, key
    block = ['', f'# inputs for the rerun exception: {", ".join(s["drop"]["run"])} left out']
    for i, f in enumerate(keep['feat'], 1):
        block += [f'set feat_files({i}) "{f}"', f'set fmri(evg{i}.1) 1', f'set fmri(groupmem.{i}) 1']
    t = t.rstrip('\n') + '\n' + '\n'.join(block) + '\n'
    # read back what FEAT will see
    files = dict(re.findall(r'^set feat_files\((\d+)\) "([^"]*)"', t, re.M))
    assert [files[str(i)] for i in range(1, n + 1)] == [str(f) for f in keep['feat']]
    assert re.findall(r'^set fmri\(ncopeinputs\) (\d+)', t, re.M)[-1] == '20'
    (D / 'HighLevel_fix.fsf').write_text(t)

    why = (f"{s['sub']} ses-{s['ses']:02d}: {', '.join(s['drop']['run'])} left out of the rerun HighLevel. "
           f"FILM (film_gls) failed on the unsmoothed data with 'pinv(): svd failed'; the same run "
           f"ran with 4 mm smoothing in the record. Rule (C.S., 2026-11-14): the session's HighLevel "
           f"uses its remaining runs ({', '.join(keep['run'])}). The record HighLevel is unchanged.")
    (D / 'RERUN_EXCEPTION.txt').write_text(why + '\n')
    notes.append(why)

    L = ['#!/bin/bash', '# HighLevel for a session with one failed first level (see RERUN_EXCEPTION.txt).',
         FSL_ENV, 'set -euo pipefail', f'D={D}', '']
    for f, src in zip(keep['feat'], keep['reg_source']):
        L += [f'[ -f {f}/stats/cope20.nii.gz ] || {{ echo "first level incomplete: {f}"; exit 1; }}',
              f'if [ -d {f}/reg ] && [ -z "$(ls -A {f}/reg)" ]; then rmdir {f}/reg; fi',
              f'if [ -e {f}/reg ]; then diff -rq {src} {f}/reg >/dev/null || {{ echo "reg differs from record: {f}"; exit 1; }}; '
              f'else cp -a {src} {f}/reg; fi']
    L += ['', '[ ! -e $D/HighLevel.gfeat ] || { echo "HighLevel.gfeat exists"; exit 1; }',
          'feat $D/HighLevel_fix.fsf', '']
    if s['links']:
        a = f"{s['anchor']:02d}"
        L += ['for c in $D/HighLevel.gfeat/cope*.feat/stats; do',
              f'  ln -s zstat1.nii.gz $c/zstat1_ses{a}.nii.gz', f'  ln -s cope1.nii.gz $c/cope1_ses{a}.nii.gz',
              'done', '']
    L += ['touch $D/RERUN_DONE', f"echo \"{s['sub']} ses-{s['ses']:02d} done\""]
    p = OUT / 'jobs' / f"{s['sub'][4:]}_{s['ses']:02d}_session_fix.sh"
    p.write_text('\n'.join(L) + '\n'); p.chmod(0o755)

(OUT / 'exceptions.txt').write_text('\n'.join(notes) + '\n')
logs = OUT / 'logs'
S = ['#!/bin/bash', '# One HighLevel job per session with a failed first level.', 'set -uo pipefail',
     f'L={logs}', 'if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qx rr3; then echo "STOP: rr3 jobs already queued"; exit 1; fi']
for s in sessions:
    tag = f"{s['sub'][4:]}_{s['ses']:02d}"
    S.append(f'sbatch --parsable --job-name=rr3 --output=$L/rr3_{tag}_%j.out --error=$L/rr3_{tag}_%j.err '
             f'--nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 {OUT}/jobs/{tag}_session_fix.sh || exit 1')
S.append(f'echo "{len(sessions)} sessions submitted"')
(OUT / 'submit_fix.sh').write_text('\n'.join(S) + '\n')
print(f'wrote HighLevel_fix.fsf and RERUN_EXCEPTION.txt in {len(sessions)} sessions; '
      f'session_fix scripts, submit_fix.sh and exceptions.txt in {OUT}')
