#!/usr/bin/env python3
"""
rerun_prepare.py - set up the unsmoothed + cope-20 rerun of the cross-sectional FEAT.

What it builds, per session, in a NEW folder next to the existing one:
    /user_data/csimmon2/sym_pt/sub-XXX/ses-NN/derivatives/fsl/loc_rerun/
        run-0N/1stLevel.fsf    copy of the design FEAT actually ran
                               (loc/run-0N/1stLevel.feat/design.fsf) with five changes:
                                 smoothing 4 mm -> 0
                                 MELODIC exploration off
                                 registration off (the record's registration is
                                   copied in afterwards, so it is held constant)
                                 contrast 20 added: Words - Face (Face -1, Words +1)
                                 output folder -> loc_rerun
        HighLevel.fsf          copy of loc/HighLevel.gfeat/design.fsf with its run
                               inputs pointed at loc_rerun and cope 20 added
Sessions: every subject-session in the record's harmonized files, so a later ComBat
refit uses the same set as the record. sub-008 ses-02 takes its corrected
registration (reg.longitudinal); every other session keeps FEAT's (5 mm rule).

Writes nothing outside loc_rerun/ folders and sandbox/rerun/. Stops before writing
if any loc_rerun folder or output already exists. Nothing is ever overwritten.

Usage (fmri env), from /user_data/csimmon2/git_repos/sym_pt/sandbox:
    python rerun_prepare.py          # 1. plan only: checks every input, writes nothing
    python rerun_prepare.py write    # 2. writes the .fsf files and the submit scripts
    bash rerun/submit_test.sh        # 3. one session (sub-008 ses-02) through FEAT
    bash rerun/submit_all.sh         # 4. everything else (skips sessions already run)
Each session's HighLevel job waits for its first-level jobs (SLURM afterok).
"""
import re
import sys
from pathlib import Path

import pandas as pd

S        = Path('/user_data/csimmon2/sym_pt')
REPO     = Path('/user_data/csimmon2/git_repos/sym_pt')
OUT      = REPO / 'sandbox' / 'rerun'
TREE     = 'loc_rerun'
RECORD   = [REPO / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv',
            REPO / 'D_liu' / 'rsa_v1_harmonized.csv']
SUB_INFO = REPO / 'sub_info.csv'
CORRECTED_REG = {('sub-008', 2): 'reg.longitudinal'}   # deviation log v6, sections 0 and 13
TEST_SESSION  = ('sub-008', 2)                          # exercises corrected reg + links
EVS = ['Face', 'House', 'Object', 'Words', 'Scramble']
FSL_ENV = ('export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; '
           'export PATH=$FSLDIR/bin:$PATH')
FSL_ENV_WRAP = FSL_ENV.replace('$', '\\$')     # escaped: expands inside the job, not at submission
SBATCH = '--nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=1-00:00:00'


# ── fsf helpers ──────────────────────────────────────────────────────────────
def get(text, key):
    m = re.findall(r'^set fmri\(' + re.escape(key) + r'\) (.*)$', text, re.M)
    return m[-1].strip().strip('"') if m else None


def put(text, key, value):
    """Replace every assignment of fmri(key); the key must already exist."""
    pat = re.compile(r'^(set fmri\(' + re.escape(key) + r'\) ).*$', re.M)
    new, n = pat.subn(lambda m: m.group(1) + str(value), text)
    if n == 0:
        raise ValueError(f'fmri({key}) not found')
    return new


def feat_files(text):
    return {int(i): p for i, p in re.findall(r'^set feat_files\((\d+)\) "([^"]*)"', text, re.M)}


def first_level(text, outdir):
    """The five changes to one run's design. Everything else stays as FEAT ran it."""
    checks = {'level': '1', 'evs_orig': '5', 'evs_real': '10', 'ncon_real': '19',
              'ncon_orig': '19', 'smooth': '4.0', 'nftests_real': '0'}
    for k, v in checks.items():
        if get(text, k) != v:
            raise ValueError(f'fmri({k}) is {get(text, k)!r}, expected {v!r}')
    for i, ev in enumerate(EVS, 1):
        if get(text, f'evtitle{i}') != ev or get(text, f'deriv_yn{i}') != '1':
            raise ValueError(f'EV {i} is not {ev} with a temporal derivative')
    t = put(text, 'outputdir', f'"{outdir}"')
    t = put(t, 'smooth', '0')
    t = put(t, 'melodic_yn', '0')
    t = put(t, 'reghighres_yn', '0')
    t = put(t, 'regstandard_yn', '0')
    t = put(t, 'ncon_real', '20')
    t = put(t, 'ncon_orig', '20')
    real = {1: -1.0, 7: 1.0}                      # columns: Face 1, Face' 2, ..., Words 7
    orig = {1: -1.0, 4: 1.0}                      # EVs: Face 1, ..., Words 4
    add = ['', '# Contrast 20, added for the rerun: Words - Face',
           'set fmri(conpic_real.20) 1', 'set fmri(conname_real.20) "Word-Face"']
    add += [f'set fmri(con_real20.{k}) {real.get(k, 0)}' for k in range(1, 11)]
    add += ['set fmri(conpic_orig.20) 1', 'set fmri(conname_orig.20) "Word-Face"']
    add += [f'set fmri(con_orig20.{k}) {orig.get(k, 0)}' for k in range(1, 6)]
    add += [f'set fmri(conmask{i}_20) 0' for i in range(1, 20)]
    add += [f'set fmri(conmask20_{j}) 0' for j in range(1, 20)]
    return t.rstrip('\n') + '\n' + '\n'.join(add) + '\n'


def high_level(text, outdir, new_inputs):
    if get(text, 'level') != '2' or get(text, 'ncopeinputs') != '19':
        raise ValueError('not a 19-cope HighLevel design')
    t = put(text, 'outputdir', f'"{outdir}"')
    t = put(t, 'ncopeinputs', '20')
    for i, p in new_inputs.items():
        t, n = re.subn(r'^set feat_files\(' + str(i) + r'\) ".*"$',
                       f'set feat_files({i}) "{p}"', t, flags=re.M)
        if n == 0:
            raise ValueError(f'feat_files({i}) not found')
    return t.rstrip('\n') + '\n\n# cope 20 added for the rerun\nset fmri(copeinput.20) 1\n'


# ── plan ─────────────────────────────────────────────────────────────────────
def sessions_of_record():
    pairs = set()
    for f in RECORD:
        d = pd.read_csv(f, usecols=['subject_id', 'session']).drop_duplicates()
        n = pd.to_numeric(d['session'].astype(str).str.extract(r'(\d+)', expand=False))
        pairs |= set(zip(d['subject_id'], n.astype(int)))
    return sorted(pairs)


def anchors():
    si = pd.read_csv(SUB_INFO)
    si['n'] = si['ses'].astype(str).str.extract(r'(\d+)', expand=False).astype(int)
    return si.groupby('sub')['n'].min().to_dict()      # params.get_sessions(...)[0]


def plan():
    anchor, todo, problems = anchors(), [], []
    for sub, ses in sessions_of_record():
        src = S / sub / f'ses-{ses:02d}' / 'derivatives' / 'fsl' / 'loc'
        dst = src.parent / TREE
        hl = src / 'HighLevel.gfeat' / 'design.fsf'
        try:
            if dst.exists():
                raise ValueError(f'{dst} already exists')
            if not hl.exists():
                raise ValueError(f'missing {hl}')
            hl_text = hl.read_text()
            runs, new_inputs = [], {}
            for i, feat in sorted(feat_files(hl_text).items()):
                feat = Path(feat)
                rname = feat.parent.name                       # run-0N
                design = feat / 'design.fsf'
                regdir = feat / CORRECTED_REG.get((sub, ses), 'reg')
                if not design.exists():
                    raise ValueError(f'missing {design}')
                if not (regdir / 'example_func2standard.mat').exists():
                    raise ValueError(f'missing {regdir}/example_func2standard.mat')
                fsf = dst / rname / '1stLevel.fsf'
                new_text = first_level(design.read_text(), dst / rname / '1stLevel')
                runs.append({'run': rname, 'fsf': fsf, 'text': new_text, 'reg_src': regdir,
                             'old_text': design.read_text()})
                new_inputs[i] = str(dst / rname / '1stLevel.feat')
            if sub not in anchor:
                raise ValueError('not in sub_info.csv')
            todo.append({'sub': sub, 'ses': ses, 'dst': dst, 'runs': runs,
                         'hl_fsf': dst / 'HighLevel.fsf', 'hl_old': hl_text,
                         'hl_text': high_level(hl_text, dst / 'HighLevel', new_inputs),
                         'anchor': anchor[sub], 'links': ses != anchor[sub]})
        except ValueError as e:
            problems.append(f'{sub} ses-{ses:02d}: {e}')
    return todo, problems


def show_diff(old, new, label):
    import difflib
    d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=0)
         if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    print(f'\n  {label}: {len(d)} changed lines')
    for l in d[:12]:
        print('    ' + l)
    if len(d) > 12:
        print(f'    ... {len(d) - 12} more lines: contrast 20 and its conmask entries')


# ── write ────────────────────────────────────────────────────────────────────
def session_script(t):
    D = t['dst']
    lines = ['#!/bin/bash', '# HighLevel step for one session; runs after its first levels.',
             'set -euo pipefail', FSL_ENV, f'D={D}', '']
    lines.append('# 1. copy the record registration into each new first level (no overwrite)')
    for r in t['runs']:
        f = f'$D/{r["run"]}/1stLevel.feat'
        lines += [f'[ -f {f}/stats/cope20.nii.gz ] || {{ echo "first level incomplete: {r["run"]}"; exit 1; }}',
                  f'if [ -e {f}/reg ]; then rmdir {f}/reg || {{ echo "reg exists: {r["run"]}"; exit 1; }}; fi',
                  f'cp -a {r["reg_src"]} {f}/reg']
    lines += ['', '# 2. HighLevel (20 copes)',
              '[ ! -e $D/HighLevel.gfeat ] || { echo "HighLevel.gfeat exists"; exit 1; }',
              'feat $D/HighLevel.fsf', '']
    if t['links']:
        a = f'{t["anchor"]:02d}'
        lines += [f'# 3. later session: the extractors read *_ses{a} files; link them (no overwrite)',
                  'for c in $D/HighLevel.gfeat/cope*.feat/stats; do',
                  f'  ln -s zstat1.nii.gz $c/zstat1_ses{a}.nii.gz',
                  f'  ln -s cope1.nii.gz $c/cope1_ses{a}.nii.gz',
                  'done', '']
    lines += ['touch $D/RERUN_DONE', f'echo "{t["sub"]} ses-{t["ses"]:02d} done"']
    return '\n'.join(lines) + '\n'


def submit_script(todo, name, logs):
    L = ['#!/bin/bash', f'# {name}: submits first levels, then each session\'s HighLevel step',
         '# after all its runs finish. Skips any session whose loc_rerun outputs exist.',
         'set -uo pipefail', f'L={logs}', 'mkdir -p $L', 'n=0', '']
    for t in todo:
        tag = f'{t["sub"][4:]}_{t["ses"]:02d}'
        L.append(f'# {t["sub"]} ses-{t["ses"]:02d}')
        L.append(f'if [ -e {t["dst"]}/HighLevel.gfeat ] || ls -d {t["dst"]}/run-*/1stLevel*.feat >/dev/null 2>&1; then')
        L.append(f'  echo "skip {t["sub"]} ses-{t["ses"]:02d}: outputs exist"')
        L.append('else')
        L.append('  ids=""')
        for r in t['runs']:
            L.append(f'  j=$(sbatch --parsable --job-name=rr1_{tag} --output=$L/rr1_{tag}_%j.out '
                     f'--error=$L/rr1_{tag}_%j.err {SBATCH} --wrap="{FSL_ENV_WRAP}; feat {r["fsf"]}") '
                     f'|| exit 1; ids="$ids:$j"')
        L.append(f'  sbatch --parsable --dependency=afterok$ids --job-name=rr2_{tag} '
                 f'--output=$L/rr2_{tag}_%j.out --error=$L/rr2_{tag}_%j.err {SBATCH} '
                 f'{OUT / "jobs" / (tag + "_session.sh")} || exit 1')
        L.append('  n=$((n+1))')
        L.append('fi')
    L += ['', 'echo "$n sessions submitted"']
    return '\n'.join(L) + '\n'


def write(todo):
    targets = [t['hl_fsf'] for t in todo] + [r['fsf'] for t in todo for r in t['runs']]
    jobs = OUT / 'jobs'
    clash = [p for p in targets if p.exists()] + [p for p in [jobs] if p.exists()]
    if clash:
        sys.exit(f'STOP, would overwrite: {clash[:5]}')
    for t in todo:
        for r in t['runs']:
            r['fsf'].parent.mkdir(parents=True)          # fails if it exists
            r['fsf'].write_text(r['text'])
        t['hl_fsf'].write_text(t['hl_text'])
    jobs.mkdir(parents=True)
    for t in todo:
        p = jobs / f'{t["sub"][4:]}_{t["ses"]:02d}_session.sh'
        p.write_text(session_script(t))
        p.chmod(0o755)
    logs = OUT / 'logs'
    test = [t for t in todo if (t['sub'], t['ses']) == TEST_SESSION]
    (OUT / 'submit_test.sh').write_text(submit_script(test, 'submit_test.sh', logs))
    (OUT / 'submit_all.sh').write_text(submit_script(todo, 'submit_all.sh', logs))
    rows = [{'sub': t['sub'], 'ses': t['ses'], 'run': r['run'], 'fsf': r['fsf'],
             'reg_source': r['reg_src'], 'anchor': t['anchor'], 'links': t['links']}
            for t in todo for r in t['runs']]
    pd.DataFrame(rows).to_csv(OUT / 'manifest.csv', index=False)
    print(f'wrote {len(targets)} .fsf files, {len(todo)} session scripts, submit_test.sh, '
          f'submit_all.sh and manifest.csv under {OUT}')


if __name__ == '__main__':
    todo, problems = plan()
    nrun = sum(len(t['runs']) for t in todo)
    print(f'sessions ready: {len(todo)}   runs: {nrun}   problems: {len(problems)}')
    for p in problems:
        print('  PROBLEM', p)
    print(f'later sessions (get links): {sum(t["links"] for t in todo)}   '
          f'corrected registration: {[f"{s} ses-{n:02d}" for s, n in CORRECTED_REG]}')
    if todo:
        t = next((x for x in todo if (x['sub'], x['ses']) == TEST_SESSION), todo[0])
        show_diff(t['runs'][0]['old_text'], t['runs'][0]['text'],
                  f'{t["sub"]} ses-{t["ses"]:02d} {t["runs"][0]["run"]} first level')
        show_diff(t['hl_old'], t['hl_text'], f'{t["sub"]} ses-{t["ses"]:02d} HighLevel')
    if len(sys.argv) > 1 and sys.argv[1] == 'write':
        if problems:
            sys.exit('STOP: fix the problems above first; nothing written')
        write(todo)
    else:
        print('\nplan only; nothing written. Run with "write" to create the files.')
