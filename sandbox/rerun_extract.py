#!/usr/bin/env python3
"""
rerun_extract.py - extraction stage for the rerun, using the record's own scripts.

The record's extractors, step 13 and the whole-OTC script run UNCHANGED. Only
their input and output locations (and, for one variant, the word-ROI cope) are
replaced from outside, the same way the sandbox runs 05_stats_harmony.py.

  links  Mirror links inside <root>/sym_pt_rerun so the scripts find everything
         where they expect it (new root only; the record is not touched):
           sub/ses/derivatives/fsl/loc  -> loc_rerun       (each rerun session)
           sub/ses-<anchor>/ROIs, anat  -> the record's    (search masks, anatomy)
  write  Job scripts and submit_extract.sh in sandbox/rerun_extract/.
  job    Runs one job (called by the job scripts).

Jobs, all on the same sessions (the rerun sessions):
  mni       step 13 (13_register2mni.py register_one), copes 6-9, every whole-OTC
            session; writes *_mni files inside the new HighLevels. Array, 20 at a time.
  uni_rec13 / rsa_rec13   01 / 04 on the RECORD tree, word ROIs on -cope13: a check;
                          must reproduce D_liu/univariate_v1.csv / rsa_v1.csv rows
  uni_new13 / rsa_new13   01 / 04 on the new tree, word ROIs on -cope13
  uni_new20 / rsa_new20   01 / 04 on the new tree, word ROIs on cope 20 (Words-Face)
  wotc_rec  whole OTC on the RECORD tree (a check; must reproduce the record CSV)
  wotc_new  whole OTC on the new tree (after mni)
Extraction subject set: params.skip_subs = ['017'], the set the record's D_liu
files were extracted with (they contain sub-027, 084, 091, 095, 096). Whole OTC
keeps the current params, the set its record file has. Every output goes to
sandbox/rerun_extract/<job>/; the record's group_results are never written.

Usage (fmri env), from /user_data/csimmon2/git_repos/sym_pt/sandbox:
    python rerun_extract.py --root /lab_data/behrmannlab/claire links
    python rerun_extract.py --root /lab_data/behrmannlab/claire write
    bash rerun_extract/submit_extract.sh
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True    # importing the record's scripts must not write __pycache__ into the repo

import pandas as pd

S     = Path('/user_data/csimmon2/sym_pt')
REPO  = Path('/user_data/csimmon2/git_repos/sym_pt')
OUTX  = REPO / 'sandbox' / 'rerun_extract'
RERUN = REPO / 'sandbox' / 'rerun'
FSL_ENV = ('export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; '
           'export PATH=$FSLDIR/bin:$PATH')
COMMON = '--nodes=1 --ntasks=1 --cpus-per-task=1'
EXTRACTORS = {'uni': REPO / 'D_liu' / 'verified' / '01_univariate_analyses.py',
              'rsa': REPO / 'D_liu' / 'verified' / '04_multivariate_analyses.py'}
STEP13 = REPO / 'pipeline' / '1_preprocessing' / '13_register2mni.py'
WOTC   = REPO / 'H_RosenkeRSM' / 'otc_rsm_rosenke.py'
RECORD_WOTC = REPO / 'H_RosenkeRSM' / 'otc_rsm_persubject.csv'
JOBS = {   # name: (memory, time)
    'uni_rec13': ('8G', '04:00:00'), 'uni_new13': ('8G', '04:00:00'), 'uni_new20': ('8G', '04:00:00'),
    'rsa_rec13': ('8G', '06:00:00'), 'rsa_new13': ('8G', '06:00:00'), 'rsa_new20': ('8G', '06:00:00'),
    'wotc_rec': ('8G', '03:00:00'), 'wotc_new': ('8G', '03:00:00')}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rerun_sessions(NEW):
    """(sub, ses) for every finished rerun session."""
    out = []
    for done in sorted(NEW.glob('sub-*/ses-*/derivatives/fsl/loc_rerun/RERUN_DONE')):
        out.append((done.parts[-6], int(done.parts[-5][4:])))
    return out


def anchors():
    si = pd.read_csv(REPO / 'sub_info.csv')
    si['n'] = si['ses'].astype(str).str.extract(r'(\d+)', expand=False).astype(int)
    return si.groupby('sub')['n'].min().to_dict()


def wotc_sessions():
    sys.path.insert(0, str(REPO))
    from params import get_post_sessions
    subs = sorted(pd.read_csv(RECORD_WOTC)['subject_id'].unique())
    return [(s, int(get_post_sessions(s[4:])[0])) for s in subs]


# ── links ────────────────────────────────────────────────────────────────────
def make_links(NEW):
    sess, anc = rerun_sessions(NEW), anchors()
    plan, problems = [], []
    for sub, ses in sess:
        plan.append((NEW / sub / f'ses-{ses:02d}' / 'derivatives' / 'fsl' / 'loc', Path('loc_rerun')))
    for sub in sorted({s for s, _ in sess}):
        a = anc.get(sub)
        if a is None:
            problems.append(f'{sub}: not in sub_info.csv'); continue
        for part in ['ROIs', 'anat']:
            tgt = S / sub / f'ses-{a:02d}' / part
            if not tgt.is_dir():
                problems.append(f'missing {tgt}')
            plan.append((NEW / sub / f'ses-{a:02d}' / part, tgt))
    made = kept = 0
    for link, tgt in plan:
        if link.is_symlink() and Path(os.readlink(link)) == tgt:
            kept += 1
        elif link.exists() or link.is_symlink():
            problems.append(f'exists and is not the expected link: {link}')
    if problems:
        sys.exit('STOP, nothing linked:\n  ' + '\n  '.join(problems[:20]))
    for link, tgt in plan:
        if not link.is_symlink():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(tgt)
            made += 1
    print(f'{len(sess)} rerun sessions; links made: {made}, already in place: {kept}')


# ── jobs ─────────────────────────────────────────────────────────────────────
def keep_sessions(mod, wanted):
    """Run the extractor's own process_session only for the rerun sessions."""
    orig = mod.process_session

    def wrapped(sid, info, session):
        return orig(sid, info, session) if (sid, int(session)) in wanted else []
    mod.process_session = wrapped


def run_job(name, NEW, task=None):
    sys.path.insert(0, str(REPO))
    import params                                  # the repo's params, shared by every script below
    wanted = set(rerun_sessions(NEW))
    out = OUTX / name
    if name == 'mni':
        sub, ses = (OUTX / 'mni_sessions.txt').read_text().split('\n')[int(task) - 1].split()
        m = load(STEP13, 'step13')
        m.processed_dir, m.COPES = str(NEW), [6, 7, 8, 9]
        d, s, f = m.register_one(sub, ses)
        print(f'{sub} ses-{ses}: {d} registered, {s} already existed, {f} failed')
        sys.exit(1 if f else 0)
    out.mkdir(parents=True, exist_ok=True)
    if name.startswith(('uni', 'rsa')):
        params.skip_subs = ['017']                 # the record extraction set
        m = load(EXTRACTORS[name[:3]], name)
        m.BASE_DIR = S if '_rec' in name else NEW
        m.OUTPUT_DIR = out
        table = m.CONTRASTS if name.startswith('uni') else m.ROIS
        if name.endswith('20'):
            for k in table:
                if k.startswith('word_'):
                    table[k] = (20, False)         # Words - Face
        keep_sessions(m, wanted)
        print(f'{name}: BASE_DIR {m.BASE_DIR}; word ROIs {table["word_VWFA"]}; {len(wanted)} sessions')
        m.main()
    elif name.startswith('wotc'):
        m = load(WOTC, name)
        root = S if name == 'wotc_rec' else NEW
        m.PROC, m.OUT = root, out / 'otc_rsm'
        m.v.processed_dir = str(root)
        sys.argv = [str(WOTC), '--csv', str(out / 'otc_rsm.csv')]
        print(f'{name}: PROC {m.PROC}; masks and CSVs to {out}')
        m.main()


# ── write ────────────────────────────────────────────────────────────────────
def write(NEW):
    if OUTX.exists():
        sys.exit(f'STOP, would overwrite: {OUTX} exists')
    wots = wotc_sessions()
    have = set(rerun_sessions(NEW))
    missing = [w for w in wots if w not in have]
    if missing:
        sys.exit(f'STOP: whole-OTC sessions not rerun yet: {missing}')
    OUTX.mkdir(parents=True)
    (OUTX / 'mni_sessions.txt').write_text(''.join(f'{s} {n:02d}\n' for s, n in wots))
    me = Path(__file__).resolve()
    L = OUTX / 'logs'
    def job(name, task=''):
        return (f'#!/bin/bash\n{FSL_ENV}\nset -euo pipefail\n'
                f'{sys.executable} {me} --root {NEW.parent} job {name} {task}\n')
    (OUTX / 'mni_task.sh').write_text(job('mni', '$SLURM_ARRAY_TASK_ID'))
    for n in JOBS:
        (OUTX / f'{n}.sh').write_text(job(n))
    sub = ['#!/bin/bash', '# Extraction stage for the rerun. Outputs: sandbox/rerun_extract/<job>/', 'set -uo pipefail',
           f'L={L}; mkdir -p $L',
           'if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qE "^rx_"; then echo "STOP: rx_ jobs already queued"; exit 1; fi',
           f'M=$(sbatch --parsable --array=1-{len(wots)}%20 --job-name=rx_mni {COMMON} --mem=2G --time=00:30:00 '
           f'--output=$L/rx_mni_%A_%a.out --error=$L/rx_mni_%A_%a.err {OUTX}/mni_task.sh) || exit 1; M=${{M%%;*}}']
    for n, (mem, t) in JOBS.items():
        dep = '--dependency=afterok:$M ' if n == 'wotc_new' else ''
        sub.append(f'sbatch --parsable {dep}--job-name=rx_{n} {COMMON} --mem={mem} --time={t} '
                   f'--output=$L/rx_{n}_%j.out --error=$L/rx_{n}_%j.err {OUTX}/{n}.sh || exit 1')
    sub.append(f'echo "submitted: MNI array ({len(wots)} sessions, 20 at a time), {len(JOBS)} jobs; '
               f'wotc_new waits for the MNI array"')
    (OUTX / 'submit_extract.sh').write_text('\n'.join(sub) + '\n')
    for p in OUTX.glob('*.sh'):
        p.chmod(0o755)
    print(f'wrote {len(JOBS) + 2} scripts and mni_sessions.txt ({len(wots)} sessions) in {OUTX}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('step', choices=['links', 'write', 'job'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('task', nargs='?')
    a = ap.parse_args()
    NEW = Path(a.root) / 'sym_pt_rerun'
    if not NEW.is_dir():
        sys.exit(f'STOP: {NEW} not found')
    {'links': lambda: make_links(NEW), 'write': lambda: write(NEW),
     'job': lambda: run_job(a.name, NEW, a.task)}[a.step]()
