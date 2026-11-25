#!/usr/bin/env python3
"""
extract_suploc_2026-11-24.py - whole OTC on the superior-LOC-only mask, run exactly like the two
Nov 18 whole-OTC jobs. It imports sandbox/extract_ventral.py unchanged and adds one entry to its
MASKS table, so the same paper tree (/user_data/csimmon2/sym_pt_rerun), the same record script
(H_RosenkeRSM/otc_rsm_rosenke.py) and the same mask swap are used.

  python extract_suploc_2026-11-24.py write     job script + a submit script for this job only
  bash   /user_data/csimmon2/git_repos/sym_pt/sandbox/ventral_otc/jobs/submit_suploc.sh
  (python extract_suploc_2026-11-24.py job      is what the job script runs)

Reads  sandbox/ventral_otc/suploc_{l,r}_mask.nii.gz   (from build_otc_suploc_2026-11-24.py)
Writes sandbox/ventral_otc/wotc_suploc/ (refuses if it already holds output),
       jobs/wotc_suploc.sh, jobs/submit_suploc.sh, logs/vx_wotc_suploc_*.
Does not write or resubmit any of the other jobs.
"""
import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(os.environ.get('SYMPT_GIT', '/user_data/csimmon2/git_repos/sym_pt'))
NAME, MASK = 'wotc_suploc', 'suploc'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


EV = load(REPO / 'sandbox' / 'extract_ventral.py', 'extract_ventral')
EV.MASKS[NAME] = MASK                      # the only addition; everything else is extract_ventral's


def job():
    out = EV.OUT / NAME
    if out.exists() and any(out.iterdir()):
        sys.exit(f'STOP: {out} already holds output; refusing to overwrite')
    EV.run_job(NAME)


def write():
    for h in 'lr':
        p = EV.OUT / f'{MASK}_{h}_mask.nii.gz'
        if not p.exists():
            sys.exit(f'STOP: {p} missing; run build_otc_suploc_2026-11-24.py first')
    J, L = EV.OUT / 'jobs', EV.OUT / 'logs'
    J.mkdir(parents=True, exist_ok=True); L.mkdir(parents=True, exist_ok=True)
    js, sub = J / f'{NAME}.sh', J / 'submit_suploc.sh'
    js.write_text(f'#!/bin/bash\n{EV.RX.FSL_ENV}\nset -euo pipefail\n{sys.executable} {Path(__file__).resolve()} job\n')
    js.chmod(0o755)
    sub.write_text('#!/bin/bash\nset -uo pipefail\n'
                   f'if squeue -u "${{USER:-$(whoami)}}" -h -o %j | grep -qx "vx_{NAME}"; then echo "STOP: vx_{NAME} already queued"; exit 1; fi\n'
                   f'sbatch --parsable --job-name=vx_{NAME} {EV.RX.COMMON} --mem=8G --time=03:00:00 '
                   f'--output={L}/vx_{NAME}_%j.out --error={L}/vx_{NAME}_%j.err {js} || exit 1\n'
                   f'echo "submitted 1 job; logs in {L}"\n')
    sub.chmod(0o755)
    print(f'wrote {js} and {sub} (python: {sys.executable})')


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else ''
    if step == 'write': write()
    elif step == 'job': job()
    else: sys.exit(__doc__)
