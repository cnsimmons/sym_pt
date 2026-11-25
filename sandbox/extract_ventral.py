#!/usr/bin/env python3
"""
extract_ventral.py - re-extract LO and whole OTC with the ventral masks, on the paper's tree
(/user_data/csimmon2/sym_pt_rerun: no smoothing, word ROIs on cope 20).

Set up exactly as sandbox/rerun_extract.py's run_job (its own load / rerun_sessions / keep_sessions,
params.skip_subs = ['017'], BASE_DIR / PROC pointed at the rerun tree). The record scripts run
UNCHANGED; only ROI tables, masks and output locations are set from outside. Writes only into
sandbox/ventral_otc/.

  python extract_ventral.py masks        ventral + inferior temporal gyrus, temporooccipital (HO 15),
                                         same side -> votc_ventral_ito_{l,r}_mask.nii.gz (sensitivity)
  python extract_ventral.py write        job scripts + submit.sh in sandbox/ventral_otc/jobs/
  bash   sandbox/ventral_otc/jobs/submit.sh
  (python extract_ventral.py job <name>  is what each job script runs)

Jobs
  uni_lo            01_univariate_analyses.py on object_LOC (check: must equal uni_new20) and object_LOC_inf
  rsa_lo            04_multivariate_analyses.py, same two ROIs
  wotc_ventral      otc_rsm_rosenke.py with votc_ventral_{l,r}_mask
  wotc_ventral_ito  otc_rsm_rosenke.py with votc_ventral_ito_{l,r}_mask
"""
import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(os.environ.get('SYMPT_GIT', '/user_data/csimmon2/git_repos/sym_pt'))
PROC = Path(os.environ.get('SYMPT_PROC', '/user_data/csimmon2/sym_pt'))
NEW  = Path(os.environ.get('SYMPT_NEW', '/user_data/csimmon2/sym_pt_rerun'))
OUT  = REPO / 'sandbox' / 'ventral_otc'
FSL  = os.environ.get('FSLDIR', '/opt/fsl/6.0.3')
MAXPROB = Path(f'{FSL}/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')
EXTRACTORS = {'uni': REPO / 'D_liu' / 'verified' / '01_univariate_analyses.py',
              'rsa': REPO / 'D_liu' / 'verified' / '04_multivariate_analyses.py'}
WOTC = REPO / 'H_RosenkeRSM' / 'otc_rsm_rosenke.py'
JOBS = {'uni_lo': ('8G', '04:00:00'), 'rsa_lo': ('8G', '06:00:00'),
        'wotc_ventral': ('8G', '03:00:00'), 'wotc_ventral_ito': ('8G', '03:00:00')}
MASKS = {'wotc_ventral': 'votc_ventral', 'wotc_ventral_ito': 'votc_ventral_ito'}
ITG_TO = 15                                   # Inferior Temporal Gyrus, temporooccipital part


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


RX = load(REPO / 'sandbox' / 'rerun_extract.py', 'rerun_extract')   # its helpers, unchanged


def make_ito_masks():
    import numpy as np, nibabel as nib
    mimg = nib.load(str(MAXPROB)); lab_atlas = np.rint(mimg.get_fdata()).astype(int); inv = np.linalg.inv(mimg.affine)
    orig = {h: nib.load(str(PROC / 'group_results' / 'tfce_votc_harmonized' / f'votc_{h}_mask.nii.gz')) for h in 'lr'}
    base = {h: nib.load(str(OUT / f'votc_ventral_{h}_mask.nii.gz')) for h in 'lr'}
    shape, aff = base['l'].get_fdata().shape, base['l'].affine
    ijk = np.indices(shape).reshape(3, -1).T; world = nib.affines.apply_affine(aff, ijk)
    a_ijk = np.rint(nib.affines.apply_affine(inv, world)).astype(int)
    ok = np.all((a_ijk >= 0) & (a_ijk < np.array(lab_atlas.shape)), axis=1)
    lab = np.zeros(len(ijk), int); lab[ok] = lab_atlas[tuple(a_ijk[ok].T)]; lab = lab.reshape(shape)
    x = world[:, 0].reshape(shape); mid = np.abs(x) < 1e-6
    mid_l = int(((orig['l'].get_fdata() > 0) & mid).sum()); mid_r = int(((orig['r'].get_fdata() > 0) & mid).sum())
    side = {'l': (x < 0) | (mid & (mid_l > 0)), 'r': (x > 0) | (mid & (mid_r > 0))}
    for h in 'lr':
        b = base[h].get_fdata() > 0
        add = (lab == ITG_TO + 1) & side[h] & ~b
        p = OUT / f'votc_ventral_ito_{h}_mask.nii.gz'
        nib.save(nib.Nifti1Image((b | add).astype(np.float32), base[h].affine), str(p))
        print(f'{h}: ventral {int(b.sum())} + {int(add.sum())} ITG temporooccipital = {int((b | add).sum())}; wrote {p}')


def run_job(name):
    if not NEW.is_dir():
        sys.exit(f'STOP: {NEW} not found')
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    if name in ('uni_lo', 'rsa_lo'):
        sys.path.insert(0, str(REPO))
        import params
        params.skip_subs = ['017']                               # the record extraction set
        kind = name[:3]
        m = load(EXTRACTORS[kind], name)
        m.BASE_DIR, m.OUTPUT_DIR, m.OUTPUT_NAME = NEW, out, f'{name}.csv'
        table = m.CONTRASTS if kind == 'uni' else m.ROIS
        lo = table['object_LOC']
        for k in [k for k in table if k != 'object_LOC']:
            del table[k]
        table['object_LOC_inf'] = lo                             # same contrast as LO (cope 3)
        if kind == 'uni':
            m.CONTROL_HEMIS = {cat: ['l', 'r'] for cat in table}
        RX.keep_sessions(m, set(RX.rerun_sessions(NEW)))
        print(f'{name}: BASE_DIR {m.BASE_DIR}; ROIs {dict(table)}')
        m.main()
    elif name in MASKS:
        paths = {h: OUT / f'{MASKS[name]}_{h}_mask.nii.gz' for h in 'lr'}
        for p in paths.values():
            if not p.exists():
                sys.exit(f'STOP: {p} missing')
        m = load(WOTC, name)
        m.PROC, m.OUT = NEW, out / 'otc_rsm'
        m.v.processed_dir = str(NEW)
        orig_build = m.v.build_votc_masks_and_save
        def build():                                             # same keys as the script's own masks
            keys = orig_build()
            new = {k: paths['l' if str(k).lower().startswith('l') else 'r'] for k in keys}
            print(f'{name}: masks {new}')
            return new
        m.v.build_votc_masks_and_save = build
        sys.argv = [str(WOTC), '--csv', str(out / 'otc_rsm.csv')]
        m.main()
    else:
        sys.exit(f'unknown job {name}')


def write():
    J, L = OUT / 'jobs', OUT / 'logs'
    J.mkdir(parents=True, exist_ok=True); L.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    sub = ['#!/bin/bash', 'set -uo pipefail',
           'if squeue -u "${USER:-$(whoami)}" -h -o %j | grep -qE "^vx_"; then echo "STOP: vx_ jobs already queued"; exit 1; fi']
    for n, (mem, t) in JOBS.items():
        (J / f'{n}.sh').write_text(f'#!/bin/bash\n{RX.FSL_ENV}\nset -euo pipefail\n{sys.executable} {me} job {n}\n')
        (J / f'{n}.sh').chmod(0o755)
        sub.append(f'sbatch --parsable --job-name=vx_{n} {RX.COMMON} --mem={mem} --time={t} '
                   f'--output={L}/vx_{n}_%j.out --error={L}/vx_{n}_%j.err {J}/{n}.sh || exit 1')
    sub.append(f'echo "submitted {len(JOBS)} jobs; logs in {L}"')
    (J / 'submit.sh').write_text('\n'.join(sub) + '\n'); (J / 'submit.sh').chmod(0o755)
    print(f'wrote {len(JOBS)} job scripts and {J}/submit.sh')


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else ''
    if step == 'masks': make_ito_masks()
    elif step == 'write': write()
    elif step == 'job': run_job(sys.argv[2])
    else: sys.exit(__doc__)
