#!/usr/bin/env python3
"""
longitudinal/session_steps.py                                        2026-11-03
Runs the Python Stage-1 steps for ONE session of the longitudinal study.

It loads the cross-sectional pipeline's own functions from 1_preprocessing/
(those files are not changed) and calls them for a single subject/session, so
the processing is identical and nothing outside sub-XXX/ses-NN is written.
The one exception is 04's shared ROI staging ($S/rois), which only copies in
files that are missing; the file guard reports it if that ever happens.

Called by longitudinal/new_session.sh; can also be run directly:
  python longitudinal/session_steps.py check_mat XXX NN   # .mat sanity: blocks, first block vs time0
  python longitudinal/session_steps.py dirs      XXX NN   # 01_organize.setup_subject
  python longitudinal/session_steps.py timing    XXX NN [--root DIR]
                                                          # 03_convert_timing.convert_run, each run
                                                          # (--root: write under DIR instead of
                                                          #  the processed dir; for validation)
  python longitudinal/session_steps.py anat      XXX NN   # 04_register_mirror.stage_rois + process_subject
"""
import glob
import importlib.util
import os
import sys

PIPE = '/user_data/csimmon2/git_repos/sym_pt/pipeline'
sys.path.insert(0, PIPE)
import sym_pt_params as prm  # noqa: E402


def load(fname):
    """Import a 1_preprocessing script as a module (its main() is not run)."""
    path = f'{PIPE}/1_preprocessing/{fname}'
    spec = importlib.util.spec_from_file_location('step_' + fname.split('_')[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_mat(sub, ses):
    import scipy.io
    mats = sorted(glob.glob(f'{prm.raw_dir}/sub-{sub}/ses-{ses:02d}/covs/*loc.mat'))
    if not mats:
        sys.exit(f'ERROR: no *loc.mat in {prm.raw_dir}/sub-{sub}/ses-{ses:02d}/covs')
    bad = 0
    for m in mats:
        d = scipy.io.loadmat(m, variable_names=['block_starts', 'time0', 'stimOrd', 'runNum'])
        bs = d['block_starts'].flatten()
        t0 = float(d['time0'].flatten()[0])
        n_codes = d['stimOrd'].size
        run = str(d['runNum'].flatten()[0]) if 'runNum' in d else '?'
        first = bs[0] - t0
        ok = len(bs) == 15 and n_codes == 15 and abs(first - 8.0) < 0.05
        bad += not ok
        print(f'  {os.path.basename(m)}: runNum={run} blocks={len(bs)} codes={n_codes} '
              f'first block {first:.3f} s after time0{"" if ok else "   <-- expected 15 blocks, 8.0 s"}')
    if bad:
        sys.exit(f'ERROR: {bad} .mat file(s) not as expected')


def dirs(sub, ses):
    load('01_organize.py').setup_subject(sub, [ses])


def timing(sub, ses, root=None):
    m = load('03_convert_timing.py')
    if root:
        m.processed_dir = root          # convert_run writes under m.processed_dir
    runs = prm.get_runs(sub, ses)
    if not runs:
        sys.exit(f'ERROR: no runs for sub-{sub} ses-{ses:02d}')
    print(f'  Session {ses}:')
    for run in runs:
        if m.convert_run(sub, ses, run) != len(prm.conditions):
            sys.exit(f'ERROR: run {run} did not produce all {len(prm.conditions)} conditions')


def anat(sub, ses):
    m = load('04_register_mirror.py')
    # Intact hemisphere does not change across sessions: if this session has no
    # row in sub_info.csv, use the subject's first row instead of '' (which
    # 04 would mirror as if the right hemisphere were intact).
    orig = m.get_sub_info

    def get_sub_info(s, se=None):
        return orig(s, se) or orig(s)

    m.get_sub_info = get_sub_info
    if not orig(sub):
        sys.exit(f'ERROR: sub-{sub} not in sub_info.csv')
    m.stage_rois()
    m.process_subject(sub, ses)


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ('check_mat', 'dirs', 'timing', 'anat'):
        sys.exit(__doc__)
    step = sys.argv[1]
    sub = sys.argv[2].replace('sub-', '')
    ses = int(sys.argv[3].replace('ses-', ''))
    root = None
    if '--root' in sys.argv:
        root = sys.argv[sys.argv.index('--root') + 1]
    if step == 'check_mat':
        check_mat(sub, ses)
    elif step == 'dirs':
        dirs(sub, ses)
    elif step == 'timing':
        timing(sub, ses, root)
    elif step == 'anat':
        anat(sub, ses)


if __name__ == '__main__':
    main()
