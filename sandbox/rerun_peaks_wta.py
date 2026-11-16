#!/usr/bin/env python3
"""
rerun_peaks_wta.py - peak location and WTA for each processing version, then the
stats script with those inputs, so Table 1's peak-location and WTA cells exist
for every version. TFCE is not rerun: its cluster table stays the record's.

  python rerun_peaks_wta.py plan     checks inputs; lists the MNI copies WTA still needs
  python rerun_peaks_wta.py submit   one job running every stage below in order
  python rerun_peaks_wta.py collect  peak-location and WTA rows side by side

Stages (what the job runs; FSL needed):
  mni    cope 6-9 MNI copies the rerun lacks, made as step 13 does
         (13_register2mni.py: flirt -applyxfm with the first-session
         anat2stand.mat, trilinear), into the rerun's own HighLevel folders
  wta    D_liu/verified/03_wta_analyses.py unchanged, once on the record's maps
         (check: equals the record's wta_percentages.csv) and once on the
         rerun's (no smoothing; copes 6-9 are the same in P1 and P2)
  peaks  D_liu/calc_peak_coords_mni.py unchanged, on each version's extraction
         CSV (the record's extraction CSV is the check against the record's
         peak_coords_mni.csv)
  stats  D_liu/verified/05_stats_harmony.py unchanged: P0 x A0 (check: its peak
         and WTA rows equal the record stats file), P0 x A1, P1 x A1, P2 x A1
Writes into sandbox/rerun_more/ (and the MNI copies into /user_data/csimmon2/sym_pt_rerun).
"""
import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/user_data/csimmon2/git_repos/sym_pt')
sys.path.insert(0, str(REPO))
from params import processed_dir, get_sessions            # noqa: E402

SB = REPO / 'sandbox'
OUT = SB / 'rerun_more'
REC = Path(processed_dir)
NEW = Path('/user_data/csimmon2/sym_pt_rerun')
MNI_BRAIN = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
WTA_SCRIPT = REPO / 'D_liu' / 'verified' / '03_wta_analyses.py'
PEAK_SCRIPT = REPO / 'D_liu' / 'calc_peak_coords_mni.py'
STATS_SCRIPT = REPO / 'D_liu' / 'verified' / '05_stats_harmony.py'
REC_WTA = REC / 'group_results' / 'wta_percentages.csv'
REC_PEAK = REC / 'group_results' / 'peak_coords' / 'peak_coords_mni.csv'
REC_STATS = REPO / 'D_liu' / 'stats_results_harmonized_corrected_ptpt4.csv'
HZ, XT = SB / 'rerun_harmonize', SB / 'rerun_extract'
VERSIONS = {   # harmonized univariate, harmonized RSA, extraction CSV for peaks
    'P0_record':       (REPO / 'D_liu' / 'univariate_v1_harmonized_sqrt.csv', REPO / 'D_liu' / 'rsa_v1_harmonized.csv',
                        REPO / 'D_liu' / 'univariate_v1.csv'),
    'P1_nosmooth':     (HZ / 'new13' / 'univariate_v1_harmonized_sqrt.csv', HZ / 'new13' / 'rsa_v1_harmonized.csv',
                        XT / 'uni_new13' / 'univariate_v1.csv'),
    'P2_nosmooth_c20': (HZ / 'new20' / 'univariate_v1_harmonized_sqrt.csv', HZ / 'new20' / 'rsa_v1_harmonized.csv',
                        XT / 'uni_new20' / 'univariate_v1.csv'),
}
EXCLUDE = ['sub-017', 'sub-091', 'sub-095', 'sub-096', 'sub-027', 'sub-084']
COHORTS = {'A0_record': EXCLUDE, 'A1_no010': EXCLUDE + ['sub-010']}
RUNS = [('P0_record', 'A0_record'), ('P0_record', 'A1_no010'),
        ('P1_nosmooth', 'A1_no010'), ('P2_nosmooth_c20', 'A1_no010')]
KEYS_PEAK = ['subject_id', 'session', 'hemi', 'category']
KEYS_WTA = ['subject_id', 'session', 'hemi', 'region', 'denominator', 'category']


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mni_targets():
    todo = []
    for hl in sorted(NEW.glob('sub-*/ses-*/derivatives/fsl/loc_rerun/HighLevel.gfeat')):
        sub, ses = hl.parts[-6], hl.parts[-5]
        first = f'{get_sessions(sub.replace("sub-", ""))[0]:02d}'
        a2s = REC / sub / f'ses-{first}' / 'anat' / 'anat2stand.mat'
        for c in (6, 7, 8, 9):
            st = hl / f'cope{c}.feat' / 'stats'
            for prefix in ('zstat', 'cope'):
                src, dst = st / f'{prefix}1.nii.gz', st / f'{prefix}1_mni.nii.gz'
                if src.exists() and not dst.exists():
                    todo.append((sub, ses, src, dst, a2s))
    return todo


def plan():
    need = [WTA_SCRIPT, PEAK_SCRIPT, STATS_SCRIPT, REC_WTA, REC_PEAK, REC_STATS, NEW] + \
           [p for v in VERSIONS.values() for p in v]
    miss = [str(p) for p in need if not p.exists()]
    print('inputs:', 'all present' if not miss else 'MISSING:\n  ' + '\n  '.join(miss))
    todo = mni_targets()
    sess = sorted({f'{s} {x}' for s, x, *_ in todo})
    print(f'MNI copies still needed for WTA: {len(todo)} files in {len(sess)} sessions: {", ".join(sess) or "none"}')
    noa = sorted({str(a) for *_, a in todo if not a.exists()})
    if noa:
        print('  MISSING anat2stand.mat:', ', '.join(noa))
    return not miss and not noa


def stage_mni():
    todo = mni_targets()
    for sub, ses, src, dst, a2s in todo:
        subprocess.run(['flirt', '-in', str(src), '-ref', MNI_BRAIN, '-out', str(dst),
                        '-applyxfm', '-init', str(a2s), '-interp', 'trilinear'], check=True)
    print(f'mni: made {len(todo)} files; still missing: {len(mni_targets())}')


def run_wta(tree, out_csv):
    mod = _load(WTA_SCRIPT, f'wta_{tree.name}')
    mod.processed_dir = str(tree)                                       # where the maps are read
    mod.TFCE_DIR = REC / 'group_results' / 'tfce_votc_harmonized'       # record masks, unchanged
    mod.OUTPUT_CSV = out_csv
    with open(OUT / 'logs' / f'{out_csv.stem}.log', 'w') as log, contextlib.redirect_stdout(log):
        mod.main()


def _compare(a, b, keys, cols, label):
    m = a.merge(b, on=keys, suffixes=('_a', '_b'))
    d = max(float((m[c + '_a'] - m[c + '_b']).abs().max()) for c in cols) if len(m) else float('nan')
    print(f'{label}: {len(m)} rows matched ({len(a)} vs {len(b)}); largest difference {d:.3g}')
    return m, d


def stage_wta():
    run_wta(REC, OUT / 'wta_record_check.csv')
    run_wta(NEW, OUT / 'wta_nosmooth.csv')
    rec, chk, new = (pd.read_csv(p, dtype={'session': str}) for p in (REC_WTA, OUT / 'wta_record_check.csv', OUT / 'wta_nosmooth.csv'))
    _, d = _compare(chk, rec, KEYS_WTA, ['wta_pct'], 'wta record check vs the record file')
    print('  ->', 'PASS' if d == 0 else 'CHECK: the record run does not reproduce the record file')
    m, d = _compare(new, rec, KEYS_WTA, ['wta_pct'], 'wta no smoothing vs record')
    print('  ->', 'the rerun maps were read (values differ)' if d > 0 else 'WARNING: identical to the record; the rerun maps were not read')


def run_peaks(in_csv, out_csv):
    mod = _load(PEAK_SCRIPT, f'peaks_{out_csv.stem}')
    mod.INPUT_CSV, mod.OUTPUT_CSV = in_csv, out_csv                     # BASE_DIR stays the record (anatomy)
    with open(OUT / 'logs' / f'{out_csv.stem}.log', 'w') as log, contextlib.redirect_stdout(log):
        mod.main()


def stage_peaks():
    for v, (_, _, ext) in VERSIONS.items():
        run_peaks(ext, OUT / f'peaks_{v}.csv')
    rec = pd.read_csv(REC_PEAK)
    p0 = pd.read_csv(OUT / 'peaks_P0_record.csv')
    roi = ['word_VWFA', 'face_FFA', 'house_PPA', 'house_PPA_strict', 'object_LOC']
    _, d = _compare(p0[p0.category.isin(roi)], rec[rec.category.isin(roi)], KEYS_PEAK,
                    ['peak_x_mni', 'peak_y_mni', 'peak_z_mni'], 'peaks record check vs the record file')
    print('  ->', 'PASS' if d < 1e-6 else 'CHECK: differs from the record file')
    p1 = pd.read_csv(OUT / 'peaks_P1_nosmooth.csv')
    _, d = _compare(p1, p0, KEYS_PEAK, ['peak_x_mni', 'peak_y_mni'], 'peaks no smoothing vs record')


def stage_stats():
    for v, c in RUNS:
        uni, rsa, _ = VERSIONS[v]
        mod = _load(STATS_SCRIPT, f'stats_{v}_{c}')                     # fresh module, fresh RNG
        mod.UNIVAR_CSV, mod.RSA_CSV = uni, rsa
        mod.PEAK_MNI = REC_PEAK if v == 'P0_record' else OUT / f'peaks_{v}.csv'
        mod.WTA_CSV = REC_WTA if v == 'P0_record' else OUT / 'wta_nosmooth.csv'
        mod.EXCLUDE, mod.EXCLUDE_SES = list(COHORTS[c]), [('sub-108', 2)]
        mod.OUT_RESULTS, mod.OUT_TFCE = OUT / f'stats_{v}__{c}.csv', OUT / f'tfce_{v}__{c}.csv'
        argv, sys.argv = sys.argv, [str(STATS_SCRIPT)]
        try:
            with open(OUT / 'logs' / f'stats_{v}__{c}.log', 'w') as log, contextlib.redirect_stdout(log):
                mod.main()
        finally:
            sys.argv = argv
        print(f'stats {v} x {c}: done')


def _rows(path):
    s = pd.read_csv(path)
    s = s[s['measure'].isin(['peak_distance', 'wta', 'wta_selective_prop'])].copy()
    for k in ['measure', 'model', 'comparison', 'hemi', 'level']:
        s[k] = s[k].fillna('')                     # blank keys would drop out of the table
    return s


def collect():
    keys = ['measure', 'model', 'comparison', 'hemi', 'level']
    have = {(v, c): OUT / f'stats_{v}__{c}.csv' for v, c in RUNS}
    have = {k: _rows(p) for k, p in have.items() if p.exists()}
    print(f'{len(have)} of {len(RUNS)} stats runs present')
    if ('P0_record', 'A0_record') in have and REC_STATS.exists():
        rec = _rows(REC_STATS)
        num = [c for c in ['diff', 'cohens_d', 'p_perm', 'q_fdr', 'omnibus_chi2', 'omnibus_p'] if c in rec.columns]
        m = have[('P0_record', 'A0_record')].merge(rec, on=keys, suffixes=('_n', '_r'))
        d = max(float((m[c + '_n'] - m[c + '_r']).abs().max()) for c in num) if len(m) else float('nan')
        print(f'record check: {len(m)} of {len(rec)} peak/WTA rows matched; largest difference {d:.3g} ->',
              'PASS' if d < 1e-9 else 'CHECK')
    rows = []
    for (v, c), s in have.items():
        for _, r in s.iterrows():
            if r['comparison'] == 'omnibus':
                cell = f"omnibus p={r['omnibus_p']:.3f}"
            else:
                d = '' if pd.isna(r.get('cohens_d')) else f"d={r['cohens_d']:+.2f} "
                cell = d + (f"q={r['q_fdr']:.3f}" if pd.notna(r.get('q_fdr')) else f"p={r['p_perm']:.3f}")
            rows.append({**{k: r[k] for k in keys}, 'version': f'{v} | {c}', 'cell': cell})
    T = pd.DataFrame(rows)
    if T.empty:
        print('no peak or WTA rows yet')
        return
    T.to_csv(OUT / 'peaks_wta_by_version.csv', index=False)
    x = T.pivot_table(index=keys, columns='version', values='cell', aggfunc='first')
    with pd.option_context('display.width', 260, 'display.max_colwidth', 24, 'display.max_rows', 200):
        print(x.to_string())
    print(f'\nsaved: {OUT / "peaks_wta_by_version.csv"}  (TFCE rows not rerun: record values)')


def submit():
    if not plan():
        sys.exit('STOP: something is missing')
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    sh = OUT / 'pkwta_job.sh'
    sh.write_text(f"""#!/bin/bash
#SBATCH --job-name=pkwta
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output={OUT}/logs/pkwta.out
#SBATCH --error={OUT}/logs/pkwta.err
source /home/csimmon2/anaconda3/etc/profile.d/conda.sh && conda activate fmri
export FSLDIR=/opt/fsl/6.0.3; . $FSLDIR/etc/fslconf/fsl.sh; export PATH=$FSLDIR/bin:$PATH
python {Path(__file__).resolve()} all
""")
    r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


def run_all():
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    for name, f in [('mni', stage_mni), ('wta', stage_wta), ('peaks', stage_peaks), ('stats', stage_stats)]:
        print(f'== {name}', flush=True)
        f()
    collect()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'plan'
    {'plan': plan, 'submit': submit, 'collect': collect, 'all': run_all,
     'mni': stage_mni, 'wta': stage_wta, 'peaks': stage_peaks, 'stats': stage_stats}[cmd]()
