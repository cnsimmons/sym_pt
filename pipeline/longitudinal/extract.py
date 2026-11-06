#!/usr/bin/env python3
"""
longitudinal/extract.py                                              2026-11-03
Step 4 (extraction) for the longitudinal study. Output: pipeline/outputs_longitudinal/
  univariate_v1.csv  (01_univariate.py)     rsa_v1.csv                    (02_multivariate.py)
  liu_exact_replication_v2.csv (03_liu_recreation.py)   peak_coords_mni.csv (04, from 03's file)

Uses the extraction scripts' own code, unchanged. Only three things are swapped in:
  - output paths: never the cross-sectional outputs;
  - which subject/sessions run ('add' restricts the scripts' loops to the named
    sessions; each session's rows depend only on that session's files);
  - maps: a session with HighLevel.gfeat.longitudinal (sub-008 ses-02, sub-010
    ses-03; found automatically) is read from that folder's real zstat1/cope1
    files. Its *_sesNN links point back into HighLevel.gfeat (step 12 makes
    absolute links; the folder was renamed on 2026-11-02).

Commands (from pipeline/, fmri env, FSL loaded, inside screen):
  python longitudinal/extract.py seed
      Start the folder from what already exists (copies, seconds):
        univariate, rsa : outputs/rereg_20261102/D_liu (extracted 2026-11-02 with
                          the corrected sub-008/010 maps; all sessions as of then)
        liu, peak MNI   : the current cross-sectional files
  python longitudinal/extract.py add XXX NN [XXX NN ...]
      Extract only these sessions (all four files) and put their rows in,
      replacing any old rows. Reports each session as new / identical / changed.
      Minutes per session.
  python longitudinal/extract.py validate
      Read-only: univariate/rsa vs the 2026-11-02 extraction; liu's univariate and
      RSA columns vs univariate/rsa (same code); one MNI peak per liu row.
  python longitudinal/extract.py full [uni rsa liu mni]
      Rebuild from scratch, all subjects and sessions (~2-3 h for all four).
"""
import glob
import importlib.util
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]      # pipeline/, wherever it sits
S = Path('/user_data/csimmon2/sym_pt')          # imaging data (outside the repo)
OUT = PIPE / 'outputs_longitudinal'
sys.path.insert(0, str(PIPE))

STEPS = {   # step: (script in 2_extraction/, output file)
    'uni': ('01_univariate.py', 'univariate_v1.csv'),
    'rsa': ('02_multivariate.py', 'rsa_v1.csv'),
    'liu': ('03_liu_recreation.py', 'liu_exact_replication_v2.csv'),
    'mni': ('04_peak_coords_mni.py', 'peak_coords_mni.csv'),
}
REFS = {    # 'seed' copies these; 'validate' compares against them
    'uni': PIPE / 'outputs/rereg_20261102/D_liu/univariate_v1.csv',
    'rsa': PIPE / 'outputs/rereg_20261102/D_liu/rsa_v1.csv',
    'liu': PIPE / 'liu_exact_replication_v2.csv',
    'mni': S / 'group_results_pipeline/peak_coords/peak_coords_mni.csv',
}
LINK = re.compile(r'/(zstat1|cope1)_ses\d+\.nii\.gz$')


# -- helpers ------------------------------------------------------------------
def longitudinal_dirs():
    """{canonical HighLevel.gfeat: its .longitudinal copy} for every session that has one."""
    found = sorted(glob.glob(f'{S}/sub-*/ses-*/derivatives/fsl/loc/HighLevel.gfeat.longitudinal'))
    return {d[:-len('.longitudinal')]: d for d in found}


def make_redirect(table):
    def redirect(fp):
        s = str(fp)
        for canon, longi in table.items():
            if s.startswith(canon + '/'):
                return LINK.sub(r'/\1.nii.gz', longi + s[len(canon):])
        return s
    return redirect


def load_module(fname):
    spec = importlib.util.spec_from_file_location('x' + fname.split('_')[0],
                                                  PIPE / '2_extraction' / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def session_label(path):
    m = re.search(r'/(sub-[^/]+)/ses-(\d+)/', path)
    return f'{m.group(1)} ses-{m.group(2)}' if m else path


def ses_num(d):
    return pd.to_numeric(d['session'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')


def numeric_cols(a, b):
    return [c for c in a.columns if c in b.columns and c not in ('session', 'ses_num')
            and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]


def same_rows(x, y, num):
    """'identical' or a short reason, comparing two row sets in file order."""
    x, y = x[num].to_numpy(float), y[num].to_numpy(float)
    if x.shape != y.shape:
        return f'{len(x)} vs {len(y)} rows'
    dd = np.abs(x - y)
    dd = dd[~np.isnan(dd)]
    mx = float(dd.max()) if dd.size else 0.0
    if (np.isnan(x) == np.isnan(y)).all() and mx <= 1e-9:
        return 'identical'
    return f'max diff {mx:.3g}'


def run_module(step, table, out_name, targets=None, mni_input=None):
    """Run one extraction script's main() with outputs (and optionally sessions) swapped."""
    fname, _ = STEPS[step]
    m = load_module(fname)
    if step == 'mni':
        m.INPUT_CSV, m.OUTPUT_CSV = mni_input, OUT / out_name
    else:
        m.OUTPUT_DIR, m.OUTPUT_NAME = OUT, out_name
        orig_load, redirect = m._load, make_redirect(table)
        m._load = lambda fp: orig_load(redirect(fp))
        if targets is not None:            # restrict the scripts' own loops
            orig_subjects, orig_session = m.load_subjects, m.process_session
            m.load_subjects = lambda: {k: v for k, v in orig_subjects().items() if k in targets}
            m.process_session = lambda sid, info, ses: (orig_session(sid, info, ses)
                                                        if ses in targets.get(sid, ()) else [])
    argv, sys.argv = sys.argv, [fname]     # 03 parses its own arguments
    t0 = time.time()
    try:
        m.main()
    finally:
        sys.argv = argv
    return time.time() - t0


# -- commands -----------------------------------------------------------------
def seed():
    OUT.mkdir(parents=True, exist_ok=True)
    for step, (_, name) in STEPS.items():
        dst = OUT / name
        if dst.exists():
            print(f'  {name}: already in {OUT.name}/ - left as is')
            continue
        if not REFS[step].exists():
            sys.exit(f'ERROR: {REFS[step]} not found')
        shutil.copy2(REFS[step], dst)
        print(f'  {name}: copied from {REFS[step]}')
    print('NEXT: python longitudinal/extract.py add XXX NN [XXX NN ...]')


def add(pairs, table):
    targets = {}
    for sub, ses in pairs:
        targets.setdefault(f'sub-{sub}', set()).add(ses)
    want = {(k, int(s)) for k, v in targets.items() for s in v}
    lab = lambda k: f'{k[0]} ses-{int(k[1]):02d}'
    for step, (_, name) in STEPS.items():
        cur_file = OUT / name
        if not cur_file.exists():
            sys.exit(f'ERROR: {cur_file} missing - run seed first')
        tmp = f'_new_{name}'
        print(f'\n===== {step}: {", ".join(sorted(lab(k) for k in want))}')
        if step == 'mni':           # convert only these sessions' rows of the liu file
            liu = pd.read_csv(OUT / STEPS['liu'][1], low_memory=False)
            keep = [(sid, int(n)) in want for sid, n in zip(liu['subject_id'], ses_num(liu))]
            mni_in = OUT / '_mni_input.csv'
            liu[keep].to_csv(mni_in, index=False)
            secs = run_module(step, table, tmp, mni_input=mni_in)
            mni_in.unlink()
        else:
            try:
                secs = run_module(step, table, tmp, targets=targets)
            except KeyError:        # the scripts' summaries fail on an empty table
                sys.exit(f'ERROR: {step} extracted no rows for {sorted(lab(k) for k in want)}')
        cur = pd.read_csv(cur_file, low_memory=False)
        new = pd.read_csv(OUT / tmp, low_memory=False)
        (OUT / tmp).unlink()
        cn, nn = ses_num(cur), ses_num(new)
        num = numeric_cols(cur, new)
        for k in sorted(want):
            o = cur[(cur['subject_id'] == k[0]) & (cn == k[1])]
            n = new[(new['subject_id'] == k[0]) & (nn == k[1])]
            status = ('NO ROWS EXTRACTED' if not len(n) else 'new' if not len(o)
                      else same_rows(o, n, num))
            print(f'  {lab(k)}: {len(n)} rows, {status}')
        hit = [(s, int(x)) in want for s, x in zip(cur['subject_id'], cn)]
        shutil.copy2(cur_file, f'{cur_file}.bak')
        merged = pd.concat([cur[[not h for h in hit]], new], ignore_index=True)[cur.columns]
        merged.to_csv(cur_file, index=False)
        print(f'  {name}: {len(merged)} rows ({secs / 60:.1f} min; previous copy {name}.bak)')
    print('\nExpected: univariate/rsa "identical" for sessions already in the seed, '
          '"new" for new sessions; liu/peak MNI "changed" for the .longitudinal sessions.')


def full(table, steps):
    OUT.mkdir(parents=True, exist_ok=True)
    for step, (_, name) in [(k, v) for k, v in STEPS.items() if k in steps]:
        print(f'\n===== {step}: 2_extraction/{STEPS[step][0]} -> {OUT / name}')
        secs = run_module(step, table, name, mni_input=OUT / STEPS['liu'][1])
        print(f'===== {step}: done in {secs / 60:.1f} min')


UCOLS = ['peak_x_native', 'peak_y_native', 'peak_z_native', 'peak_z',
         'sum_selec_norm', 'mean_act', 'volume', 'n_searchmask']
RCOLS = ['fisher_r', 'liu_distinctiveness']


def maxdiff(m, cols, sa, sb):
    """Largest |a - b| over matched rows, and whether missing cells agree."""
    worst, same_nan = 0.0, True
    for c in cols:
        a, b = m[c + sa].astype(float).to_numpy(), m[c + sb].astype(float).to_numpy()
        same_nan &= bool((np.isnan(a) == np.isnan(b)).all())
        d = np.abs(a - b)
        d = d[~np.isnan(d)]
        if d.size:
            worst = max(worst, float(d.max()))
    return worst, same_nan


def compare(step, table):
    """uni, rsa: vs the 2026-11-02 extraction. liu: its univariate and RSA columns must
    equal univariate_v1/rsa_v1 (03 computes them with the same code as 01 and 02).
    peak MNI: one row per liu row with a native peak, all converted."""
    new = OUT / STEPS[step][1]
    if not new.exists():
        print(f'  {step}: missing {new}')
        return False
    lab = lambda k: f'{k[0]} ses-{int(k[1]):02d}'
    if step in ('uni', 'rsa'):
        ref = REFS[step]
        a, b = pd.read_csv(ref, low_memory=False), pd.read_csv(new, low_memory=False)
        a['ses_num'], b['ses_num'] = ses_num(a), ses_num(b)
        num = numeric_cols(a, b)
        ga, gb = a.groupby(['subject_id', 'ses_num'], sort=False), b.groupby(['subject_id', 'ses_num'], sort=False)
        ka, kb = set(ga.groups), set(gb.groups)
        differ = [(k, w) for k in sorted(ka & kb)
                  if (w := same_rows(ga.get_group(k), gb.get_group(k), num)) != 'identical']
        ok = not differ and not (ka - kb)
        print(f'  {step}: {len(ka & kb) - len(differ)} sessions identical to {ref.parent.parent.name}/{ref.name}'
              f' | differ: {", ".join(lab(k) + " (" + w + ")" for k, w in differ) or "none"}'
              f' | only in new: {", ".join(lab(k) for k in sorted(kb - ka)) or "none"}'
              f' | missing: {", ".join(lab(k) for k in sorted(ka - kb)) or "none"} -> {"PASS" if ok else "CHECK"}')
        return ok
    liu = pd.read_csv(OUT / STEPS['liu'][1], low_memory=False)
    liu['ses_num'] = ses_num(liu)
    if step == 'liu':
        uni = pd.read_csv(OUT / STEPS['uni'][1], low_memory=False)
        rsa = pd.read_csv(OUT / STEPS['rsa'][1], low_memory=False)
        uni['ses_num'], rsa['ses_num'] = ses_num(uni), ses_num(rsa)
        k = ['subject_id', 'ses_num', 'hemi', 'category']
        mu = liu.drop_duplicates(k)[k + UCOLS].merge(uni[k + UCOLS], on=k, suffixes=('_l', '_u'))
        du, nu = maxdiff(mu, UCOLS, '_l', '_u')
        lr = liu[liu['pair'].notna()][k + ['pair'] + RCOLS]
        mr = lr.merge(rsa[k + ['pair'] + RCOLS], on=k + ['pair'], suffixes=('_l', '_r'))
        dr, nr = maxdiff(mr, RCOLS, '_l', '_r')
        bad = sorted({lab((r.subject_id, r.ses_num)) for r in mu.itertuples()
                      if any(abs(getattr(r, c + '_l') - getattr(r, c + '_u')) > 1e-9
                             for c in UCOLS if not (pd.isna(getattr(r, c + '_l')) and pd.isna(getattr(r, c + '_u'))))})
        ok = len(mu) > 0 and len(mr) > 0 and du <= 1e-9 and dr <= 1e-9 and nu and nr
        print(f'  liu: vs univariate_v1.csv {len(mu)} ROI rows, max diff {du:.2g}; '
              f'vs rsa_v1.csv {len(mr)} rows, max diff {dr:.2g} -> {"PASS" if ok else "CHECK"}')
        if bad:
            print(f'        sessions that differ: {", ".join(bad[:12])}{" ..." if len(bad) > 12 else ""}')
        return ok
    mni = pd.read_csv(new, low_memory=False)
    expect = liu.dropna(subset=['peak_x_native', 'peak_y_native', 'peak_z_native'])
    n_nan = int(mni[['peak_x_mni', 'peak_y_mni', 'peak_z_mni']].isna().any(axis=1).sum())
    ok = len(mni) == len(expect) and n_nan == 0
    print(f'  mni: {len(mni)} rows for {len(expect)} liu rows with a native peak; '
          f'{n_nan} without MNI coordinates -> {"PASS" if ok else "CHECK"}')
    return ok


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ('seed', 'add', 'validate', 'full'):
        sys.exit(__doc__)
    table = longitudinal_dirs()
    print('sessions read from HighLevel.gfeat.longitudinal: '
          + (', '.join(session_label(c + '/') for c in table) or 'none'))
    if a[0] == 'seed':
        seed()
    elif a[0] == 'add':
        rest = a[1:]
        if not rest or len(rest) % 2:
            sys.exit('usage: python longitudinal/extract.py add XXX NN [XXX NN ...]')
        pairs = [(rest[i].replace('sub-', ''), f'{int(rest[i + 1].replace("ses-", "")):02d}')
                 for i in range(0, len(rest), 2)]
        add(pairs, table)
    elif a[0] == 'full':
        steps = a[1:] or list(STEPS)
        if [x for x in steps if x not in STEPS]:
            sys.exit(f'usage: python longitudinal/extract.py full [{" ".join(STEPS)}]')
        full(table, steps)
    else:
        print(f'== validate (read-only): {OUT} vs reference files')
        ok = all([compare(s, table) for s in STEPS])
        print('VALIDATED' if ok else 'NOT VALIDATED - see CHECK lines')
        print('Cross-sectional outputs untouched? Run (expect no output):\n'
              '  python ../check_outputs_v2.py --ignore sub-092:2 sub-108:2 | diff outputs/check_restored_20261102.txt -')


if __name__ == '__main__':
    main()
