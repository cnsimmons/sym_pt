#!/usr/bin/env python3
"""
longitudinal/harmonize.py                                            2026-11-11
Step 3 (harmonization) for the longitudinal study.
  reads   pipeline/outputs_longitudinal/univariate_v1.csv, rsa_v1.csv   (longitudinal/extract.py)
  writes  pipeline/outputs_longitudinal/univariate_v1_harmonized_sqrt.csv   every session
          pipeline/outputs_longitudinal/rsa_v1_harmonized.csv               every session
          pipeline/outputs_longitudinal/harmonization/
              sub_info_scanner.csv          the scanner table used here
              fits/                         02_harmonize.py's models, fitted subjects and
                                            features, plus winsor_limits.csv
              univariate_fit_sessions.csv   02_harmonize.py's own output (fit sessions only)
              rsa_fit_sessions.csv

The fit is 3_harmonization/02_harmonize.py's own univariate() and rsa(), unchanged:
ComBat (neuroHarmonize), SITE = scanner, age, sex and group preserved, one fit per
hemisphere, 5/95 winsorizing within group, sqrt for volume and sum_selec_norm. Fit set as
in that script: controls' first session, patients' last (lib_stats exclusions, sub-017 out,
EXCLUDE_SES as it stands). Two things are swapped in, as extract.py swaps the extraction
scripts' outputs:
  - paths: inputs, outputs, fits and the scanner table, all under outputs_longitudinal/;
  - the scanner table: the cross-sectional one (read only) plus the sessions it lacks,
    labelled in SCANNER_ADDED.
New code, the apply step only: each saved fit is applied (harmonizationApply) to every
session of the fitted subjects, with that session's own scanner, age and sex, the same
sqrt, and the fit's own winsorizing limits. On the fit sessions the applied values must
equal 02_harmonize.py's own output, or the two output files are not written.

Batch is scanner only, as in the cross-sectional pipeline (C.S., 2026-11-11).
02_harmonize.py's tfce() is never called (it writes outputs/F_harmonization). Nothing
is written outside outputs_longitudinal/.

Command (fmri env, from anywhere):
  python /user_data/csimmon2/git_repos/sym_pt/pipeline/longitudinal/harmonize.py
Refuses to run if outputs_longitudinal/harmonization/ exists: move it aside to rerun.
"""
import glob
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True               # no __pycache__ beside 02_harmonize.py or lib_stats.py

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply

PIPE = Path(__file__).resolve().parents[1]           # pipeline/, as in extract.py
OUT = PIPE / 'outputs_longitudinal'
HARM = OUT / 'harmonization'
RAW = Path('/lab_data/behrmannlab/hemi/Raw')
HARMONIZE = PIPE / '3_harmonization' / '02_harmonize.py'
LIB_STATS = PIPE / '3_harmonization' / 'lib_stats.py'
SCANNER_XS = PIPE / 'outputs' / 'F_harmonization' / 'sub_info_scanner.csv'   # cross-sectional; read only
SUB_INFO = PIPE / 'sub_info.csv'

UNIVAR_IN, RSA_IN = OUT / 'univariate_v1.csv', OUT / 'rsa_v1.csv'
UNIVAR_OUT, RSA_OUT = OUT / 'univariate_v1_harmonized_sqrt.csv', OUT / 'rsa_v1_harmonized.csv'

# Sessions the cross-sectional scanner table lacks (C.S., 2026-11-11: all on the Prisma).
# A raw JSON with no scanner field is expected; one that names another scanner stops the run.
SCANNER_ADDED = {('sub-092', 'ses-02'): 'Prisma',
                 ('sub-098', 'ses-02'): 'Prisma',
                 ('sub-101', 'ses-02'): 'Prisma'}

TOL = 1e-8                                   # self-check tolerance (fit sessions)
UNI_KEY = ['subject_id', 'ses_num', 'hemi', 'category']
RSA_KEY = UNI_KEY + ['pair']
RSA_COLS = ['liu_distinctiveness', 'fisher_r']


# -- helpers ------------------------------------------------------------------
def stop(msg):
    sys.exit(f'STOPPED: {msg}')


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def writable(path):
    """Every path this script writes must sit under outputs_longitudinal/."""
    p = Path(path).resolve()
    if OUT.resolve() not in p.parents:
        stop(f'refusing to write outside {OUT}: {p}')
    return p


def ses_num(d):
    return pd.to_numeric(d['session'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')


def ses_label(n):
    return f'ses-{int(n):02d}'


def read_input(path, key):
    """An extraction file with a ses_num column (lib_stats adds the same column for 02)."""
    d = pd.read_csv(path, low_memory=False)
    n = ses_num(d)
    if n.isna().any():
        stop(f'{path.name}: {int(n.isna().sum())} rows have no session number')
    d['ses_num'] = n.astype(int)
    k = d[key].copy()
    if 'pair' in key:
        k['pair'] = k['pair'].fillna('')
    dup = int(k.duplicated().sum())
    if dup:
        stop(f'{path.name}: {dup} rows repeat a {"/".join(key)} key; the write-back assumes one row per key')
    return d


def load_harmonize():
    spec = importlib.util.spec_from_file_location('harmonize02', str(HARMONIZE))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


# -- scanner table ------------------------------------------------------------
def json_scanner(sub, ses):
    """01_add_scanner.py's get_scanner, copied: that script writes the cross-sectional table."""
    hits = sorted(glob.glob(f'{RAW}/{sub}/{ses}/func/*task-loc*_bold.json'))
    if not hits:
        return 'MISSING'
    with open(hits[0]) as f:
        meta = json.load(f)
    name = (meta.get('ManufacturersModelName') or meta.get('Manufacturer') or '').lower()
    if 'prisma' in name:
        return 'Prisma'
    if 'verio' in name:
        return 'Verio'
    return f'UNKNOWN:{name}'


def scanner_table(needed):
    """The cross-sectional scanner table plus the needed sessions it lacks (rows from sub_info.csv)."""
    xs = pd.read_csv(SCANNER_XS)
    si = pd.read_csv(SUB_INFO)
    lacking = sorted(needed - set(zip(xs['sub'], xs['ses'])))
    unlabelled = [k for k in lacking if k not in SCANNER_ADDED]
    if unlabelled:
        stop('no scanner label for ' + ', '.join(f'{a} {b}' for a, b in unlabelled)
             + '. Add them to SCANNER_ADDED once their scanner is known. Nothing written.')
    rows = []
    for sub, ses in lacking:
        r = si[(si['sub'] == sub) & (si['ses'] == ses)]
        if not len(r):
            stop(f'{sub} {ses} is not in {SUB_INFO}. Nothing written.')
        found, label = json_scanner(sub, ses), SCANNER_ADDED[(sub, ses)]
        if found in ('Prisma', 'Verio') and found != label:
            stop(f'{sub} {ses}: its JSON says {found}, SCANNER_ADDED says {label}. Nothing written.')
        r = r.iloc[[0]].copy()
        r['scanner'] = label
        rows.append(r)
        print(f'  added {sub} {ses}: {label} (SCANNER_ADDED; JSON: {found})')
    unused = sorted(set(SCANNER_ADDED) - set(lacking))
    if unused:
        print('  SCANNER_ADDED entries not needed: ' + ', '.join(f'{a} {b}' for a, b in unused))
    return pd.concat([xs] + rows, ignore_index=True).reindex(columns=xs.columns)


# -- fit (02_harmonize.py) ----------------------------------------------------
def capture_fits(h):
    """Keep each fit 02_harmonize.py saves: model, fitted subjects with covariates, features."""
    fits, save = {}, h.save_fit

    def save_fit(model, name, subs, cov, features):
        fits[name] = {'model': model, 'subs': list(subs), 'cov': cov.loc[list(subs)].copy(),
                      'features': list(features)}
        save(model, name, subs, cov, features)

    h.save_fit = save_fit
    return fits


# -- apply (new) --------------------------------------------------------------
def session_covariates(obs, fit, scan):
    """Covariates per session, taken as 02's build_covariates takes them: scanner, age and sex
    from that session's scanner-table row; group from the fit (it does not change by session)."""
    rows = []
    for sid, n in zip(obs['subject_id'], obs['ses_num']):
        m = scan[(scan['sub'] == sid) & (scan['ses'] == ses_label(n))]
        if not len(m):
            stop(f'no scanner-table row for {sid} {ses_label(n)}')
        m = m.iloc[0]
        if pd.isna(m['scanner']) or pd.isna(m['age']) or pd.isna(m['sex']):
            stop(f'blank scanner, age or sex for {sid} {ses_label(n)}')
        rows.append({'subject_id': sid, 'ses_num': int(n), 'group': fit['cov'].loc[sid, 'group'],
                     'scanner': str(m['scanner']), 'age': m['age'], 'sex': m['sex']})
    return pd.DataFrame(rows)


def design_for(h, fit, cov):
    """The fit's own design columns (02's roi_design), filled in for these sessions."""
    ref = h.roi_design(fit['cov'], fit['subs'])
    for col in ('scanner', 'sex', 'group'):
        new = set(cov[col].astype(str)) - set(fit['cov'][col].astype(str))
        if new:
            stop(f'{col} {sorted(new)} does not occur in the fit, so the fit cannot be applied')
    d = pd.DataFrame({'SITE': cov['scanner'].astype(str).to_numpy(),
                      'age': cov['age'].astype(float).to_numpy()})
    for c in [c for c in ref.columns if c not in ('SITE', 'age')]:
        col, level = c.split('_', 1)
        d[c] = (cov[col].astype(str).to_numpy() == level).astype(int)
    return d[list(ref.columns)]


def apply_fit(h, fit, X, cov, block, hemi, lim):
    """Winsorize with the fit's limits, then apply the saved fit to every row of X."""
    X = X.astype(float)
    groups = cov['group'].astype(str).to_numpy()
    for j, f in enumerate(fit['features']):
        for g in np.unique(groups):
            if (block, hemi, f, g) in lim:
                lo, hi = lim[(block, hemi, f, g)]
                X[groups == g, j] = np.clip(X[groups == g, j], lo, hi)     # NaN stays NaN
    nan_mask = h.impute_for_fit(X, pd.Series(groups), f'{block} {hemi.upper()}H, all sessions')
    adj = np.asarray(harmonizationApply(X, design_for(h, fit, cov), fit['model']), dtype=float)
    if adj.shape != X.shape:
        stop(f'{block} {hemi}: harmonizationApply returned shape {adj.shape}, expected {X.shape}')
    lost = np.isnan(adj) & ~nan_mask
    if lost.any():
        stop(f'{block} {hemi}: harmonizationApply returned NaN for {int(lost.sum())} non-missing cells')
    adj[nan_mask] = np.nan
    return adj


def sessions_of(rows):
    obs = (rows[['subject_id', 'ses_num']].drop_duplicates()
           .sort_values(['subject_id', 'ses_num']).reset_index(drop=True))
    return obs, pd.MultiIndex.from_frame(obs)


def column(rows, idx, value):
    """One value per session (first non-missing, as 02's pivot takes it), in idx order."""
    return rows.groupby(['subject_id', 'ses_num'])[value].first().reindex(idx)


def write_back(out, col, tgt, values, idx, done, keep_raw_if_missing):
    keys = pd.MultiIndex.from_frame(out.loc[tgt, ['subject_id', 'ses_num']])
    v = pd.Series(values, index=idx).reindex(keys).to_numpy()
    if keep_raw_if_missing:                 # 02's RSA: only valid cells are written
        ok = ~np.isnan(v)
        out.loc[tgt, col] = np.where(ok, v, out.loc[tgt, col].to_numpy())
        done[np.flatnonzero(tgt)[ok]] = True
    else:                                   # 02's univariate: every cell of the fitted rows
        out.loc[tgt, col] = v
        done[tgt] = True


def univariate_all(h, fits, lim, scan, d, report):
    out = d.copy()
    out[h.MEASURES] = out[h.MEASURES].astype(float)
    done = np.zeros(len(out), bool)
    for hemi in ['l', 'r']:
        name = f'univariate_{hemi}'
        if name not in fits:
            continue
        fit = fits[name]
        mine = (out['hemi'] == hemi).to_numpy() & out['subject_id'].isin(fit['subs']).to_numpy()
        rows = out[mine]
        obs, idx = sessions_of(rows)
        cov = session_covariates(obs, fit, scan)
        cols = []
        for f in fit['features']:
            roi, meas = f.split('__')
            cols.append(h._fwd(column(rows[rows['category'] == roi], idx, meas), meas).to_numpy(dtype=float))
        X = np.column_stack(cols)
        adj = apply_fit(h, fit, X, cov, 'univariate', hemi, lim)
        for j, f in enumerate(fit['features']):
            roi, meas = f.split('__')
            tgt = mine & (out['category'] == roi).to_numpy()
            write_back(out, meas, tgt, adj[:, j], idx, done, keep_raw_if_missing=False)
        report.append((name, len(fit['subs']), cov))
    for meas in h.SQRT_MEASURES:
        out[f'{meas}_scale'] = 'sqrt'
    raw = d[h.MEASURES].notna().any(axis=1).to_numpy()
    return out, {'univariate measure': (done, raw)}


def rsa_all(h, fits, lim, scan, d, report):
    out = d.copy()
    for c in RSA_COLS:
        out[c] = out[c].astype(float)
    done_d, done_g = np.zeros(len(out), bool), np.zeros(len(out), bool)
    row_feat = (out['category'].astype(str) + '__' + out['pair'].astype(str)).to_numpy()
    for hemi in ['l', 'r']:
        fd, fg = fits.get(f'rsa_distinct_{hemi}'), fits.get(f'rsa_geometry_{hemi}')
        if fd is None or fg is None:
            continue
        if fd['subs'] != fg['subs']:
            stop(f'RSA {hemi}: the distinctiveness and geometry fits have different subjects')
        mine = (out['hemi'] == hemi).to_numpy() & out['subject_id'].isin(fd['subs']).to_numpy()
        rows = out[mine]
        obs, idx = sessions_of(rows)
        cov = session_covariates(obs, fd, scan)

        dd = rows.dropna(subset=['liu_distinctiveness'])        # one value per session and category
        X = np.column_stack([column(dd[dd['category'] == f], idx, 'liu_distinctiveness').to_numpy(dtype=float)
                             for f in fd['features']])
        adj = apply_fit(h, fd, X, cov, 'rsa_distinct', hemi, lim)
        for j, f in enumerate(fd['features']):
            tgt = mine & (out['category'] == f).to_numpy()
            write_back(out, 'liu_distinctiveness', tgt, adj[:, j], idx, done_d, keep_raw_if_missing=True)

        dg = rows.dropna(subset=['fisher_r'])                   # one value per session, category, pair
        feat = (dg['category'].astype(str) + '__' + dg['pair'].astype(str)).to_numpy()
        X = np.column_stack([column(dg[feat == f], idx, 'fisher_r').to_numpy(dtype=float)
                             for f in fg['features']])
        adj = apply_fit(h, fg, X, cov, 'rsa_geometry', hemi, lim)
        for j, f in enumerate(fg['features']):
            tgt = mine & (row_feat == f)
            write_back(out, 'fisher_r', tgt, adj[:, j], idx, done_g, keep_raw_if_missing=True)
        report.append((f'rsa_{hemi}', len(fd['subs']), cov))
    need = {'liu_distinctiveness': (done_d, d['liu_distinctiveness'].notna().to_numpy()),
            'fisher_r': (done_g, d['fisher_r'].notna().to_numpy())}
    return out, need


def keep_fitted(out, need, fitted, label):
    """Rows of fitted subjects only; stop if any non-missing raw value was left unharmonized."""
    in_fit = out['subject_id'].isin(fitted).to_numpy()
    for what, (done, raw) in need.items():
        left = in_fit & raw & ~done
        if left.any():
            e = out.loc[left, UNI_KEY].drop_duplicates().head(10).to_string(index=False)
            stop(f'{label}: {int(left.sum())} rows of fitted subjects would keep a raw {what} value '
                 f'(no fit covers them). First rows:\n{e}\nThe two output files were not written.')
    dropped = sorted(set(out.loc[~in_fit, 'subject_id']))
    return out[in_fit].copy(), dropped


def self_check(mine, ref, fitted_by_hemi, key, cols, label):
    """On the fit sessions, the applied values must equal 02_harmonize.py's own output."""
    ref = ref[[s in fitted_by_hemi.get(hm, ()) for s, hm in zip(ref['subject_id'], ref['hemi'])]]
    a, b = ref[key + cols].copy(), mine[key + cols].copy()
    if 'pair' in key:
        a['pair'], b['pair'] = a['pair'].fillna(''), b['pair'].fillna('')
    m = a.merge(b, on=key, how='left', suffixes=('_ref', '_new'), indicator=True)
    missing = int((m['_merge'] == 'left_only').sum())
    worst, same_nan = 0.0, True
    for c in cols:
        x, y = m[c + '_ref'].astype(float).to_numpy(), m[c + '_new'].astype(float).to_numpy()
        same_nan &= bool((np.isnan(x) == np.isnan(y)).all())
        diff = np.abs(x - y)
        diff = diff[~np.isnan(diff)]
        if diff.size:
            worst = max(worst, float(diff.max()))
    ok = len(m) > 0 and missing == 0 and same_nan and worst <= TOL
    print(f'  {label}: {len(m)} fit-session rows vs 02_harmonize.py output | max diff {worst:.2g} | '
          f'missing {missing} | NaN {"same" if same_nan else "DIFFER"} -> {"PASS" if ok else "FAIL"}')
    return ok


def write_final(df, path):
    path = writable(path)
    tmp = writable(path.with_name('_new_' + path.name))
    df.to_csv(tmp, index=False)
    note = ''
    if path.exists():
        bak = writable(path.with_name(f'{path.name}.bak_{time.strftime("%Y%m%d_%H%M%S")}'))
        shutil.copy2(path, bak)
        note = f'; previous copy {bak.name}'
    shutil.move(str(tmp), str(path))
    n_ses = len(df[['subject_id', 'ses_num']].drop_duplicates())
    print(f'  wrote {path} ({df["subject_id"].nunique()} subjects, {n_ses} sessions, {len(df)} rows{note})')


# -- main ---------------------------------------------------------------------
def main():
    t0 = time.time()
    for p in (UNIVAR_IN, RSA_IN, SCANNER_XS, SUB_INFO, HARMONIZE, LIB_STATS):
        if not p.exists():
            stop(f'missing {p}. Nothing written.')
    if HARM.exists():
        stop(f'{HARM} exists; move it aside to rerun. Nothing written.')
    print(f'{Path(__file__).name} md5 {md5(__file__)} | 02_harmonize.py md5 {md5(HARMONIZE)} | '
          f'lib_stats.py md5 {md5(LIB_STATS)}')
    print(f'{UNIVAR_IN.name} md5 {md5(UNIVAR_IN)} | {RSA_IN.name} md5 {md5(RSA_IN)}')

    print('\n========== scanner table ==========')
    uni, rsa = read_input(UNIVAR_IN, UNI_KEY), read_input(RSA_IN, RSA_KEY)
    needed = {(s, ses_label(n)) for d in (uni, rsa) for s, n in zip(d['subject_id'], d['ses_num'])}
    scan = scanner_table(needed)
    HARM.mkdir(parents=True)
    scan_path = writable(HARM / 'sub_info_scanner.csv')
    scan.to_csv(scan_path, index=False)
    scan = pd.read_csv(scan_path)            # read back, as 02_harmonize.py reads it
    print(f'  {scan_path}: {len(scan)} rows | ' + str(scan['scanner'].value_counts().to_dict()))

    h = load_harmonize()
    h.UNIVAR, h.RSA, h.SCANNER = UNIVAR_IN, RSA_IN, scan_path
    h.UNIVAR_OUT = writable(HARM / 'univariate_fit_sessions.csv')
    h.RSA_OUT = writable(HARM / 'rsa_fit_sessions.csv')
    h.FITS = writable(HARM / 'fits')
    h.OUT = OUT                               # never used by univariate() or rsa(); set anyway
    h.TFCE_IN, h.TFCE_OUT = HARM / 'tfce_unused_in', HARM / 'tfce_unused_out'
    print(f'  fit-set exclusions: 02 ROI_FIT_EXCLUDE {h.ROI_FIT_EXCLUDE} | lib_stats EXCLUDE_SES {h.s.EXCLUDE_SES}')
    fits = capture_fits(h)

    h.univariate()                            # 02_harmonize.py, unchanged
    h.rsa()
    lim_path = writable(h.FITS / 'winsor_limits.csv')
    pd.DataFrame(h.LIMITS).to_csv(lim_path, index=False)
    lim = {(r['block'], r['hemi'], r['feature'], r['group']): (r['lo'], r['hi']) for r in h.LIMITS}

    print('\n========== apply the fits to every session ==========')
    report = []
    u_out, u_need = univariate_all(h, fits, lim, scan, uni, report)
    r_out, r_need = rsa_all(h, fits, lim, scan, rsa, report)
    u_fit = {hm: set(fits[f'univariate_{hm}']['subs']) for hm in 'lr' if f'univariate_{hm}' in fits}
    r_fit = {hm: set(fits[f'rsa_distinct_{hm}']['subs']) for hm in 'lr' if f'rsa_distinct_{hm}' in fits}
    u_out, u_drop = keep_fitted(u_out, u_need, set().union(*u_fit.values()), 'univariate')
    r_out, r_drop = keep_fitted(r_out, r_need, set().union(*r_fit.values()), 'RSA')

    u_ref = pd.read_csv(h.UNIVAR_OUT, low_memory=False)
    r_ref = pd.read_csv(h.RSA_OUT, low_memory=False)
    for ref in (u_ref, r_ref):
        ref['ses_num'] = ses_num(ref).astype(int)
    for name, n_fit, cov in report:
        fit_out = u_ref if name.startswith('univariate') else r_ref
        fs = set(zip(fit_out['subject_id'], fit_out['ses_num']))
        extra = sum((s, n) not in fs for s, n in zip(cov['subject_id'], cov['ses_num']))
        print(f'  {name}: fit on {n_fit} subjects; applied to {len(cov)} sessions, {extra} of them beyond the fit')
    print('  not fitted, left out: univariate ' + (', '.join(u_drop) or 'none')
          + ' | RSA ' + (', '.join(r_drop) or 'none'))

    print('\n========== self-check ==========')
    ok = all([self_check(u_out, u_ref, u_fit, UNI_KEY, h.MEASURES, 'univariate'),
              self_check(r_out, r_ref, r_fit, RSA_KEY, RSA_COLS, 'RSA')])
    for mine, ref, label in ((u_out, u_ref, 'univariate'), (r_out, r_ref, 'RSA')):
        if list(mine.columns) != list(ref.columns):
            print(f'  {label}: columns differ from 02_harmonize.py output: '
                  f'{sorted(set(mine.columns) ^ set(ref.columns)) or "same set, other order"}')
            ok = False
    if not ok:
        stop(f'self-check failed; the two output files were not written. Fits are in {HARM} for review.')

    print('\n========== outputs ==========')
    write_final(u_out, UNIVAR_OUT)
    write_final(r_out, RSA_OUT)

    ses = {}
    for _, _, cov in report:
        for sid, n, sc in zip(cov['subject_id'], cov['ses_num'], cov['scanner']):
            ses.setdefault(sid, {})[int(n)] = sc
    changed = sorted(s for s, v in ses.items() if len(set(v.values())) > 1)
    print(f'\nSubjects whose scanner changes between sessions: {len(changed)}')
    for sid in changed:
        print(f'  {sid}: ' + ', '.join(f'{ses_label(n)} {ses[sid][n]}' for n in sorted(ses[sid])))

    print(f'\nDone in {(time.time() - t0) / 60:.1f} min.')
    print('Cross-sectional outputs untouched? From pipeline/ (expect no output):\n'
          '  python ../check_outputs_v2.py --ignore sub-092:2 sub-108:2 | diff outputs/check_restored_20261102.txt -')


if __name__ == '__main__':
    main()
