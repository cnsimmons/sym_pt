#!/usr/bin/env python3
"""
renumber_pipeline.py - give pipeline/ numbered stage folders and file names.

Moves (git mv) the 53 scripts built by build_pipeline.sh into numbered stage folders
and updates every path that refers to a moved file or to a repo-side output folder.
Only paths change; checks at the end confirm that.

  1_preprocessing  2_extraction  3_harmonization  4_analyses  5_figures  outputs/
  params.py, sym_pt_params.py, sub_info.csv and resources/ stay where they are.

Outputs the scripts write inside the repo (D_liu, F_harmonization, H_RosenkeRSM,
C_results, logs_combat) go to pipeline/outputs/<same name>, so the validation run can
be compared file by file with the committed originals.

Run once on the cluster:   python renumber_pipeline.py
Every changed line is written to pipeline/renumber_changes.txt for review.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ABS = '/user_data/csimmon2/git_repos/sym_pt/pipeline'      # path string inside the scripts
P = Path(os.environ.get('PIPE_ROOT', ABS))                  # PIPE_ROOT only for testing
G = P.parent

MAP = {
    # 1 - preprocessing
    'A_preprocessing/00_mat2tsv_timing.py':           '1_preprocessing/00a_mat2tsv_timing.py',
    'A_preprocessing/00_prt2tsv_timing.py':           '1_preprocessing/00b_prt2tsv_timing.py',
    'A_preprocessing/01_organize.py':                 '1_preprocessing/01_organize.py',
    'A_preprocessing/1.5_extract_confounds.sh':       '1_preprocessing/02_extract_confounds.sh',
    'A_preprocessing/02_convert_timing.py':           '1_preprocessing/03_convert_timing.py',
    'A_preprocessing/03_submit_anatomy_new.sh':       '1_preprocessing/04_submit_anatomy_new.sh',
    'A_preprocessing/register_mirror.py':             '1_preprocessing/04_register_mirror.py',
    'A_preprocessing/check_anatomy_03_register.ipynb':'1_preprocessing/04_check_anatomy.ipynb',
    'A_preprocessing/05_create_fsf.sh':               '1_preprocessing/05_create_fsf.sh',
    'A_preprocessing/template_1stLevel.fsf':          '1_preprocessing/05_template_1stLevel.fsf',
    'A_preprocessing/06_submit_1stLevel_feat.sh':     '1_preprocessing/06_submit_1stLevel_feat.sh',
    'A_preprocessing/00_spike_verification.sh':       '1_preprocessing/06_spike_verification.sh',
    'A_preprocessing/07_register_anat_to_ses01.sh':   '1_preprocessing/07_register_anat_to_ses01.sh',
    'A_preprocessing/08_submit_1stLevel_register.py': '1_preprocessing/08_submit_1stLevel_register.py',
    'A_preprocessing/register_1stLevel.py':           '1_preprocessing/08_register_1stLevel.py',
    'A_preprocessing/09_submit_zstat_register.py':    '1_preprocessing/09_submit_zstat_register.py',
    'A_preprocessing/register_zstats.py':             '1_preprocessing/09_register_zstats.py',
    'A_preprocessing/10_create_highlevel_fsf.sh':     '1_preprocessing/10_create_highlevel_fsf.sh',
    'A_preprocessing/template_HighLevel.fsf':         '1_preprocessing/10_template_HighLevel.fsf',
    'A_preprocessing/11_submit_highlevel_feat.sh':    '1_preprocessing/11_submit_highlevel_feat.sh',
    'A_preprocessing/12_submit_highlevel_register.py':'1_preprocessing/12_submit_highlevel_register.py',
    'A_preprocessing/register_highlevel.py':          '1_preprocessing/12_register_highlevel.py',
    'A_preprocessing/check_registration.sh':          '1_preprocessing/12_check_registration.sh',
    'A_preprocessing/13_register2mni.py':             '1_preprocessing/13_register2mni.py',
    'B_analyses/01_create_searchmasks.py':            '1_preprocessing/14_create_searchmasks.py',
    'B_analyses/add_ppa_strict_searchmasks.py':       '1_preprocessing/15_add_ppa_strict_searchmasks.py',
    # 2 - extraction
    'D_liu/verified/01_univariate_analyses.py':       '2_extraction/01_univariate.py',
    'D_liu/verified/04_multivariate_analyses.py':     '2_extraction/02_multivariate.py',
    'D_liu/liu_recreation_csv_v2.py':                 '2_extraction/03_liu_recreation.py',
    'D_liu/calc_peak_coords_mni.py':                  '2_extraction/04_peak_coords_mni.py',
    'extract_selective_voxel_counts.py':              '2_extraction/05_selective_voxel_counts.py',
    'H_RosenkeRSM/otc_rsm_rosenke.py':                '2_extraction/06_otc_rsm_rosenke.py',
    'D_liu/verified/02_tfce_analyses_dontuse_useharmony.py': '2_extraction/lib_tfce.py',
    # 3 - harmonization (the four ComBat scripts stay as reference until 02_harmonize.py matches them)
    'F_harmonization/add_scanner.py':                 '3_harmonization/01_add_scanner.py',
    'F_harmonization/combat_05b_harmonize_univariate_sqrt.py': '3_harmonization/reference/combat_05b_harmonize_univariate_sqrt.py',
    'F_harmonization/combat_06_harmonize_rsa.py':     '3_harmonization/reference/combat_06_harmonize_rsa.py',
    'F_harmonization/combat_01_build_matrix.py':      '3_harmonization/reference/combat_01_build_matrix.py',
    'F_harmonization/combat_02_harmonize.py':         '3_harmonization/reference/combat_02_harmonize.py',
    'F_harmonization/combat_03_tfce_harmonized.py':   '3_harmonization/03_tfce.py',
    'F_harmonization/submit_combat.sh':               '3_harmonization/03_submit_tfce.sh',
    'D_liu/verified/05_stats.py':                     '3_harmonization/lib_stats.py',
    # 4 - analyses
    'D_liu/verified/03_wta_analyses.py':              '4_analyses/01_wta.py',
    'D_liu/verified/05_stats_harmony.py':             '4_analyses/02_stats_harmony.py',
    'control_li.py':                                  '4_analyses/03_control_li.py',
    'marlene_grid.py':                                '4_analyses/04_marlene_grid.py',
    'marlene_lmm.py':                                 '4_analyses/05_marlene_lmm.py',
    'one_last_8.27/marlene_roi.py':                   '4_analyses/06_marlene_roi.py',
    'D_liu/verified/07_grid.py':                      '4_analyses/07_grid.py',
    'D_liu/verified/grid_cohort.py':                  '4_analyses/grid_cohort.py',
    'bf_patient_vs_control.py':                       '4_analyses/08_bf_patient_vs_control.py',
    'E_longitudinal/longitudinal_final_2026-10-02.ipynb': '4_analyses/09_longitudinal.ipynb',
    # 5 - figures (whole_otc_bayes makes Figure 10 and needs final_figures' variables, so it runs second)
    'final_figures.ipynb':                            '5_figures/01_final_figures.ipynb',
    'I_bayes/whole_otc_bayes.ipynb':                  '5_figures/02_whole_otc_bayes.ipynb',
}
STAY = {'params.py', 'sym_pt_params.py'}
OUTDIRS = ['D_liu', 'F_harmonization', 'H_RosenkeRSM', 'C_results', 'logs_combat']
OLD_DIRS = ['A_preprocessing', 'B_analyses', 'D_liu/verified', 'D_liu', 'F_harmonization',
            'H_RosenkeRSM', 'I_bayes', 'E_longitudinal', 'one_last_8.27']
CODE = ('.py', '.sh', '.ipynb', '.fsf')
Q = r"""\\?['"]"""        # a quote, possibly escaped inside notebook JSON


def stop(msg):
    sys.exit(f'STOPPED, nothing changed: {msg}')


# ---------------------------------------------------------------- preconditions
if not (P / 'SOURCES.txt').exists():
    stop(f'{P}/SOURCES.txt not found - run build_pipeline.sh first')
if (P / '1_preprocessing').exists():
    stop('1_preprocessing/ already exists - pipeline/ was renumbered before')
sources = [l.strip() for l in (P / 'SOURCES.txt').read_text().splitlines() if l.strip()]
unmapped = sorted(set(sources) - set(MAP) - STAY)
if unmapped:
    stop(f'files with no new name (they would be left behind): {unmapped}')
missing = [o for o in MAP if not (P / o).exists()]
if missing:
    stop(f'files expected in pipeline/ but not there: {missing}')
if len(set(MAP.values())) != len(MAP):
    stop('two files would get the same new name')


# ---------------------------------------------------------------- path rewrites
def rewrite(text, fname):
    # moved files: relative paths (covers full paths and ${var}/... forms)
    for old in sorted(MAP, key=len, reverse=True):
        text = text.replace(old, MAP[old])
    # moved files: pathlib joins such as GIT / 'D_liu' / 'verified' / '05_stats.py'
    for old in sorted(MAP, key=len, reverse=True):
        op, npart = old.split('/'), MAP[old].split('/')
        new = ' / '.join(f"'{p}'" for p in npart)
        if len(op) == 1:
            pat = r'(/\s*)' + Q + re.escape(op[0]) + Q
            text = re.sub(pat, lambda m: m.group(1) + new, text)
        else:
            pat = Q + re.escape(op[0]) + Q + ''.join(r'\s*/\s*' + Q + re.escape(p) + Q for p in op[1:])
            text = re.sub(pat, new, text)
    # directories that held code, used as directories
    if fname in ('marlene_lmm.py', 'marlene_roi.py'):            # sys.path for grid_cohort
        text = text.replace(f'{ABS}/D_liu/verified', f'{ABS}/4_analyses')
    if fname == 'otc_rsm_rosenke.py':                            # where the TFCE module is looked for
        text = re.sub(Q + 'D_liu' + Q + r'\s*/\s*' + Q + 'verified' + Q, "'2_extraction'", text)
    if fname == 'add_scanner.py':                                # write straight to where ComBat reads
        text = text.replace(f"'{ABS}/sub_info_scanner.csv'",
                            f"'{ABS}/outputs/F_harmonization/sub_info_scanner.csv'")
    # repo-side output folders -> outputs/<same name>
    for d in OUTDIRS:
        text = re.sub(re.escape(ABS) + '/' + d + r'(?=[/\'"\\\s]|$)', f'{ABS}/outputs/{d}', text)
        text = re.sub(r'(' + Q + r')' + d + r'(?=[/\'"\\])', lambda m: m.group(1) + 'outputs/' + d, text)
    return text


originals = {}
for o in list(MAP) + sorted(STAY):
    f = P / o
    originals[o] = f.read_text()

log = []
new_text = {}
for o, txt in originals.items():
    if not o.endswith(CODE):
        continue
    out = rewrite(txt, Path(o).name)
    new_text[o] = out
    if out != txt:
        a, b = txt.splitlines(), out.splitlines()
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                log.append(f'{MAP.get(o, o)}:{i+1}\n  - {x.strip()[:200]}\n  + {y.strip()[:200]}')
        if len(a) != len(b):
            stop(f'line count changed in {o} - nothing written')

# ---------------------------------------------------------------- write, then move
for o, out in new_text.items():
    (P / o).write_text(out)

in_git = subprocess.run(['git', '-C', str(G), 'rev-parse'], capture_output=True).returncode == 0
for old, new in MAP.items():
    (P / new).parent.mkdir(parents=True, exist_ok=True)
    moved = False
    if in_git:
        moved = subprocess.run(['git', '-C', str(P), 'mv', old, new], capture_output=True).returncode == 0
    if not moved:
        shutil.move(str(P / old), str(P / new))

# subject scanner table: into outputs/F_harmonization, link at the repo's old place follows it
for d in OUTDIRS + ['C_results/figures']:
    (P / 'outputs' / d).mkdir(parents=True, exist_ok=True)
scan_old = P / 'F_harmonization' / 'sub_info_scanner.csv'
scan_new = P / 'outputs' / 'F_harmonization' / 'sub_info_scanner.csv'
if scan_old.exists() and not scan_old.is_symlink():
    if not (in_git and subprocess.run(['git', '-C', str(P), 'mv', str(scan_old.relative_to(P)),
                                       str(scan_new.relative_to(P))], capture_output=True).returncode == 0):
        shutil.move(str(scan_old), str(scan_new))
    link = G / 'F_harmonization' / 'sub_info_scanner.csv'
    if link.is_symlink():
        link.unlink()
        link.symlink_to(scan_new)

for d in sorted(OLD_DIRS, key=len, reverse=True):
    dd = P / d
    if dd.exists() and not any(dd.rglob('*')):
        shutil.rmtree(dd)

# ---------------------------------------------------------------- SOURCES and checks
(P / 'SOURCES.txt').write_text(''.join(
    f'{MAP.get(s, s)}  <-  {s}\n' for s in sources))
(P / 'renumber_changes.txt').write_text('\n'.join(log) + '\n')

problems = []
for new in MAP.values():
    if not (P / new).exists():
        problems.append(f'missing after move: {new}')
old_refs = []
code_now = [p for p in P.rglob('*') if p.is_file() and p.suffix in CODE and 'resources' not in p.parts]
root_keys = [o for o in MAP if '/' not in o]
pat_old = re.compile('|'.join(re.escape(o) for o in MAP if '/' in o)
                     + '|' + '|'.join(r"""['"/]""" + re.escape(o) for o in root_keys)
                     + r"""|['"/]A_preprocessing['"/]|['"/]B_analyses['"/]|D_liu/verified"""
                     + r"""|['"/]I_bayes['"/]|['"/]E_longitudinal['"/]|one_last_8\.27""")
for f in code_now:
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if pat_old.search(line):
            old_refs.append(f'{f.relative_to(P)}:{i}: {line.strip()[:160]}')
abs_paths = set()
for f in code_now:
    abs_paths |= set(re.findall(re.escape(ABS) + r'/[A-Za-z0-9_./-]+\.(?:py|sh|fsf|ipynb)', f.read_text()))
dead = sorted(p for p in abs_paths if not Path(p.replace(ABS, str(P), 1)).exists())
# pathlib joins from the repo root, e.g. GIT / '3_harmonization' / 'lib_stats.py'
join_pat = re.compile(r'(?:GIT_DIR|GIT|REPO)\s*/\s*((?:' + Q + r'[^\'"\\]+' + Q + r'\s*/\s*)*'
                      + Q + r'[^\'"\\]+\.(?:py|sh|fsf|ipynb)' + Q + ')')
for f in code_now:
    for m in join_pat.finditer(f.read_text()):
        parts = re.findall(Q + r'([^\'"\\]+)' + Q, m.group(1))
        target = P.joinpath(*[x for part in parts for x in part.split('/')])
        if not target.exists():
            dead.append(f'{f.relative_to(P)}: ' + ' / '.join(parts))

print(f'files moved:                   {len(MAP)}')
print(f'lines changed (all paths):     {len(log)}   -> renumber_changes.txt')
print(f'files missing after the move:  {len(problems)}')
print(f'old names still referenced:    {len(old_refs)}')
for r in old_refs[:20]:
    print('   ', r)
# the TFCE-module fallbacks in otc_rsm_rosenke never existed in the repo either
dead = [d for d in dead if not d.endswith(('02_tfce_analyses_not_as_verified.py', '02_tfce_analyses.py'))]
print(f'script paths that do not exist: {len(dead)}')
for d in dead[:20]:
    print('   ', d)
