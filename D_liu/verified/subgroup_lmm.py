import importlib.util, sys
import numpy as np, pandas as pd

P = '/user_data/csimmon2/git_repos/sym_pt/D_liu/verified/05_stats_harmony.py'
spec = importlib.util.spec_from_file_location('sh', P)
sh = importlib.util.module_from_spec(spec)
sys.modules['sh'] = sh
spec.loader.exec_module(sh)

UNI = [n for n in dir(sh) if 'UNI' in n or 'UNIVAR' in n]
print('univariate constants found:', UNI)

# ---- sum-selectivity ----
uni = sh.apply_exclusions(pd.read_csv(getattr(sh, UNI[0])))
uni = sh.select_sessions(uni, pt_rule='last')
uni = uni[uni['sum_selec_norm'] > 0].copy()
uni['log_sumsel'] = np.log10(uni['sum_selec_norm'])

# ---- distinctiveness ----
rsa = sh.apply_exclusions(pd.read_csv(sh.RSA_CSV))
rsa_summary = rsa.drop(columns=['pair', 'fisher_r']).drop_duplicates()
rsa_summary = sh.select_sessions(rsa_summary, pt_rule='last')

for name, frame, col in [('sum_selectivity', uni, 'log_sumsel'),
                         ('distinctiveness', rsa_summary, 'liu_distinctiveness')]:
    pt = frame[(frame['group'] == 'OTC') &
               (frame['category'].isin(sh.PRIMARY_ROIS))]
    n_lh = pt[pt['hemi'] == 'l']['subject_id'].nunique()
    n_rh = pt[pt['hemi'] == 'r']['subject_id'].nunique()
    chi, dfree, p, mse = sh.lmm_omnibus(pt, col, 'category', 'hemi')
    print(f'{name}: n_LH={n_lh} n_RH={n_rh} chi2({dfree})={chi:.3f} p={p:.4f} mse={mse:.4f}')
