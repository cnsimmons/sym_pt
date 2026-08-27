#!/usr/bin/env python
"""
report_marlene.py — render the three result tables as readable markdown.

Inputs (all optional; whatever exists is included):
  grid.csv   90 rows   permutation OLS, 5 specs x 6 comparisons x 3 measures
  lmm.csv    18 rows   mixed-model omnibus, 6 comparisons x 3 measures
  roi.csv    per ROI (and per pair for geometry), 6 comparisons x 3 measures

Sections: PRIMARY (comparisons 1-3), SUPPLEMENTAL (4-6), PER-ROI.

Usage
  python report_marlene.py --out marlene_table.md
"""
import argparse
from pathlib import Path
import pandas as pd

MEASURE_LABEL = {
    'peak_z': 'peak_z  (univariate peak selectivity)',
    'distinctiveness': 'RSA distinctiveness',
    'geometry': 'between-category pattern similarity  (6 pairs)',
}
MEASURE_ORDER = {'peak_z': 0, 'distinctiveness': 1, 'geometry': 2}
SPEC_ORDER = {'binA': 0, 'binB': 1, 'binC': 2, 'cont': 3, 'omnibus': 9}


def esc(s):
    return str(s).replace('|', '\\|')


def stars(p):
    if pd.isna(p):
        return ''
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def fmt_p(p):
    if pd.isna(p):
        return '—'
    return f'{p:.1e}' if p < .0001 else f'{p:.4f}'


def num(v, fmt='{:+.3f}'):
    return '—' if pd.isna(v) else fmt.format(v)


def spec_key(s):
    return (SPEC_ORDER.get(str(s).split()[0], 8), str(s))


def grid_lmm_block(L, g, l, comps, title, blurb):
    L.append(f'# {title}')
    L.append('')
    L.append(blurb)
    L.append('')
    for measure in sorted(set(g.measure) | set(l.measure),
                          key=lambda m: MEASURE_ORDER.get(m, 9)):
        L.append('---')
        L.append('')
        L.append(f'## {MEASURE_LABEL.get(measure, measure)}')
        L.append('')
        for comp in comps:
            gm = g[(g.measure == measure) & (g.comparison == comp)]
            lm = l[(l.measure == measure) & (l.comparison == comp)]
            if not len(gm) and not len(lm):
                continue
            src = gm if len(gm) else lm
            name = src.comparison_name.iloc[0]
            na, nb = int(src.n_group_a.iloc[0]) if 'n_group_a' in src else \
                int(src.n_a.iloc[0]), int(src.n_group_b.iloc[0]) \
                if 'n_group_b' in src else int(src.n_b.iloc[0])
            note = ''
            if comp == 3:
                note = '  *(patient-group × category; no control reference)*'
            elif comp in (4, 5):
                note = '  *(crossed; confounded with normal asymmetry)*'
            elif comp == 6:
                note = '  *(within subject)*'
            L.append(f'**{comp}. {esc(name)}** — nA={na}, nB={nb}{note}')
            L.append('')
            L.append('| specification | beta | χ² (df) | p | |')
            L.append('|---|---|---|---|---|')
            for _, r in gm.sort_values('spec', key=lambda s: s.map(spec_key)).iterrows():
                b = '**' if (not pd.isna(r.p) and r.p < .05) else ''
                L.append(f'| {esc(r.spec)} | {b}{num(r.beta)}{b} | — | '
                         f'{b}{fmt_p(r.p)}{b} | {stars(r.p)} |')
            for _, r in lm.iterrows():
                b = '**' if (not pd.isna(r.p) and r.p < .05) else ''
                cv = '—' if pd.isna(r.chi2) else f'{r.chi2:.2f} ({int(r.df)})'
                conv = '' if r.get('converged', True) else '  ⚠ no convergence'
                L.append(f'| LMM omnibus category × group{conv} | — | {b}{cv}{b} '
                         f'| {b}{fmt_p(r.p)}{b} | {stars(r.p)} |')
            L.append('')
            if len(lm):
                r = lm.iloc[0]
                bits = []
                if 'age_beta' in r and not pd.isna(r.age_beta):
                    sig = '  **sig**' if r.age_p < .05 else ''
                    bits.append(f'age β={r.age_beta:+.3f} p={fmt_p(r.age_p)}{sig}')
                if 'surg_beta' in r and not pd.isna(r.surg_beta):
                    sig = '  **sig**' if r.surg_p < .05 else ''
                    bits.append(f'surgery (OTC vs Hemi) β={r.surg_beta:+.3f} '
                                f'p={fmt_p(r.surg_p)}{sig}')
                if bits:
                    L.append('LMM covariates: ' + ';  '.join(bits))
                    L.append('')


def roi_block(L, roi):
    L.append('---')
    L.append('')
    L.append('# Per-ROI table')
    L.append('')
    L.append('The pooled grid above collapses the four ROIs into one '
             'interaction, so an effect carried by a single ROI does not '
             'appear there. This section reports each ROI separately, and each '
             'category pair separately for pattern similarity.')
    L.append('')
    L.append('`diff` = group B − group A on the oriented scale (higher = more '
             'selective / more distinct / more separated). The manuscript '
             'reports distinctiveness and geometry on the raw similarity '
             'scale, so its signs are the reverse of these.')
    L.append('')
    for measure in sorted(roi.measure.unique(),
                          key=lambda m: MEASURE_ORDER.get(m, 9)):
        L.append(f'## {MEASURE_LABEL.get(measure, measure)}')
        L.append('')
        m = roi[roi.measure == measure]
        for comp in sorted(m.comparison.unique()):
            c = m[m.comparison == comp]
            role = c.role.iloc[0]
            tag = '' if role == 'primary' else '  *[supplemental]*'
            L.append(f'**{comp}. {esc(c.comparison_name.iloc[0])}**{tag}')
            L.append('')
            haspair = c['pair'].astype(str).str.len().gt(0).any()
            if haspair:
                L.append('| ROI | pair | diff | d | p | q | |')
                L.append('|---|---|---|---|---|---|---|')
            else:
                L.append('| ROI | diff | d | p | q | |')
                L.append('|---|---|---|---|---|---|')
            for _, r in c.iterrows():
                sig = stars(r.q_fdr) if not pd.isna(r.q_fdr) else ''
                b = '**' if sig else ''
                cells = [esc(r.roi)]
                if haspair:
                    cells.append(esc(r['pair']) if str(r['pair']) else '')
                cells += [f'{b}{num(r["diff"])}{b}',
                          num(r.cohen_d, '{:+.2f}'),
                          fmt_p(r.p),
                          f'{b}{fmt_p(r.q_fdr)}{b}', sig]
                L.append('| ' + ' | '.join(cells) + ' |')
            L.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='grid.csv')
    ap.add_argument('--lmm', default='lmm.csv')
    ap.add_argument('--roi', default='roi.csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    g = pd.read_csv(args.grid) if Path(args.grid).exists() else pd.DataFrame()
    l = pd.read_csv(args.lmm) if Path(args.lmm).exists() else pd.DataFrame()
    roi = pd.read_csv(args.roi) if Path(args.roi).exists() else pd.DataFrame()
    if len(l) and 'model' in l:
        l = l[l.model == 'primary']

    L = []
    if len(g) or len(l):
        grid_lmm_block(
            L, g, l, [1, 2, 3], 'Marlene\'s grid + LMM — primary',
            'Three comparisons × five specifications (permutation OLS, 1 df) '
            'plus the mixed-model omnibus on the full category factor. '
            'All measures oriented so higher = more selective / more distinct. '
            'beta positive = the first-named category set is relatively LESS '
            'affected in group B.')
        L.append('---')
        L.append('')
        grid_lmm_block(
            L, g, l, [4, 5, 6], 'Supplemental — crossed and paired',
            'Comparisons 4 and 5 pit a patient\'s intact hemisphere against '
            'the OPPOSITE control hemisphere, so they confound resection with '
            'normal hemispheric asymmetry. Comparison 6 measures that '
            'asymmetry alone: the same controls, both hemispheres, paired '
            'within subject. Read 4 and 5 against 6.')
    if len(roi):
        roi_block(L, roi)

    L.append('---')
    L.append('')
    L.append('TFCE has no subject × category value, so it cannot take this '
             'form. Report the existing cluster table instead.')
    L.append('')
    L.append('`* p<.05   ** p<.01   *** p<.001`')

    txt = '\n'.join(L)
    if args.out:
        Path(args.out).write_text(txt + '\n')
        print(f'wrote {args.out}')
    else:
        print(txt)


if __name__ == '__main__':
    main()
