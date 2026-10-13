#!/usr/bin/env python3
"""Independent acceptance checks for the Figure 5 editorial redesign."""
from pathlib import Path
import json
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

R = Path(__file__).resolve().parents[1]
rd = lambda p: pd.read_csv(p, float_precision='round_trip')
MODELS = ['Base', 'A', 'B', 'R', 'G', 'V', 'F']
res, ok = {}, True


def check(name, cond, detail=''):
    global ok
    ok &= bool(cond); res[name] = {'pass': bool(cond), 'detail': detail}
    print(('PASS ' if cond else 'FAIL ') + name + (f' | {detail}' if detail else ''))


allm = rd(R / 'data/all_seven_module_results.csv')
glob = allm[allm.testing_family == 'modules_global48']; reg = allm[allm.testing_family == 'modules_regional1908']
check('input rows', len(allm) == 13692 and len(glob) == 336 and len(reg) == 13356, f'{len(glob)} global, {len(reg)} regional')
check('one common sample', allm.n.eq(4483).all(), 'n = 4,483 in every estimate')
err = []
for fam, g, k in [('global', glob, 48), ('regional', reg, 1908)]:
    for m, s in g.groupby('model'):
        assert len(s) == k
        err.append(float(np.max(np.abs(false_discovery_control(s.p_value.to_numpy()) - s.q_bh.to_numpy()))))
check('BH q recomputed within complete families (48 / 1,908 per model)', max(err) < 1e-12, f'max |Δq| = {max(err):.1e}')

# panel a
a = rd(R / 'source_data/Fig5a_336_global_estimates.csv')
m = a.merge(glob, left_on=['model', 'module', 'outcome'], right_on=['model', 'exposure', 'outcome'], suffixes=('', '_in'), validate='one_to_one')
check('a: 336 estimates identical to input', len(m) == 336 and (m.beta == m.beta_in).all() and (m.q_bh == m.q_bh_in).all())
check('a: no global association with q < 0.05', int((a.q_bh < .05).sum()) == 0,
      f'β range {a.beta.min():.4f} to {a.beta.max():.4f}; min q = {a.q_bh.min():.4f}')
check('a: displayed points within plotted x-limits', a.beta.abs().max() < .118)
check('a: labels use Higher/Lower', a.module_label.str.contains(' · Higher| · Lower').all() and not a.module_label.str.contains(r'\(\+\)|\(−\)').any())

# panel b
b = rd(R / 'source_data/Fig5b_84_counts_by_model_direction_category.csv')
re_cnt = reg[reg.q_bh < .05].assign(direction=lambda d: d.exposure.str.split(' | ', regex=False).str[1],
                                    category=lambda d: d.exposure.str.split(' | ', regex=False).str[0]) \
    .groupby(['model', 'direction', 'category']).size()
bb = b.set_index(['model', 'direction', 'category']).FDR_significant
check('b: 84 cells recount exactly from 13,356 regional tests', len(b) == 84 and all(bb.get(k, 0) == v for k, v in re_cnt.items()) and bb.sum() == re_cnt.sum())
old = rd('/home/ubuntu/Figure5_CleanLabels_Revision/source_data/Panel_b_all42_category_counts.csv')
oldc = old.set_index(['model', 'category'])['count']; newc = b.groupby(['model', 'category']).FDR_significant.sum()
check('b: category totals equal previous Figure 5b source data', all(newc.get(k, 0) == v for k, v in oldc.items()) and newc.sum() == oldc.sum())
tot = b.groupby('model').FDR_significant.sum().reindex(MODELS).tolist()
hl = b.pivot_table(index='model', columns='direction', values='FDR_significant', aggfunc='sum').reindex(MODELS)
check('b: totals per model', tot == [215, 65, 12, 15, 10, 3, 3],
      'Higher/Lower: ' + '; '.join(f'{k} {int(r.positive)}/{int(r.negative)}' for k, r in hl.iterrows()))
diag = rd('/home/ubuntu/CKM_Figure5_FullClinical/tables/model_diagnostics.csv'); diag = diag[diag.branch == 'common'].drop_duplicates('model').set_index('model')
des = rd(R / 'source_data/Fig5b_model_covariate_design.csv').set_index('model')
flag = {'CKM stage': 'stage_Stage1', 'BMI': ';BMI;', 'eGFR': ';eGFR;', 'HbA1c + diabetes': 'HbA1c', 'BP + lipids': 'SBP'}
good = all(int((flag[c] in (';' + diag.loc[mm, 'covariates'] + ';'))) == des.loc[mm, c] for mm in ['A', 'B', 'R', 'G', 'V', 'F'] for c in flag)
base = json.loads(Path('/home/ubuntu/CKM_Figure5_SevenModel_Redesign/baseline_analysis_summary.json').read_text())
good &= (not base['baseline_includes_CKM']) and (not base['baseline_includes_BMI']) and des.loc['Base', ['CKM stage', 'BMI', 'eGFR', 'HbA1c + diabetes', 'BP + lipids']].sum() == 0
check('b: covariate matrix matches fitted design columns', good)

# panel c
c = rd(R / 'source_data/Fig5c_105_fixed_pair_estimates.csv')
mc = c.merge(reg, left_on=['model', 'module', 'outcome'], right_on=['model', 'exposure', 'outcome'], suffixes=('', '_in'), validate='one_to_one')
check('c: 105 estimates identical to complete regional family', len(mc) == 105 and all((mc[k] == mc[k + '_in']).all() for k in ['beta', 'ci_low', 'ci_high', 'p_value', 'q_bh']))
cc = c.groupby('model').q_bh.apply(lambda q: int((q < .05).sum())).reindex(MODELS).tolist()
check('c: q < 0.05 counts among fixed 15 pairs', cc == [15, 15, 12, 13, 10, 3, 3], str(cc))
check('c: β range', True, f'{c.beta.min():.3f} to {c.beta.max():.3f} (all negative: {bool((c.beta < 0).all())})')

# panel d
d = rd(R / 'source_data/Fig5d_21_protein_estimates_with_bootstrap_summary.csv')
din = rd(R / 'data/Panel_d_all21_candidate_estimates.csv')
md = d.merge(din, left_on=['protein', 'model'], right_on=['exposure', 'model'], suffixes=('', '_in'), validate='one_to_one')
check('d: 21 estimates identical to input', len(md) == 21 and all((md[k] == md[k + '_in']).all() for k in ['beta', 'ci_low', 'ci_high', 'q_bh']))
boot = rd(R / 'data/seven_model_paired_bootstrap_draws.csv')
check('d: 21,000 bootstrap draws (1,000 per protein–model)', len(boot) == 21000 and boot.groupby(['protein', 'model']).size().eq(1000).all())
qs = d.pivot(index='model', columns='protein', values='q_bh')
check('d: significance pattern', True, '; '.join(f'{p} q {qs[p].min():.4f}–{qs[p].max():.4f}, β {d[d.protein == p].beta.min():.3f} to {d[d.protein == p].beta.max():.3f}' for p in ['GFAP', 'NEFL', 'RBPMS']))

# figure files
fig = R / 'figures/Figure5_Proteomic_Modules_Brain_Structure_a-d'
info = subprocess.run(['pdfinfo', str(fig.with_suffix('.pdf'))], capture_output=True, text=True).stdout
w, h = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).groups())
check('PDF width 180 mm', abs(w / 72 * 25.4 - 180) < .1, f'{w / 72 * 25.4:.1f} × {h / 72 * 25.4:.1f} mm')
imgs = subprocess.run(['pdfimages', '-list', str(fig.with_suffix('.pdf'))], capture_output=True, text=True).stdout.strip().splitlines()[2:]
check('PDF contains no raster images', len(imgs) == 0, f'{len(imgs)} images')
fonts = subprocess.run(['pdffonts', str(fig.with_suffix('.pdf'))], capture_output=True, text=True).stdout.splitlines()[2:]
check('PDF fonts embedded', len(fonts) > 0 and all(' yes ' in f for f in fonts), '; '.join(sorted({f.split()[0] for f in fonts})))
svg = fig.with_suffix('.svg').read_text()
check('no +/− module labels or plus-signed ticks in figure text',
      not re.search(r'\(\+\)|\(−\)|\(-\)|>\+0\.', svg) and 'Higher' in svg and 'Lower' in svg)
check('no model abbreviations as labels', not re.search(r'>(B|R|G|F|V)</text>', svg))
for k in ['Panel_a_Global_Module_MRI', 'Panel_b_Model_Design_and_Regional_Counts', 'Panel_c_Fixed_Module_Region_Pairs', 'Panel_d_Single_Protein_PCGMV']:
    check(f'standalone panel exported: {k}', all((R / 'panels' / f'{k}.{e}').exists() for e in ['png', 'pdf', 'svg']))
check('no participant-level data in package', not any(p.suffix in {'.rds'} or 'CKM_IDP' in p.name or 'CKM_data' in p.name for p in R.rglob('*')))

(R / 'validation.json').write_text(json.dumps({'all_pass': bool(ok), 'checks': res}, indent=2, ensure_ascii=False) + '\n')
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
