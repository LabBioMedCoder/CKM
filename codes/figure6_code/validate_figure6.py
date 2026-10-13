#!/usr/bin/env python3
"""Independent acceptance checks for Figure 6 (analysis integrity, figure-data agreement, privacy, vector output)."""
import sys, json, subprocess, re, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, '/home/ubuntu/Figure6_Prediction/code')
from f6utils import *
warnings.filterwarnings('ignore')
P = Path(R); RES = P / 'results'; SD = P / 'source_data'; FIG = P / 'figures'
ok = []
def check(name, cond, detail=''):
    ok.append(bool(cond)); print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))

a = pd.read_parquet(P / 'private/analysis.parquet'); n = len(a)
check('analytic sample n = 37,713', n == 37713, n)
check('stage counts 8,139 / 15,672 / 12,927 / 975', a.stage.value_counts().sort_index().tolist() == [8139, 15672, 12927, 975])
h1 = pd.read_csv(RES / '04_H1_associations.csv'); prim = h1[h1.model == 'M2: CKM-PS'].set_index('outcome')
check('event counts in H1 table equal analysis data', all(int(a[o].sum()) == int(prim.loc[o, 'events']) for o in OUT))

# outer folds: partition, no train/test overlap, out-of-fold scores only
part, overlap = True, True
for rep in range(10):
    seen = np.zeros(n, int)
    for k in range(5):
        z = np.load(P / f'private/score_folds/rep{rep}_fold{k}.npz')
        seen[z['te']] += 1
        overlap &= len(np.intersect1d(z['tr'], z['te'])) == 0 and len(z['tr']) + len(z['te']) == n
    part &= (seen == 1).all()
check('10 repeats x 5 outer folds partition the sample exactly once per repeat', part)
check('no participant in both training and test fold', overlap)
s = pd.read_parquet(P / 'private/oof_scores.parquet')
check('CKM-PS orthogonal to age (|r| < 0.01)', abs(np.corrcoef(s.ckm_ps, a.Age)[0, 1]) < .01, f"{np.corrcoef(s.ckm_ps, a.Age)[0, 1]:.4f}")
check('CKM-PS orthogonal to sex (|r| < 0.01)', abs(np.corrcoef(s.ckm_ps, a.Sex)[0, 1]) < .01)

# fold metrics completeness and independent recomputation of the primary increment
fm = pd.read_csv(RES / '03_fold_metrics.csv')
check('fold metrics: 9 outcomes x (50 x 7 + 5) = 3,195 rows', len(fm) == 3195, len(fm))
check('test-fold sizes sum to n within each outcome x repeat x model',
      (fm.groupby(['outcome', 'rep', 'model']).n_test.sum() == n).all())
inc = pd.read_csv(RES / '05_increments.csv')
mx = 0
for o in OUT:
    g = fm[fm.outcome == o]; d = (g[g.model == 'M3'].set_index(['rep', 'fold']).C - g[g.model == 'M2'].set_index(['rep', 'fold']).C)
    mx = max(mx, abs(d.mean() - inc[(inc.outcome == o) & (inc.new == 'M3') & (inc.ref == 'M2')].dC.iloc[0]))
check('primary dC recomputed from fold metrics', mx < 1e-12, f'max diff {mx:.1e}')
check('all primary dC 95% intervals include 0', ((inc[inc.new.eq('M3') & inc.ref.eq('M2')].dC_lo < 0) & (inc[inc.new.eq('M3') & inc.ref.eq('M2')].dC_hi > 0)).all())

# H1 cross-check with lifelines (dementia)
try:
    from lifelines import CoxPHFitter
    o = 'Dementia'; idx = np.arange(n)
    X, names = design(a, idx, idx[:2], o, 'M2'); zps = (s.ckm_ps - s.ckm_ps.mean()) / s.ckm_ps.std()
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])]); df['ps'] = zps.to_numpy()
    df['T'] = a[o + '_time'].to_numpy(); df['E'] = a[o].to_numpy()
    cph = CoxPHFitter().fit(df, 'T', 'E', fit_options=dict(step_size=.5))
    hr = float(np.exp(cph.params_['ps']))
    check('H1 dementia HR reproduced by lifelines', abs(hr - prim.loc[o, 'HR']) < 2e-3, f'{hr:.4f} vs {prim.loc[o, "HR"]:.4f}')
except Exception as e:
    check('H1 dementia HR reproduced by lifelines', False, repr(e)[:120])

# figure source data agree with result tables
c = pd.read_csv(SD / 'Fig6c_9_HR_per_SD_M2.csv').set_index('outcome')
check('Fig6c source = H1 table', np.allclose(c.loc[OUT, 'HR'], prim.loc[OUT, 'HR'], rtol=0, atol=1e-12))
dd = pd.read_csv(SD / 'Fig6d_27_deltaC.csv'); check('Fig6d: 27 estimates (9 outcomes x 3 comparisons)', len(dd) == 27)
e = pd.read_csv(SD / 'Fig6e_risk_grid_108_cells_and_36_differences.csv')
check('Fig6e: 108 cells + 36 differences', (e.tertile.astype(str) != 'T3-T1').sum() == 108 and (e.tertile == 'T3-T1').sum() == 36)
bal = pd.read_csv(RES / '04_H3_tertile_balance.csv')
check('gap tertiles balanced on age within stage (max diff < 1 y)', bal.groupby('stage').mean_age.agg(lambda v: v.max() - v.min()).max() < 1.0)
w = pd.read_csv(SD / 'Fig6b_all_2923_protein_weights.csv'); top = pd.read_csv(SD / 'Fig6b_16_displayed_proteins.csv')
exp = set(w[w.mean_weight > 0].nlargest(8, 'mean_weight').protein) | set(w[w.mean_weight < 0].nsmallest(8, 'mean_weight').protein)
check('Fig6b displayed proteins follow top-8 Higher/Lower rule', set(top.protein) == exp)
bar = pd.read_csv(SD / 'Fig6b_stable_set_composition.csv'); check('Fig6b stable-set composition 194/72/49 = 315', bar.n.tolist() == [194, 72, 49])
g = pd.read_csv(SD / 'Fig6g_8_MRI_bridge_estimates.csv'); check('Fig6g: 8 estimates, n = 4,456', len(g) == 8 and (g.n == 4456).all())

# privacy: no participant-level export
priv = True; big = []
for f in SD.glob('*.csv'):
    x = pd.read_csv(f, nrows=5)
    if any(cn.lower() in ('eid', 'id', 'participant') for cn in x.columns): priv = False
    if sum(1 for _ in open(f)) - 1 >= 37713: big.append(f.name)
check('source data contain no participant identifiers', priv)
check('no source data file has participant-level row count', not big, ','.join(big))

# vector PDF checks
pdf = FIG / 'Figure6_CKM_Proteomic_Score_Neuropsychiatric_Risk_a-g.pdf'
info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
wpt, hpt = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).groups())
check('composite width = 180 mm', abs(wpt / 72 * 25.4 - 180) < .5, f'{wpt / 72 * 25.4:.1f} x {hpt / 72 * 25.4:.1f} mm')
imgs = subprocess.run(['pdfimages', '-list', str(pdf)], capture_output=True, text=True).stdout.strip().split('\n')[2:]
check('composite PDF contains no raster images', len([l for l in imgs if l.strip()]) == 0, f'{len(imgs)} images')
fonts = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True).stdout
check('fonts embedded as TrueType (Liberation Sans)', 'LiberationSans' in fonts and 'Type 3' not in fonts)
svg = (FIG / 'Figure6_CKM_Proteomic_Score_Neuropsychiatric_Risk_a-g.svg').read_text()
check('direction wording Higher/Lower (no +/− module labels)', 'Higher' in svg and 'Lower' in svg and '(+)' not in svg)
print(f'\n{sum(ok)}/{len(ok)} checks passed')
json.dump(dict(passed=sum(ok), total=len(ok)), open(P / 'validation_summary.json', 'w'))
