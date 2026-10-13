#!/usr/bin/env python3
"""Independent validation of Figure 4 outputs against the source tables."""
from pathlib import Path
import subprocess, json, sys
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
T = ['stage_f1', 'stage_f2', 'stage_f3']
ok = []
def check(name, cond, detail=''):
    ok.append(bool(cond)); print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))

src = pd.read_csv(R / 'data/05_protein_ckm_stage_contrasts.csv', float_precision='round_trip')
a = pd.read_csv(R / 'source_data/Fig4a_all_8769_protein_contrasts.csv', float_precision='round_trip')
m = src.merge(a, on=['protein', 'contrast'], suffixes=('', '_fig'))
check('Fig4a rows = 8,769 (2,923 x 3)', len(a) == 8769 and len(m) == 8769)
check('beta identical to source', np.array_equal(m.beta.values, m.beta_fig.values))
check('q identical to source', np.array_equal(m.p_fdr.values, m.p_fdr_fig.values))
exp = {'stage_f1': (2352, 1821, 531), 'stage_f2': (2428, 1920, 508), 'stage_f3': (2256, 2054, 202)}
for t, (n, hi, lo) in exp.items():
    g = a[a.contrast == t]
    check(f'{t} FDR counts {n}/{hi}/{lo}', (g.p_fdr < .05).sum() == n and (g.direction == 'higher').sum() == hi and (g.direction == 'lower').sum() == lo)
rec = a[a.neglog10P_source != 'source_P']
check('recomputed -log10P only where source P < 1e-290', (rec.p_value < 1e-290).all() and len(rec) == 185, f'n = {len(rec)}; source P = 0: {(a.p_value == 0).sum()}')
check('recomputed values finite and >= 289', np.isfinite(rec.neglog10P).all() and (rec.neglog10P >= 289).all(), f'range {rec.neglog10P.min():.1f}-{rec.neglog10P.max():.1f}')
fin = a[a.neglog10P_source == 'source_P']
check('non-underflow y equals -log10(source P)', np.allclose(fin.neglog10P, -np.log10(fin.p_value)))
lab = pd.read_csv(R / 'source_data/Fig4a_36_labels_top6_abs_beta.csv')
lab12 = pd.read_csv(R / 'source_data/SuppFig4a_72_labels_top12_abs_beta.csv')
rule_ok = True
for k, L in [(6, lab), (12, lab12)]:
    for t in T:
        g = a[(a.contrast == t) & (a.p_fdr < .05)]
        for dr, sgn in [('higher', 1), ('lower', -1)]:
            h = g[g.beta * sgn > 0].assign(ab=lambda d: d.beta.abs()).sort_values(['ab', 'protein'], ascending=[False, True]).head(k)
            rule_ok &= list(h.protein) == list(L[(L.contrast == t) & (L.direction == dr)].protein)
check('labels follow top-|beta| rule (36 main, 72 supplementary)', rule_ok and len(lab) == 36 and len(lab12) == 72)
it = pd.read_csv(R / 'source_data/Fig4b_intersections_by_direction.csv')
mem = pd.read_csv(R / 'source_data/Fig4b_membership_2923_proteins.csv', dtype={'pattern': str})
check('membership covers 2,923 proteins', len(mem) == 2923)
check('union of FDR sets = 2,629; none = 294', it.total.sum() == 2629 and (mem.pattern == '000').sum() == 294)
s111 = it[it.pattern == 'S111'].iloc[0]
check('all-three 1,977 = 1,656 higher + 178 lower + 143 discordant', (s111.total, s111.all_higher, s111.all_lower, s111.discordant) == (1977, 1656, 178, 143))
check('intersection components sum to totals', (it[['all_higher', 'all_lower', 'discordant']].sum(axis=1) == it.total).all())
gr = pd.read_csv(R / 'source_data/Fig4b_shared_direction_consistent_1834_betas.csv', float_precision='round_trip')
check('gradient set = 1,834 proteins', len(gr) == 1834 and set(gr.direction_class.value_counts().items()) == {('all_higher', 1656), ('all_lower', 178)})
b = src.pivot(index='protein', columns='contrast', values='beta')
check('gradient betas identical to source', np.allclose(gr.set_index('protein')[T].values, b.loc[gr.protein, T].values, rtol=0, atol=0))
c = pd.read_csv(R / 'source_data/Fig4c_36_category_tiles.csv')
check('Fig4c 36 tiles, family = 29 per set', len(c) == 36 and (c.testing_family_size == 29).all())
d = pd.read_csv(R / 'source_data/Fig4d_54_pathway_cells.csv')
check('Fig4d 9 pathways x 6 sets, family = 1,140 per set', len(d) == 54 and d.pathway.nunique() == 9 and (d.testing_family_size == 1140).all())
pdf = R / 'figures/Figure4_CKM_Proteomic_Profiles_a-d.pdf'
info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
w, h = [float(x) for x in info.split('Page size:')[1].split('pts')[0].split('x')]
check('composite width 180 mm', abs(w / 72 * 25.4 - 180) < .1, f'{w / 72 * 25.4:.1f} x {h / 72 * 25.4:.1f} mm')
imgs = subprocess.run(['pdfimages', '-list', str(pdf)], capture_output=True, text=True).stdout.strip().splitlines()[2:]
check('composite PDF contains no raster images (fully vector)', len(imgs) == 0, f'{len(imgs)} images')
fonts = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True).stdout
check('fonts embedded TrueType (Liberation Sans)', 'LiberationSans' in fonts and 'Type 3' not in fonts)
print('\nKey values for legend/results')
print(c[c.q_bh < .05][['category', 'contrast', 'selection', 'fold_enrichment', 'q_bh']].round(4).to_string(index=False))
print(pd.read_csv(R / 'source_data/Fig4b_shared_gradient_summary.csv').round(3).to_string(index=False))
print(pd.read_csv(R / 'source_data/Fig4a_FDR_boundaries.csv').round(3).to_string(index=False))
print(f'\n{sum(ok)}/{len(ok)} checks passed')
sys.exit(0 if all(ok) else 1)
