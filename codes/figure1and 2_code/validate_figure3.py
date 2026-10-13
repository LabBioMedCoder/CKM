#!/usr/bin/env python3
"""Independent acceptance checks for the Figure 3 editorial redesign.
Re-reads the source workbook, recomputes BH q values, and checks every exported Source Data
table, the 3-D colour assignments, atlas coverage and the vector PDF properties.
Usage: python3 code/validate_figure3.py
"""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from f3style import beta_rgb  # noqa: E402

SRC = Path('/home/ubuntu/upload/CKM脑区关联分析_完整结果.xlsx')
SD = ROOT / 'source_data'
CON = ['Stage1 vs Stage0', 'Stage2 vs Stage0', 'Stage3 vs Stage0']
ok = []


def check(cond, msg):
    ok.append(bool(cond))
    print(('PASS ' if cond else 'FAIL ') + msg)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


x = pd.read_excel(SRC, sheet_name='完整分类结果')
m1 = x[x.model.eq('M1')].copy()
i = m1[m1.p_fdr_idp159.notna()].copy()
g = pd.read_excel(SRC, sheet_name='全局关联')
g = g[g.contrast.isin(CON)]

print('== 1. Source workbook and FDR families')
check(m1.n.eq(35508).all(), 'M1 complete-case n = 35,508 for every estimate')
check(len(i) == 477 and i.outcome.nunique() == 159, '159 regional IDPs x 3 contrasts = 477 estimates')
err = max(np.abs(bh(d.p_value) - d.p_fdr_idp159).max() for _, d in i.groupby('contrast'))
check(err < 1e-12, f'q (159 IDPs) = BH within each contrast; max abs error {err:.1e}')
err = max(np.abs(bh(d.p_value) - d.p_fdr_family).max() for _, d in g.groupby('contrast'))
check(len(g) == 12 and err < 1e-12, f'global q = BH over 4 outcomes within each contrast; max abs error {err:.1e}')
z = (i.beta_std / i.se_std).abs()
check(np.allclose(i.ci_low_std, i.beta_std - 1.959964 * i.se_std, atol=2e-4), '95% CI = beta +/- 1.96 x HC3 SE')

print('== 2. Panel a (global)')
a = pd.read_csv(SD / 'Fig3a_12_global_estimates.csv')
mm = a.merge(g, on=['outcome', 'contrast'], suffixes=('', '_src'))
check(len(mm) == 12 and np.allclose(mm.beta_std, mm.beta_std_src) and np.allclose(mm.q_fdr_global4, mm.p_fdr_family),
      'Fig3a beta/CI/q identical to 全局关联 sheet')
check((mm.fdr_significant == (mm.p_fdr_family < .05)).all(), 'Fig3a filled/open coding = q < 0.05')
for ct in CON:
    s = mm[mm.contrast.eq(ct)]
    print('   ', ct, 'FDR-significant global outcomes:', ', '.join(s[s.fdr_significant].outcome), f'({s.fdr_significant.sum()}/4)')

print('== 3. Panel b (159-IDP landscape)')
b = pd.read_csv(SD / 'Fig3b_477_regional_IDP_estimates.csv')
check(len(b) == 477 and np.allclose(b.sort_values(['outcome', 'contrast']).beta_std.values,
                                    i.sort_values(['outcome', 'contrast']).beta_std.values), 'Fig3b 477 estimates identical to workbook')
cnt = pd.read_csv(SD / 'Fig3b_FDR_counts_by_family_and_stage.csv')
exp_tot = {CON[0]: (41, 7), CON[1]: (106, 9), CON[2]: (87, 5)}
for ct, (lo, hi) in exp_tot.items():
    d = i[i.contrast.eq(ct)]
    nl = int(((d.p_fdr_idp159 < .05) & (d.beta_std < 0)).sum())
    nh = int(((d.p_fdr_idp159 < .05) & (d.beta_std > 0)).sum())
    r = cnt[(cnt.family == 'All regional IDPs') & (cnt.contrast == ct)].iloc[0]
    check((nl, nh) == (lo, hi) == (r.fdr_lower, r.fdr_higher), f'{ct}: lower/higher = {nl}/{nh} (total {nl + nh}/159)')
fam = cnt[cnt.family != 'All regional IDPs']
for ct in CON:
    f = fam[fam.contrast.eq(ct)]
    a_ = cnt[(cnt.family == 'All regional IDPs') & (cnt.contrast == ct)].iloc[0]
    check(f.fdr_lower.sum() == a_.fdr_lower and f.fdr_higher.sum() == a_.fdr_higher and f.n_idps.sum() == 159,
          f'{ct}: family counts sum to the all-IDP row')
print(cnt[['family', 'contrast', 'n_idps', 'fdr_lower', 'fdr_higher']].to_string(index=False))

print('== 4. Panel c (cortical volume 3-D)')
c = pd.read_csv(SD / 'Fig3c_198_cortical_volume_estimates_rendered.csv')
check(len(c) == 198 and c.region.nunique() == 33, '198 cortical-volume estimates (33 DK regions x 2 hemispheres x 3 stages)')
check((c.coloured == (c.q_fdr_159IDPs < .05)).all(), 'coloured parcels = q < 0.05 (159-IDP family); others grey')
rgb_ok = all(rr == ('#%02X%02X%02X' % tuple((beta_rgb(bb) * 255).round().astype(int))) for rr, bb, q in
             zip(c.render_rgb, c.beta_std, c.q_fdr_159IDPs) if q < .05)
check(rgb_ok, 'rendered colour of every significant parcel = beta_rgb(beta)')
for hemi, code in (('left', 'L'), ('right', 'R')):
    names = set(nib.load(ROOT / 'atlas' / f'{code}_DK_164k_dseg.label.gii').labeltable.get_labels_as_dict().values())
    miss = set(c[c.hemisphere.eq(hemi)].region) - names
    check(not miss, f'{hemi} hemisphere: all 33 workbook regions present in fsaverage-164k Desikan labels')
for s, ct in enumerate(CON, 1):
    d = c[c.stage.eq(s)]
    lo = d[(d.q_fdr_159IDPs < .05) & (d.beta_std < 0)]
    hi = d[(d.q_fdr_159IDPs < .05) & (d.beta_std > 0)]
    print(f'    Stage {s}: {len(lo)} lower, {len(hi)} higher; higher =',
          ', '.join(f'{r.hemisphere[0].upper()} {r.region} ({r.beta_std:.3f})' for r in hi.itertuples()) or 'none')
ent = c[c.region.eq('entorhinal')]
check((ent.q_fdr_159IDPs >= .05).all(), 'entorhinal volume q >= 0.05 at every stage, both hemispheres')

print('== 5. Panels d and e (subcortical / ventricular)')
dd = pd.read_csv(SD / 'Fig3d_48_subcortical_estimates_rendered.csv')
check(len(dd) == 48 and (dd.coloured == (dd.q_fdr_159IDPs < .05)).all(), 'Fig3d 48 structure-stage colours = q < 0.05')
src = i.set_index(['outcome', 'contrast']).beta_std
check(np.allclose([src[(o, CON[s - 1])] for o, s in zip(dd.outcome, dd.stage)], dd.beta_std), 'Fig3d beta identical to workbook')
e = pd.read_csv(SD / 'Fig3e_54_subcortical_ventricular_estimates.csv')
check(len(e) == 54 and e.outcome.nunique() == 18, 'Fig3e 18 measures x 3 stages = 54 estimates')
check(np.allclose([src[(o, ct)] for o, ct in zip(e.outcome, e.contrast)], e.beta_std), 'Fig3e beta identical to workbook')
check((e.fdr_significant == (e.q_fdr_159IDPs < .05)).all(), 'Fig3e filled/open coding = q < 0.05')
for ct in CON:
    s = e[e.contrast.eq(ct) & ~e.fdr_significant]
    print('   ', ct, 'q >= 0.05:', ', '.join(f'{r.structure} {r.hemisphere}' for r in s.itertuples()))

print('== 6. Composite PDF')
pdf = ROOT / 'figures' / 'Figure3_CKM_Brain_Structure_a-e.pdf'
info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
w, h = [float(v) for v in info.split('Page size:')[1].split('pts')[0].split('x')]
check(abs(w / 72 * 25.4 - 180) < .1, f'page width = {w / 72 * 25.4:.1f} mm (double column); height = {h / 72 * 25.4:.1f} mm')
imgs = subprocess.run(['pdfimages', '-list', str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
rgb = [l.split() for l in imgs if l.split()[2] == 'image']
check(len(rgb) == 16, f'raster objects = {len(rgb)} (12 cortical + 4 subcortical renders only; text, axes, colour bar vector)')
check(all(int(r[12]) >= 590 for r in rgb), 'embedded 3-D renders >= 600 ppi at print size')
fonts = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True).stdout
check('Type 3' not in fonts and 'LiberationSans' in fonts, 'fonts embedded as TrueType (editable text), Liberation Sans')
svg = (ROOT / 'figures' / 'Figure3_CKM_Brain_Structure_a-e.svg').read_text()
check(all(k in svg for k in ['Lower', 'Higher', 'Stage 1', 'Stage 2', 'Stage 3']), 'SVG keeps live text labels')

print(f'\n{sum(ok)}/{len(ok)} checks passed')
sys.exit(0 if all(ok) else 1)
