#!/usr/bin/env python3
"""Figure 3 editorial redesign: CKM stage and global/regional structural MRI phenotypes.

Panels a-e are drawn as standalone figures by the same functions that build the native vector
composite (180 mm). All values are read from the source workbook (M1 primary model, n = 35,508);
nothing is re-estimated. 3-D views are produced beforehand by render3d.py (reference atlases).

Usage:  python3 code/figure3_editorial.py [--panels-only]
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3style import (SC, SHAPES, INK, MUTED, AXIS, BAND, LOWER, HIGHER, NS_SUB, NS_CTX, VMAX, beta_rgb)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path('/home/ubuntu/upload/CKM脑区关联分析_完整结果.xlsx')
REN = ROOT / 'renders'
CON = ['Stage1 vs Stage0', 'Stage2 vs Stage0', 'Stage3 vs Stage0']
SIZES = {'a': (78, 62), 'b': (102, 62), 'c': (180, 82), 'd': (80, 70), 'e': (100, 70)}
STEM = {'a': 'Panel_a_Global_Brain_Structure', 'b': 'Panel_b_Regional_IDP_Landscape',
        'c': 'Panel_c_Cortical_Volume_3D', 'd': 'Panel_d_Subcortical_3D', 'e': 'Panel_e_Subcortical_Forest_Matrix'}
NSDOT = '#C3CBD1'

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 6.4, 'axes.labelsize': 6.6,
                     'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'axes.linewidth': .7,
                     'xtick.major.width': .7, 'ytick.major.width': .7, 'xtick.major.size': 2.2,
                     'ytick.major.size': 2.2, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': AXIS,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


class Canvas:
    """Millimetre coordinates inside a panel box (origin bottom-left)."""

    def __init__(self, fig, box, key, standalone=True):
        self.fig, self.box, self.key, self.standalone = fig, box, key, standalone
        self.w, self.h = SIZES[key]

    def fx(self, x):
        return self.box[0] + x / self.w * self.box[2]

    def fy(self, y):
        return self.box[1] + y / self.h * self.box[3]

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([self.fx(x), self.fy(y), w / self.w * self.box[2], h / self.h * self.box[3]], **kw)

    def text(self, x, y, s, **kw):
        return self.fig.text(self.fx(x), self.fy(y), s, **kw)

    def line(self, xs, ys, **kw):
        self.fig.add_artist(Line2D([self.fx(v) for v in xs], [self.fy(v) for v in ys],
                                   transform=self.fig.transFigure, **kw))

    def mark(self, x, y, k, filled=True, ms=3.8, color=None):
        col = SC[k] if color is None else color
        self.fig.add_artist(Line2D([self.fx(x)], [self.fy(y)], transform=self.fig.transFigure, ls='',
                                   marker=SHAPES[k], ms=ms * (0.9 if k == 1 else 1), mfc=col if filled else 'white',
                                   mec=col, mew=.85))

    def rect(self, x, y, w, h, **kw):
        self.fig.add_artist(Rectangle((self.fx(x), self.fy(y)), w / self.w * self.box[2], h / self.h * self.box[3],
                                      transform=self.fig.transFigure, **kw))

    def letter(self):
        self.text(0.6, self.h - 0.4, self.key, fontsize=9.5, fontweight='bold', va='top', ha='left')


def fmt(v, nd=3):
    return f'{v:.{nd}f}'.replace('-', '−')


def tick_labels(vals):
    return [('0' if abs(v) < 1e-12 else f'{v:g}'.replace('-', '−')) for v in vals]


def clean(ax, keep=('bottom',)):
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(s in keep)


# ------------------------------------------------------------------------------------- data
def load():
    x = pd.read_excel(SRC, sheet_name='完整分类结果')
    m1 = x[x.model.eq('M1')].copy()
    assert m1.n.eq(35508).all()
    g = pd.read_excel(SRC, sheet_name='全局关联')
    g = g[g.contrast.isin(CON)].copy()
    gm = m1[m1.analysis_tier.eq('Global primary')]
    chk = g.merge(gm[['outcome', 'contrast', 'beta_std', 'p_value']], on=['outcome', 'contrast'], suffixes=('', '_m1'))
    assert len(chk) == 12 and np.allclose(chk.beta_std, chk.beta_std_m1) and np.allclose(chk.p_value, chk.p_value_m1)
    i = m1[m1.p_fdr_idp159.notna()].copy()
    assert len(i) == 477 and i.outcome.nunique() == 159
    i['sig'] = i.p_fdr_idp159 < .05
    i['direction'] = np.where(~i.sig, 'n.s.', np.where(i.beta_std < 0, 'Lower', 'Higher'))
    return dict(g=g, i=i)


GOUT = ['GMV', 'PCGMV', 'WMV_matter', 'VCSF']
GLAB = {'GMV': 'Total grey matter', 'PCGMV': 'Peripheral cortical\ngrey matter', 'WMV_matter': 'White matter',
        'VCSF': 'Ventricular CSF'}


# ------------------------------------------------------------------------------------- panel a
def panel_a(c, d):
    c.letter()
    g = d['g']
    X0, XW = 22.5, 25.5
    y_bot, y_top = 8.0, c.h - 9.0
    ax = c.ax(X0, y_bot, XW, y_top - y_bot)
    GAP = 3.75
    ypos = {(o, s): gi * GAP + s for gi, o in enumerate(GOUT) for s in range(3)}
    ymax = 3 * GAP + 2
    ax.set_ylim(ymax + .75, -.75)
    lim = .45
    ax.set_xlim(-lim, lim)
    # alternating bands spanning label, plot and text columns
    for gi in range(4):
        if gi % 2 == 0:
            y0 = gi * GAP - .55
            y1 = gi * GAP + 2.55
            f = lambda yy: y_bot + (ymax + .75 - yy) / (ymax + 1.5) * (y_top - y_bot)
            c.rect(1.2, f(y1), c.w - 1.6, f(y0) - f(y1), facecolor=BAND, edgecolor='none', zorder=0)
    ax.patch.set_alpha(0)
    ax.axvline(0, color=AXIS, lw=.6, ls=(0, (2, 2)), zorder=1)
    rows = []
    for gi, o in enumerate(GOUT):
        ax.text((1.6 - X0) / XW, gi * GAP + 1, GLAB[o], transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=6.3, color=INK, linespacing=1.05)
        for s, ct in enumerate(CON):
            r = g[(g.outcome == o) & (g.contrast == ct)].iloc[0]
            yy = ypos[(o, s)]
            sig = bool(r.p_fdr_family < .05)
            ax.plot([r.ci_low_std, r.ci_high_std], [yy, yy], color=SC[s], lw=1.0, solid_capstyle='butt', zorder=2)
            ax.plot(r.beta_std, yy, marker=SHAPES[s], ms=3.5 * (0.9 if s == 1 else 1), mfc=SC[s] if sig else 'white',
                    mec=SC[s], mew=.85, zorder=3, ls='')
            txt = f'{fmt(r.beta_std)} ({fmt(r.ci_low_std)}, {fmt(r.ci_high_std)})'
            ax.text((X0 + XW + 1.3 - X0) / XW, yy, txt, transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontsize=5.5, color=INK if sig else MUTED)
            rows.append(dict(outcome=o, label=GLAB[o].replace('\n', ' '), contrast=ct, n=int(r.n), beta_std=r.beta_std,
                             ci_low_std=r.ci_low_std, ci_high_std=r.ci_high_std, p_value=r.p_value,
                             q_fdr_global4=r.p_fdr_family, fdr_significant=sig))
    ax.set_yticks([])
    clean(ax)
    ax.spines['bottom'].set_color(AXIS)
    ticks = [-.4, -.2, 0, .2, .4]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels(ticks))
    ax.tick_params(axis='x', pad=1.5)
    ax.set_xlabel('β (SD)', labelpad=1.2)
    ax.text(0, -.115, 'Lower', transform=ax.transAxes, ha='left', va='top', fontsize=5.6, color=LOWER)
    ax.text(1, -.115, 'Higher', transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color=HIGHER)
    c.text(X0 + XW + 1.3, c.h - 7.6, 'β (95% CI)', fontsize=5.9, color=MUTED, ha='left', va='bottom')
    # legend
    yl = c.h - 3.0
    x = 8.0
    for s in range(3):
        c.mark(x, yl, s, True, 3.6)
        c.text(x + 1.6, yl, f'Stage {s + 1}', fontsize=6.2, color=SC[s], fontweight='bold', va='center')
        x += 13.6
    c.mark(x + .6, yl, 2, False, 3.6, color=MUTED)
    c.text(x + 2.2, yl, 'q ≥ 0.05', fontsize=6.0, color=MUTED, va='center')
    d['out_a'] = pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- panel b
FAM = [('all', 'All regional\nIDPs'),
       ('Cortical volume (Desikan-Killiany)', 'Cortical\nvolume'),
       ('Cortical surface area (Desikan-Killiany)', 'Cortical\nsurface area'),
       ('Subcortical volume (FIRST)', 'Subcortical\nvolume'),
       ('Ventricle/CSF/cerebellar volume (aseg)', 'Ventricles, CSF\nand cerebellum')]


def panel_b(c, d):
    c.letter()
    i = d['i']
    y_bot, y_top = 8.0, c.h - 9.6
    GAP = 3.6
    ymax = (len(FAM) - 1) * GAP + 2
    SX0, SXW = 24.0, 41.0
    BX0, BXW = 69.5, 29.0
    sax = c.ax(SX0, y_bot, SXW, y_top - y_bot)
    bax = c.ax(BX0, y_bot, BXW, y_top - y_bot)
    for a_ in (sax, bax):
        a_.set_ylim(ymax + .7, -.7)
        a_.set_yticks([])
        a_.patch.set_alpha(0)
    f = lambda yy: y_bot + (ymax + .7 - yy) / (ymax + 1.4) * (y_top - y_bot)
    xl = (-.56, .64)
    sax.set_xlim(*xl)
    sax.axvline(0, color=AXIS, lw=.6, ls=(0, (2, 2)), zorder=1)
    bax.set_xlim(0, 1)
    rows, cnt = [], []
    for k, (fam, lab) in enumerate(FAM):
        if k % 2 == 0:
            c.rect(1.2, f(k * GAP + 2.55), c.w - 1.6, f(k * GAP - .55) - f(k * GAP + 2.55), facecolor=BAND,
                   edgecolor='none', zorder=0)
        sub = i if fam == 'all' else i[i.phenotype_family.eq(fam)]
        ntot = sub.outcome.nunique()
        c.text(1.8, f(k * GAP + 1) + 1.0, lab, fontsize=6.1, va='center', ha='left', linespacing=1.02,
               fontweight='bold' if fam == 'all' else 'normal')
        c.text(1.8, f(k * GAP + 1) - 2.75, f'{ntot} IDPs', fontsize=5.4, va='center', ha='left', color=MUTED)
        for s, ct in enumerate(CON):
            yy = k * GAP + s
            r = sub[sub.contrast.eq(ct)].sort_values('beta_std').reset_index(drop=True)
            assert len(r) == ntot
            c.mark(SX0 - 1.6, f(yy), s, True, 2.9)
            jit = ((np.arange(len(r)) * 0.6180339887) % 1 - .5) * .64
            ns = ~r.sig
            sax.scatter(r.beta_std[ns], yy + jit[ns], s=1.5, color=NSDOT, lw=0, zorder=2)
            lo = r.sig & (r.beta_std < 0)
            hi = r.sig & (r.beta_std > 0)
            sax.scatter(r.beta_std[lo], yy + jit[lo], s=2.4, color=LOWER, lw=0, zorder=3)
            sax.scatter(r.beta_std[hi], yy + jit[hi], s=2.4, color=HIGHER, lw=0, zorder=3)
            med = float(np.median(r.beta_std))
            nl, nh = int(lo.sum()), int(hi.sum())
            pl, ph = nl / ntot, nh / ntot
            bh = .64
            bax.add_patch(Rectangle((0, yy - bh / 2), 1, bh, facecolor='#E6EBEE', edgecolor='none', zorder=1))
            if nl:
                bax.add_patch(Rectangle((0, yy - bh / 2), pl, bh, facecolor=LOWER, edgecolor='none', zorder=2))
            if nh:
                bax.add_patch(Rectangle((1 - ph, yy - bh / 2), ph, bh, facecolor=HIGHER, edgecolor='none', zorder=2))
            wmm = BXW
            if nl and pl * wmm >= 3.4:
                bax.text(.015, yy, str(nl), ha='left', va='center', fontsize=5.3, color='white', fontweight='bold', zorder=3)
            else:
                bax.text(pl + .015, yy, str(nl), ha='left', va='center', fontsize=5.3,
                         color=LOWER if nl else MUTED, zorder=3)
            if nh and ph * wmm >= 3.4:
                bax.text(.985, yy, str(nh), ha='right', va='center', fontsize=5.3, color='white', fontweight='bold', zorder=3)
            else:
                bax.text(1 - ph - .015, yy, str(nh), ha='right', va='center', fontsize=5.3,
                         color=HIGHER if nh else MUTED, zorder=3)
            cnt.append(dict(family='All regional IDPs' if fam == 'all' else fam, contrast=ct, n_idps=ntot,
                            fdr_lower=nl, fdr_higher=nh, not_significant=ntot - nl - nh,
                            median_beta=med, min_beta=float(r.beta_std.min()), max_beta=float(r.beta_std.max())))
    clean(sax)
    clean(bax)
    for a_ in (sax, bax):
        a_.spines['bottom'].set_color(AXIS)
        a_.tick_params(axis='x', pad=1.5)
    ticks = [-.4, -.2, 0, .2, .4, .6]
    sax.set_xticks(ticks)
    sax.set_xticklabels(tick_labels(ticks))
    sax.set_xlabel('β (SD)', labelpad=1.2)
    bax.set_xticks([0, .5, 1])
    bax.set_xticklabels(['0', '50', '100'])
    bax.set_xlabel('IDPs (%)', labelpad=1.2)
    # column headers
    c.text(SX0 + SXW / 2, c.h - 8.2, 'Estimates for each IDP', fontsize=5.9, color=MUTED, ha='center', va='bottom')
    c.text(BX0, c.h - 8.2, 'Lower', fontsize=5.9, color=LOWER, ha='left', va='bottom', fontweight='bold')
    c.text(BX0 + BXW, c.h - 8.2, 'Higher', fontsize=5.9, color=HIGHER, ha='right', va='bottom', fontweight='bold')
    c.text(BX0 + BXW / 2, c.h - 8.2, 'q < 0.05', fontsize=5.9, color=MUTED, ha='center', va='bottom')
    # legend
    yl = c.h - 3.0
    x = 8.0
    for col, labl, sz in [(LOWER, 'Lower, q < 0.05', 2.4), (HIGHER, 'Higher, q < 0.05', 2.4), (NSDOT, 'q ≥ 0.05', 1.9)]:
        c.fig.add_artist(Line2D([c.fx(x)], [c.fy(yl)], transform=c.fig.transFigure, ls='', marker='o', ms=sz,
                                mfc=col, mec='none'))
        c.text(x + 1.3, yl, labl, fontsize=6.0, va='center', color=INK)
        x += 21.5
    d['out_b'] = i[['outcome', 'official_description', 'phenotype_family', 'region', 'hemisphere', 'contrast', 'n',
                    'beta_std', 'ci_low_std', 'ci_high_std', 'p_value', 'p_fdr_idp159', 'direction']].rename(
        columns={'p_fdr_idp159': 'q_fdr_159IDPs'})
    d['out_b_counts'] = pd.DataFrame(cnt)


# ------------------------------------------------------------------------------------- colour bar
def colourbar(c, x, y, w, h, vertical=True):
    cax = c.ax(x, y, w, h)
    n = 200
    edges = np.linspace(-VMAX, VMAX, n + 1)
    for a0, a1 in zip(edges[:-1], edges[1:]):
        mid = (a0 + a1) / 2
        col = beta_rgb(mid)
        if vertical:
            cax.add_patch(Rectangle((0, a0), 1, a1 - a0 + 1e-4, facecolor=col, edgecolor='none'))
        else:
            cax.add_patch(Rectangle((a0, 0), a1 - a0 + 1e-4, 1, facecolor=col, edgecolor='none'))
    ticks = [-.5, -.25, 0, .25, .5]
    labs = ['−0.5', '−0.25', '0', '0.25', '0.5']
    if vertical:
        cax.set_xlim(0, 1); cax.set_ylim(-VMAX, VMAX)
        cax.set_xticks([]); cax.yaxis.tick_right(); cax.set_yticks(ticks); cax.set_yticklabels(labs)
        cax.tick_params(axis='y', length=1.8, pad=1.2, labelsize=5.6)
    else:
        cax.set_ylim(0, 1); cax.set_xlim(-VMAX, VMAX)
        cax.set_yticks([]); cax.set_xticks(ticks); cax.set_xticklabels(labs)
        cax.tick_params(axis='x', length=1.8, pad=1.2, labelsize=5.6)
    for s in cax.spines.values():
        s.set_visible(False)
    return cax


# ------------------------------------------------------------------------------------- panel c
VIEWS = [('left', 'lateral'), ('left', 'medial'), ('right', 'medial'), ('right', 'lateral')]


def panel_c(c, d):
    c.letter()
    imw = 33.6
    im0 = Image.open(REN / 'cortex_left_lateral_stage1.png')
    asp = im0.size[1] / im0.size[0]
    imh = imw * asp
    x0, gapx = 15.5, 1.3
    top = c.h - 6.6
    xs = [x0 + j * (imw + gapx) for j in range(4)]
    for j, (hemi, view) in enumerate(VIEWS):
        c.text(xs[j] + imw / 2, c.h - 6.0, view.capitalize(), fontsize=6.1, ha='center', va='bottom', color=INK)
    for hemi, js in (('Left hemisphere', (0, 1)), ('Right hemisphere', (2, 3))):
        xa, xb = xs[js[0]] + 3, xs[js[1]] + imw - 3
        c.text((xa + xb) / 2, c.h - 2.2, hemi, fontsize=6.3, ha='center', va='center', color=INK, fontweight='bold')
        c.line([xa, (xa + xb) / 2 - 11], [c.h - 2.2] * 2, color=AXIS, lw=.6)
        c.line([(xa + xb) / 2 + 11, xb], [c.h - 2.2] * 2, color=AXIS, lw=.6)
    for s in range(3):
        yb = top - (s + 1) * imh - s * .4
        c.mark(3.3, yb + imh / 2, s, True, 3.8)
        c.text(5.2, yb + imh / 2, f'Stage {s + 1}', fontsize=6.6, color=SC[s], fontweight='bold', va='center')
        for j, (hemi, view) in enumerate(VIEWS):
            ax = c.ax(xs[j], yb, imw, imh)
            ax.imshow(Image.open(REN / f'cortex_{hemi}_{view}_stage{s + 1}.png'), interpolation='lanczos')
            ax.set_axis_off()
    # shared colour bar (also used by panel d in the composite)
    cbx = xs[3] + imw + 4.0
    cb = colourbar(c, cbx, 24, 2.4, 38)
    c.text(cbx + 1.2, 64.4, 'Higher', fontsize=5.9, ha='center', va='bottom', color=HIGHER, fontweight='bold')
    c.text(cbx + 1.2, 22.6, 'Lower', fontsize=5.9, ha='center', va='top', color=LOWER, fontweight='bold')
    c.text(cbx + 9.6, 43, 'β (SD)', fontsize=6.0, rotation=90, ha='center', va='center')
    c.rect(cbx, 14.2, 2.4, 2.4, facecolor=NS_CTX, edgecolor='none')
    c.text(cbx + 3.4, 15.4, 'n.s.', fontsize=5.9, va='center', color=MUTED)
    v = pd.read_csv(REN / 'cortex_render_values.csv')
    v['render_rgb'] = [('#%02X%02X%02X' % tuple((beta_rgb(b) * 255).round().astype(int))) if q < .05 else 'grey (n.s.)'
                       for b, q in zip(v.beta, v.q)]
    i = d['i']
    cv = i[i.phenotype_family.eq('Cortical volume (Desikan-Killiany)')][
        ['outcome', 'region', 'hemisphere', 'contrast', 'beta_std', 'ci_low_std', 'ci_high_std', 'p_value', 'p_fdr_idp159']]
    cv = cv.assign(stage=cv.contrast.map({k: j + 1 for j, k in enumerate(CON)}))
    out = cv.merge(v[['hemisphere', 'stage', 'region', 'coloured', 'render_rgb']], on=['hemisphere', 'stage', 'region'])
    assert len(out) == 198 and (out.coloured == (out.p_fdr_idp159 < .05)).all()
    d['out_c'] = out.rename(columns={'p_fdr_idp159': 'q_fdr_159IDPs'})


# ------------------------------------------------------------------------------------- panel d
SUBLAB = {'thalamus': 'Thalamus', 'caudate': 'Caudate', 'putamen': 'Putamen', 'pallidum': 'Pallidum',
          'hippocampus': 'Hippocampus', 'amygdala': 'Amygdala', 'accumbens': 'Accumbens',
          'Lateral-Ventricle': 'Lateral ventricle'}
# label side: image left shows the participant's right hemisphere (inferior view, radiological display)
KEY_LEFT = [('caudate', 'right'), ('accumbens', 'right'), ('pallidum', 'right'), ('Lateral-Ventricle', 'right')]
KEY_RIGHT = [('putamen', 'left'), ('amygdala', 'left'), ('hippocampus', 'left'), ('thalamus', 'left')]


def panel_d(c, d):
    c.letter()
    anchors = json.load(open(REN / 'subcortex_anchors.json'))
    kim = Image.open(REN / 'subcortex_key.png')
    asp = kim.size[1] / kim.size[0]
    sw = 24.5
    sh = sw * asp
    sy = (9.0 if c.standalone else 2.0)
    ky = sy + sh + 5.4
    kh = min(c.h - 3.6 - ky, 33.0)
    kw = kh / asp
    kx = (c.w - kw) / 2
    ax = c.ax(kx, ky, kw, kh)
    ax.imshow(kim, interpolation='lanczos')
    ax.set_axis_off()
    c.text(kx + .30 * kw, ky + kh + 1.5, 'R', fontsize=6.4, fontweight='bold', ha='center', va='center', color=MUTED)
    c.text(kx + .70 * kw, ky + kh + 1.5, 'L', fontsize=6.4, fontweight='bold', ha='center', va='center', color=MUTED)

    def put(lst, side):
        pts = []
        for reg, hemi in lst:
            a = anchors[f'{reg}|{hemi}']
            pts.append((reg, kx + a['x'] * kw, ky + (1 - a['y']) * kh))
        pts.sort(key=lambda t: -t[2])
        ys, prev = [], ky + kh + 1.0
        for _, _, py in pts:                       # keep each label near its anchor, >= 2.9 mm apart
            yv = min(py, prev - 2.9, ky + kh - 1.2)
            ys.append(yv)
            prev = yv
        for (reg, px, py), ty in zip(pts, ys):
            if side == 'left':
                tx = kx - 4.0
                c.text(tx, ty, SUBLAB[reg], fontsize=5.9, ha='right', va='center')
                c.line([tx + .5, kx - 1.0, px], [ty, ty, py], color=INK, lw=.45)
            else:
                tx = kx + kw + 4.0
                c.text(tx, ty, SUBLAB[reg], fontsize=5.9, ha='left', va='center')
                c.line([tx - .5, kx + kw + 1.0, px], [ty, ty, py], color=INK, lw=.45)
            c.fig.add_artist(Line2D([c.fx(px)], [c.fy(py)], transform=c.fig.transFigure, ls='', marker='o', ms=1.4,
                                    mfc=INK, mec='white', mew=.3))
    put(KEY_LEFT, 'left')
    put(KEY_RIGHT, 'right')
    gap = (c.w - 3 * sw - 2.0) / 2
    for s in range(3):
        sx = 1.0 + s * (sw + gap)
        ax = c.ax(sx, sy, sw, sh)
        ax.imshow(Image.open(REN / f'subcortex_stage{s + 1}.png'), interpolation='lanczos')
        ax.set_axis_off()
        c.mark(sx + sw / 2 - 5.3, sy + sh + 1.6, s, True, 3.4)
        c.text(sx + sw / 2 - 3.7, sy + sh + 1.6, f'Stage {s + 1}', fontsize=6.3, color=SC[s], fontweight='bold',
               va='center')
    if c.standalone:
        colourbar(c, 22, 3.2, 36, 1.8, vertical=False)
        c.text(20.8, 4.1, 'Lower', fontsize=5.8, ha='right', va='center', color=LOWER, fontweight='bold')
        c.text(59.2, 4.1, 'Higher', fontsize=5.8, ha='left', va='center', color=HIGHER, fontweight='bold')
        c.rect(70.0, 3.2, 1.8, 1.8, facecolor=NS_SUB, edgecolor='none')
        c.text(72.4, 4.1, 'n.s.', fontsize=5.8, va='center', color=MUTED)
    v = pd.read_csv(REN / 'subcortex_render_values.csv')
    v['render_rgb'] = [('#%02X%02X%02X' % tuple((beta_rgb(b) * 255).round().astype(int))) if q < .05 else 'grey (n.s.)'
                       for b, q in zip(v.beta, v.q)]
    assert len(v) == 48 and (v.coloured == (v.q < .05)).all()
    d['out_d'] = v.rename(columns={'q': 'q_fdr_159IDPs', 'beta': 'beta_std'})


# ------------------------------------------------------------------------------------- panel e
EROWS = [('thalamus', 'Thalamus', True), ('caudate', 'Caudate', True), ('putamen', 'Putamen', True),
         ('pallidum', 'Pallidum', True), ('hippocampus', 'Hippocampus', True), ('amygdala', 'Amygdala', True),
         ('accumbens', 'Accumbens', True), ('brain stem + 4th ventricle', 'Brain stem and\n4th ventricle', False),
         ('Lateral-Ventricle', 'Lateral ventricle', True), ('WM-hypointensities (whole brain)', 'WM hypointensities', False)]


def panel_e(c, d):
    c.letter()
    i = d['i']
    fam = ['Subcortical volume (FIRST)', 'Ventricle/CSF/cerebellar volume (aseg)']
    sub = i[i.phenotype_family.isin(fam)]
    y_bot, y_top = 8.2, c.h - 6.4
    pos, labels, k = [], [], 0.0
    for reg, lab, bil in EROWS:
        hs = ['left', 'right'] if bil else [None]
        blk = []
        for h in hs:
            pos.append((reg, h, k))
            blk.append(k)
            k += 1
        labels.append((lab, np.mean(blk), bil, blk))
        k += .5
    ymax = k - 1.5
    f = lambda yy: y_bot + (ymax + .6 - yy) / (ymax + 1.2) * (y_top - y_bot)
    for b, (lab, yc, bil, blk) in enumerate(labels):
        if b % 2 == 0:
            c.rect(1.2, f(blk[-1] + .5), c.w - 1.6, f(blk[0] - .5) - f(blk[-1] + .5), facecolor=BAND,
                   edgecolor='none', zorder=0)
        c.text(19.6, f(yc), lab, fontsize=6.0, ha='right', va='center', linespacing=1.0)
        if bil:
            c.text(21.4, f(blk[0]), 'L', fontsize=5.2, ha='center', va='center', color=MUTED)
            c.text(21.4, f(blk[1]), 'R', fontsize=5.2, ha='center', va='center', color=MUTED)
    cw, cg, cx0 = 23.4, 2.4, 24.0
    xl = (-.70, .84)
    rows = []
    for s, ct in enumerate(CON):
        x = cx0 + s * (cw + cg)
        ax = c.ax(x, y_bot, cw, y_top - y_bot)
        ax.set_ylim(ymax + .6, -.6)
        ax.set_xlim(*xl)
        ax.patch.set_alpha(0)
        ax.axvline(0, color=AXIS, lw=.6, ls=(0, (2, 2)), zorder=1)
        for reg, h, yy in pos:
            r = sub[sub.region.eq(reg) & sub.contrast.eq(ct)]
            if h is not None:
                r = r[r.hemisphere.eq(h)]
            assert len(r) == 1, (reg, h, ct)
            r = r.iloc[0]
            sig = bool(r.sig)
            ax.plot([r.ci_low_std, r.ci_high_std], [yy, yy], color=SC[s], lw=.9, solid_capstyle='butt', zorder=2)
            ax.plot(r.beta_std, yy, marker=SHAPES[s], ms=2.9 * (0.9 if s == 1 else 1), mfc=SC[s] if sig else 'white',
                    mec=SC[s], mew=.8, ls='', zorder=3)
            rows.append(dict(outcome=r.outcome, description=r.official_description, structure=reg,
                             hemisphere=h or 'whole brain', contrast=ct, beta_std=r.beta_std, ci_low_std=r.ci_low_std,
                             ci_high_std=r.ci_high_std, p_value=r.p_value, q_fdr_159IDPs=r.p_fdr_idp159,
                             fdr_significant=sig))
        clean(ax)
        ax.spines['bottom'].set_color(AXIS)
        ax.set_yticks([])
        ax.set_xticks([-.5, 0, .5])
        ax.set_xticklabels(['−0.5', '0', '0.5'])
        ax.tick_params(axis='x', pad=1.5)
        c.mark(x + cw / 2 - 5.0, c.h - 3.4, s, True, 3.4)
        c.text(x + cw / 2 - 3.5, c.h - 3.4, f'Stage {s + 1}', fontsize=6.3, color=SC[s], fontweight='bold', va='center')
        if s == 1:
            ax.set_xlabel('β (SD)', labelpad=1.2)
    d['out_e'] = pd.DataFrame(rows)


FUN = {'a': panel_a, 'b': panel_b, 'c': panel_c, 'd': panel_d, 'e': panel_e}


def save(fig, path):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(path.with_suffix('.' + ext), dpi=600)
    plt.close(fig)
    print(path.with_suffix('.png'), flush=True)


def export(d):
    sd = ROOT / 'source_data'
    sd.mkdir(exist_ok=True)
    d['out_a'].to_csv(sd / 'Fig3a_12_global_estimates.csv', index=False)
    d['out_b'].to_csv(sd / 'Fig3b_477_regional_IDP_estimates.csv', index=False)
    d['out_b_counts'].to_csv(sd / 'Fig3b_FDR_counts_by_family_and_stage.csv', index=False)
    d['out_c'].to_csv(sd / 'Fig3c_198_cortical_volume_estimates_rendered.csv', index=False)
    d['out_d'].to_csv(sd / 'Fig3d_48_subcortical_estimates_rendered.csv', index=False)
    d['out_e'].to_csv(sd / 'Fig3e_54_subcortical_ventricular_estimates.csv', index=False)


ROW1, ROW2, ROW3, GAP = 62, 82, 70, 1.5


def composite(d):
    H = ROW1 + ROW2 + ROW3 + 2 * GAP
    fig = plt.figure(figsize=(180 / 25.4, H / 25.4))
    W = 180.0
    y3 = 0
    y2 = ROW3 + GAP
    y1 = y2 + ROW2 + GAP
    boxes = {'a': (0, y1, 78, ROW1), 'b': (78, y1, 102, ROW1), 'c': (0, y2, 180, ROW2),
             'd': (0, y3, 80, ROW3), 'e': (80, y3, 100, ROW3)}
    for k, (x, y, w, h) in boxes.items():
        assert (w, h) == SIZES[k]
        FUN[k](Canvas(fig, (x / W, y / H, w / W, h / H), k, standalone=False), d)
    save(fig, ROOT / 'figures' / 'Figure3_CKM_Brain_Structure_a-e')


def main(args):
    d = load()
    for k in ['a', 'b', 'c', 'd', 'e']:
        w, h = SIZES[k]
        fig = plt.figure(figsize=(w / 25.4, h / 25.4))
        FUN[k](Canvas(fig, (0, 0, 1, 1), k, standalone=True), d)
        save(fig, ROOT / 'panels' / STEM[k])
    export(d)
    if not args.panels_only:
        composite(d)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panels-only', action='store_true')
    main(ap.parse_args())
