#!/usr/bin/env python3
"""Figure 5 editorial redesign (visual grammar shared with Figure 4).

Panels a-d are drawn as standalone 180-mm figures by the same functions that build the
native vector composite. Inputs are existing summary tables only; no model is refitted.
Usage: python code/figure5_editorial.py [--panels-only]
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------- shared visual grammar (Figure 4)
HIGH, LOW = '#7B4F9E', '#1E8C8C'
INK, MUTED, AXIS, GRID = '#24394A', '#647B88', '#9FB0BA', '#EEF2F4'
CATS = ['Immune System', 'Metabolism', 'Hemostasis', 'Extracellular matrix organization', 'Signal Transduction', 'Neuronal System']
CATLAB = dict(zip(CATS, ['Immune system', 'Metabolism', 'Hemostasis', 'ECM organization', 'Signal transduction', 'Neuronal system']))
CATCOL = {'Immune System': '#6D91A8', 'Metabolism': '#CAA450', 'Hemostasis': '#A08098',
          'Extracellular matrix organization': '#5C9D98', 'Signal Transduction': '#878FB4', 'Neuronal System': '#A6AB75'}
DIRLAB = {'positive': 'Higher', 'negative': 'Lower'}
DIRCOL = {'positive': HIGH, 'negative': LOW}
BETA_CMAP = LinearSegmentedColormap.from_list('beta', ['#245E8C', '#7FAED0', '#DCE9F2', '#F7F7F5', '#F0D5CC', '#D98E7D', '#AE4636'])
NEG, POS = '#3B75A6', '#B5523F'          # brain beta sign, as in Figure 3 (blue negative, red positive)

MODELS = ['Base', 'A', 'B', 'R', 'G', 'V', 'F']
MLAB = {'Base': 'Basic covariates', 'A': 'CKM adjusted', 'B': 'BMI adjusted', 'R': 'Renal adjusted',
        'G': 'Glycaemic adjusted', 'V': 'Vascular–lipid adjusted', 'F': 'Combined adjustment'}
MHEAD = {'Base': 'Basic\ncovariates', 'A': 'CKM\nadjusted', 'B': 'BMI\nadjusted', 'R': 'Renal\nadjusted',
         'G': 'Glycaemic\nadjusted', 'V': 'Vascular–\nlipid adjusted', 'F': 'Combined\nadjustment'}
YPOS = np.array([0, 1, 2, 3.4, 4.4, 5.4, 6.8])     # [core sequence] | [parallel clinical extensions] | [combined]
YLIM = (7.4, -.6)
BLOCKS = ['Basic', 'CKM stage', 'BMI', 'eGFR', 'HbA1c + diabetes', 'BP + lipids']
INCL = {'Base': [1, 0, 0, 0, 0, 0], 'A': [1, 1, 0, 0, 0, 0], 'B': [1, 1, 1, 0, 0, 0], 'R': [1, 1, 1, 1, 0, 0],
        'G': [1, 1, 1, 0, 1, 0], 'V': [1, 1, 1, 0, 0, 1], 'F': [1, 1, 1, 1, 1, 1]}
GLOB = ['GMV', 'PCGMV', 'WMV_matter', 'VCSF']
GLAB = {'GMV': 'Total grey matter', 'PCGMV': 'Peripheral cortical\ngrey matter', 'WMV_matter': 'White matter', 'VCSF': 'Ventricular CSF'}
PROT = ['GFAP', 'NEFL', 'RBPMS']
ROI = {'IDP25016': 'Right putamen', 'IDP25017': 'Left pallidum', 'IDP25018': 'Right pallidum', 'IDP26812': 'Left precuneus',
       'IDP26818': 'Left supramarginal', 'IDP26911': 'Right posterior cingulate', 'IDP26919': 'Right supramarginal',
       'IDP25012': 'Right thalamus', 'IDP26802': 'Left middle temporal', 'IDP26836': 'Right middle temporal (area)',
       'IDP26903': 'Right middle temporal'}
HEIGHTS = {'a': 54, 'b': 50, 'c': 66, 'd': 56}
FADE = .38

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 6.5, 'axes.labelsize': 6.8,
                     'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'axes.linewidth': .8,
                     'xtick.major.width': .8, 'ytick.major.width': .8, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': AXIS,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


class Canvas:
    """Millimetre coordinates inside a panel box (origin bottom-left)."""
    def __init__(self, fig, box, key):
        self.fig, self.box, self.key, self.h = fig, box, key, HEIGHTS[key]

    def ax(self, x, y, w, h, **kw):
        bx, by, bw, bh = self.box
        return self.fig.add_axes([bx + x / 180 * bw, by + y / self.h * bh, w / 180 * bw, h / self.h * bh], **kw)

    def text(self, x, y, s, **kw):
        bx, by, bw, bh = self.box
        return self.fig.text(bx + x / 180 * bw, by + y / self.h * bh, s, **kw)

    def letter(self):
        self.text(1.5, self.h - 1.0, self.key, fontsize=9.5, fontweight='bold', va='top')


def fmt(v, nd=3):
    return f'{v:.{nd}f}'.replace('-', '−')


def clean(ax, keep=('bottom',)):
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(s in keep)
    ax.spines['bottom'].set_color(AXIS)


def beta_axis(ax, lim=.118):
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-.1, 0, .1]); ax.set_xticklabels(['−0.1', '0', '0.1'])
    ax.tick_params(axis='x', length=2.5, pad=1.8)


def model_labels(c, x_right, y, h, fs=6.4):
    la = c.ax(0, y, x_right, h); la.axis('off'); la.set_xlim(0, 1); la.set_ylim(*YLIM)
    for i, m in enumerate(MODELS):
        la.text(1, YPOS[i], MLAB[m], ha='right', va='center', fontsize=fs, color=INK)
    return la


def module_name(e):
    cat, dr = e.split(' | ')
    return f'{CATLAB[cat]} · {DIRLAB[dr]}'


# ---------------------------------------------------------------- data
def load(root):
    rd = lambda f: pd.read_csv(root / 'data' / f, float_precision='round_trip')
    allm = rd('all_seven_module_results.csv')
    mods = rd('module_definitions.csv')
    order = sorted(mods.module, key=lambda e: (CATS.index(e.split(' | ')[0]), 0 if e.endswith('positive') else 1))
    glob = allm[allm.testing_family == 'modules_global48'].copy()
    reg = allm[allm.testing_family == 'modules_regional1908'].copy()
    for t in (glob, reg):
        t['category'] = t.exposure.str.split(' | ', regex=False).str[0]
        t['direction'] = t.exposure.str.split(' | ', regex=False).str[1]
    assert len(glob) == 336 and len(reg) == 13356
    cnt = (reg[reg.q_bh < .05].groupby(['model', 'direction', 'category']).size()
           .reindex(pd.MultiIndex.from_product([MODELS, ['positive', 'negative'], CATS], names=['model', 'direction', 'category']), fill_value=0)
           .reset_index(name='FDR_significant'))
    pairs = rd('Panel_c_all105_fixed_pair_estimates.csv')
    prot = rd('Panel_d_all21_candidate_estimates.csv')
    boot = rd('seven_model_paired_bootstrap_draws.csv')
    return dict(glob=glob, reg=reg, cnt=cnt, pairs=pairs, prot=prot, boot=boot, modules=order)


# ---------------------------------------------------------------- panel a
def swarm_y(vals, base, w_mm, mm_per_row, lim=.118, r=.62):
    xs = w_mm / (2 * lim)
    cand = [0] + [s * k * .07 for k in range(1, 7) for s in (1, -1)]
    placed, out = [], np.zeros(len(vals))
    for i in np.argsort(vals, kind='stable'):
        dy_ok = cand[-1]
        for dy in cand:
            if all(((vals[i] - vx) * xs) ** 2 + ((base + dy - vy) * mm_per_row) ** 2 > (2 * r) ** 2 for vx, vy in placed):
                dy_ok = dy; break
        out[i] = base + dy_ok; placed.append((vals[i], out[i]))
    return out


def panel_a(c, d):
    c.letter()
    g = d['glob']; y0, h, fw = 11, 34, 30
    xs = [36 + j * 33.2 for j in range(4)]
    model_labels(c, 33, y0, h)
    rows = []
    for j, o in enumerate(GLOB):
        ax = c.ax(xs[j], y0, fw, h)
        ax.axvline(0, color='#B9C6CD', lw=.7, zorder=0)
        for i, m in enumerate(MODELS):
            ax.axhline(YPOS[i], color=GRID, lw=.6, zorder=0)
            rr = g[(g.model == m) & (g.outcome == o)].set_index('exposure').loc[d['modules']]
            ax.plot([rr.beta.min(), rr.beta.max()], [YPOS[i]] * 2, color='#AFBDC5', lw=1.1, solid_capstyle='round', zorder=1)
            yy = swarm_y(rr.beta.to_numpy(), YPOS[i], fw, h / (YLIM[0] - YLIM[1]))
            for dr, mk, sz in [('positive', 'o', 9), ('negative', 'D', 6.8)]:
                z = np.asarray(rr.index.str.endswith(dr)); sig = rr.q_bh.to_numpy()[z] < .05
                ax.scatter(rr.beta.to_numpy()[z], yy[z], s=sz, marker=mk, lw=.75, zorder=3,
                           facecolor=[DIRCOL[dr] if s else 'white' for s in sig], edgecolor=DIRCOL[dr])
            for e, b, q, yv in zip(rr.index, rr.beta, rr.q_bh, yy):
                rows.append(dict(model=m, model_label=MLAB[m], outcome=o, module=e, module_label=module_name(e),
                                 beta=b, q_bh=q, display_y=yv))
        beta_axis(ax); ax.set_ylim(*YLIM); ax.set_yticks([]); clean(ax)
        ax.set_title(GLAB[o], fontsize=6.8, fontweight='bold', pad=3.5, color=INK)
    # q < 0.05 count column
    ca = c.ax(167, y0, 11, h); ca.axis('off'); ca.set_xlim(0, 1); ca.set_ylim(*YLIM)
    ca.text(.5, YLIM[1] - .45, 'q < 0.05', ha='center', va='bottom', fontsize=6.0, color=MUTED)
    for i, m in enumerate(MODELS):
        k = int((g[g.model == m].q_bh < .05).sum()); n = int((g.model == m).sum())
        ca.text(.5, YPOS[i], f'{k}/{n}', ha='center', va='center', fontsize=6.2, color=MUTED)
    c.text(36 + (3 * 33.2 + fw) / 2, 4.9, 'Standardised β', ha='center', va='center', fontsize=6.8)
    lg = c.ax(36, .2, 130, 3.2); lg.axis('off')
    lg.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec=HIGH, mew=.8, ms=3.4, label='Higher module'),
                       Line2D([], [], marker='D', ls='', mfc='white', mec=LOW, mew=.8, ms=2.9, label='Lower module'),
                       Line2D([], [], color='#AFBDC5', lw=1.2, label='β range')],
              ncol=3, loc='center', frameon=False, fontsize=6.2, columnspacing=2.2, handletextpad=.5, handlelength=1.4)
    d['out_a'] = pd.DataFrame(rows)


# ---------------------------------------------------------------- panel b
def panel_b(c, d):
    c.letter()
    y0, h = 8.5, 27.5
    model_labels(c, 31, y0, h)
    # covariate inclusion matrix
    mx = c.ax(33.5, y0, 29, h); mx.set_xlim(-.5, 5.5); mx.set_ylim(*YLIM); mx.axis('off')
    for i, m in enumerate(MODELS):
        mx.add_patch(Rectangle((-.5, YPOS[i] - .36), 6, .72, facecolor='#F2F5F7', edgecolor='none', zorder=0))
        inc = np.flatnonzero(INCL[m])
        mx.plot([inc.min(), inc.max()], [YPOS[i]] * 2, color='#7F909B', lw=.9, zorder=1)
        mx.scatter(range(6), [YPOS[i]] * 6, s=13, zorder=2, lw=0,
                   color=[INK if v else '#D5DDE2' for v in INCL[m]])
    for k, bname in enumerate(BLOCKS):
        mx.text(k - .1, YLIM[1] - .15, bname, rotation=48, rotation_mode='anchor', ha='left', va='bottom', fontsize=5.9, color=INK)
    # diverging stacked bars
    cnt = d['cnt']; L, R = 36, 228
    ax = c.ax(71, y0, 105, h); ax.set_xlim(-L, R); ax.set_ylim(*YLIM)
    for i, m in enumerate(MODELS):
        for dr, sgn in [('positive', -1), ('negative', 1)]:
            sub = cnt[(cnt.model == m) & (cnt.direction == dr)].set_index('category').loc[CATS, 'FDR_significant']
            left = 0
            for cat, v in sub.items():
                if v == 0: continue
                ax.barh(YPOS[i], sgn * v, left=sgn * left, height=.66, color=CATCOL[cat], edgecolor='white', lw=.6, zorder=2)
                if v >= 9:
                    ax.text(sgn * (left + v / 2), YPOS[i], str(v), ha='center', va='center', fontsize=5.5, color='white', fontweight='bold', zorder=3)
                left += v
            tot = int(sub.sum())
            ax.text(sgn * (left + 2.5), YPOS[i], f'{tot}', ha='left' if sgn > 0 else 'right', va='center',
                    fontsize=6.3, fontweight='bold', color=INK if tot else MUTED, zorder=3)
    ax.axvline(0, color=INK, lw=.8, zorder=4)
    ax.set_yticks([]); clean(ax)
    ax.set_xticks([-20, 0, 50, 100, 150, 200]); ax.set_xticklabels(['20', '0', '50', '100', '150', '200'])
    ax.tick_params(axis='x', length=2.5, pad=1.8)
    ax.set_xlabel('FDR-significant regional module–MRI associations', fontsize=6.6, labelpad=2.5)
    # Higher / Lower headers (Figure 4c style)
    hy = y0 + h + 1.6
    for lab, col, a0, a1 in [('Higher', HIGH, -L + 2, -1.5), ('Lower', LOW, 1.5, R - 2)]:
        xa = 71 + (a0 + L) / (L + R) * 105; xb = 71 + (a1 + L) / (L + R) * 105
        hl = c.ax(xa, hy, xb - xa, .5); hl.axis('off'); hl.add_patch(Rectangle((0, 0), 1, 1, color=col, transform=hl.transAxes))
        c.text((xa + xb) / 2 if lab == 'Higher' else xa + 2, hy + .9, lab, ha='center' if lab == 'Higher' else 'left',
               va='bottom', fontsize=6.6, fontweight='bold', color=col)
    lg = c.ax(98, 42.8, 80, 7); lg.axis('off')
    lg.legend(handles=[Patch(color=CATCOL[k], label=CATLAB[k]) for k in CATS], ncol=3, loc='center',
              frameon=False, fontsize=5.9, columnspacing=1.3, handlelength=1.1, handleheight=.8, labelspacing=.35, handletextpad=.45)
    out = cnt.copy(); out['model_label'] = out.model.map(MLAB); out['direction_label'] = out.direction.map(DIRLAB)
    d['out_b'] = out
    d['out_b_design'] = pd.DataFrame([dict(model=m, model_label=MLAB[m], **dict(zip(BLOCKS, INCL[m]))) for m in MODELS])


# ---------------------------------------------------------------- panel c
def pair_order(p):
    pr = p[['exposure', 'outcome']].drop_duplicates().copy()
    pr['catg'] = pr.exposure.str.split(' | ', regex=False).str[0]; pr['dir'] = pr.exposure.str.split(' | ', regex=False).str[1]
    pr['k'] = pr.catg.map(CATS.index) * 2 + (pr.dir == 'negative')
    return pr.sort_values(['k', 'outcome']).reset_index(drop=True)


def panel_c(c, d):
    c.letter()
    p = d['pairs']; pr = pair_order(p); n = len(pr); assert n == 15
    look = p.set_index(['model', 'exposure', 'outcome'])
    XP = np.array([0, 1, 2, 3.25, 4.25, 5.25, 6.5]); y0, h = 10.5, 44
    ax = c.ax(88, y0, 89, h); ax.set_xlim(-.5, 7.0); ax.set_ylim(n - .5, -.5); ax.axis('off')
    norm = Normalize(-.08, .08); rows = []
    for i, r in pr.iterrows():
        for j, m in enumerate(MODELS):
            rr = look.loc[(m, r.exposure, r.outcome)]; sig = rr.q_bh < .05
            ax.add_patch(Rectangle((XP[j] - .5, i - .5), 1, 1, facecolor=BETA_CMAP(norm(rr.beta)), alpha=1 if sig else FADE,
                                   edgecolor='white', lw=.9, zorder=1))
            dark = sig and abs(rr.beta) > .056
            ax.text(XP[j], i, fmt(rr.beta), ha='center', va='center', fontsize=5.9 if sig else 5.6,
                    fontweight='bold' if sig else 'normal', color='white' if dark else (INK if sig else '#8A99A2'), zorder=3)
            rows.append(dict(row=i + 1, module=r.exposure, module_label=module_name(r.exposure), outcome=r.outcome, region=ROI[r.outcome],
                             model=m, model_label=MLAB[m], beta=rr.beta, ci_low=rr.ci_low, ci_high=rr.ci_high, p_value=rr.p_value, q_bh=rr.q_bh))
    # group separators
    grp = pr.k.to_numpy(); brk = [i for i in range(1, n) if grp[i] != grp[i - 1]]
    for b in brk:
        ax.plot([-.5, 7.0], [b - .5] * 2, color='white', lw=2.4, zorder=2)
    for j, m in enumerate(MODELS):
        ax.text(XP[j], -.95, MHEAD[m], ha='center', va='bottom', fontsize=5.9, color=INK, linespacing=1.0)
        k = int(sum(look.loc[(m, r.exposure, r.outcome)].q_bh < .05 for _, r in pr.iterrows()))
        ax.text(XP[j], n - .1, f'{k}/{n}', ha='center', va='top', fontsize=6.1, color=MUTED)
    ax.text(-.75, n - .1, 'q < 0.05', ha='right', va='top', fontsize=6.0, color=MUTED)
    # left label columns (shared y)
    la = c.ax(0, y0, 87, h); la.set_xlim(0, 87); la.set_ylim(n - .5, -.5); la.axis('off')
    la.text(2, -.95, 'Protein module', ha='left', va='bottom', fontsize=6.1, color=MUTED)
    la.text(46, -.95, 'Regional MRI measure', ha='left', va='bottom', fontsize=6.1, color=MUTED)
    starts = [0] + brk; ends = brk + [n]
    for s, e in zip(starts, ends):
        r = pr.iloc[s]; mid = (s + e - 1) / 2
        la.add_patch(Rectangle((2, s - .4), 1.3, e - s - .2, color=CATCOL[r.catg], lw=0))
        la.text(4.6, mid, CATLAB[r.catg], ha='left', va='center', fontsize=6.3, color=INK)
        la.text(31.5, mid, DIRLAB[r.dir], ha='left', va='center', fontsize=6.3, color=DIRCOL[r.dir], fontweight='bold')
        if e - s > 1:
            la.plot([43.3, 43.3], [s - .3, e - .7], color='#B8C5CC', lw=.7)
            for yy in (s - .3, e - .7):
                la.plot([43.3, 44.3], [yy, yy], color='#B8C5CC', lw=.7)
    for i, r in pr.iterrows():
        la.text(46, i, ROI[r.outcome], ha='left', va='center', fontsize=6.2, color=INK)
    # legend
    cb = c.ax(94, 3.0, 34, 1.7)
    cbo = ColorbarBase(cb, cmap=BETA_CMAP, norm=norm, orientation='horizontal', boundaries=np.linspace(-.08, .08, 41), ticks=[-.08, 0, .08])
    cbo.solids.set_rasterized(False); cbo.solids.set_edgecolor('face'); cbo.outline.set_visible(False)
    cb.set_xticklabels(['−0.08', '0', '0.08']); cb.tick_params(length=1.8, pad=1, labelsize=5.8)
    c.text(132, 3.85, 'Standardised β', ha='left', va='center', fontsize=6.2)
    for j, (al, lab) in enumerate([(1, 'q < 0.05'), (FADE, 'q ≥ 0.05')]):
        q = c.ax(152 + j * 13, 2.25, 3.2, 3.2); q.axis('off')
        q.add_patch(Rectangle((0, 0), 1, 1, facecolor=BETA_CMAP(norm(-.06)), alpha=al, edgecolor='none', transform=q.transAxes))
        c.text(156 + j * 13, 3.85, lab, ha='left', va='center', fontsize=6.0)
    d['out_c'] = pd.DataFrame(rows)


# ---------------------------------------------------------------- panel d
def panel_d(c, d):
    c.letter()
    x, boot = d['prot'], d['boot']; y0, h, fw = 10, 36, 44
    model_labels(c, 33, y0, h)
    kdes, peak = {}, 0
    for p in PROT:
        for m in MODELS:
            v = boot[(boot.protein == p) & (boot.model == m)].beta_bootstrap.to_numpy(); assert len(v) == 1000
            k = gaussian_kde(v, bw_method='scott'); bw = np.sqrt(k.covariance[0, 0])
            grid = np.linspace(v.min() - 2 * bw, v.max() + 2 * bw, 240); den = k(grid)
            kdes[p, m] = (grid, den); peak = max(peak, den.max())
    rows = []
    for j, p in enumerate(PROT):
        ax = c.ax(36 + j * 47, y0, fw, h); ax.axvline(0, color='#B9C6CD', lw=.7, zorder=0)
        for i, m in enumerate(MODELS):
            rr = x[(x.exposure == p) & (x.model == m)].iloc[0]; col = NEG if rr.beta < 0 else POS
            grid, den = kdes[p, m]; base = YPOS[i] + .02
            ax.fill_between(grid, base, base - den / peak * .58, facecolor=col, edgecolor=col, alpha=.30, lw=.6, zorder=1)
            yc = YPOS[i] + .2; sig = rr.q_bh < .05
            ax.plot([rr.ci_low, rr.ci_high], [yc, yc], color=col, lw=1.1, solid_capstyle='round', zorder=2)
            ax.scatter(rr.beta, yc, s=15, marker='o', facecolor=col if sig else 'white', edgecolor=col, lw=.9, zorder=3)
            v = boot[(boot.protein == p) & (boot.model == m)].beta_bootstrap
            rows.append(dict(protein=p, outcome='PCGMV', model=m, model_label=MLAB[m], beta=rr.beta, ci_low=rr.ci_low, ci_high=rr.ci_high,
                             p_value=rr.p_value, q_bh=rr.q_bh, bootstrap_n=len(v), bootstrap_median=v.median(),
                             bootstrap_q025=v.quantile(.025), bootstrap_q975=v.quantile(.975)))
        ax.set_xlim(-.14, .12); ax.set_xticks([-.1, 0, .1]); ax.set_xticklabels(['−0.1', '0', '0.1'])
        ax.tick_params(axis='x', length=2.5, pad=1.8); ax.set_ylim(*YLIM); ax.set_yticks([]); clean(ax)
        ax.set_title(p, fontsize=7.4, fontweight='bold', pad=3, color=INK)
    c.text(36 + (2 * 47 + fw) / 2, 4.6, 'Standardised β, peripheral cortical grey-matter volume', ha='center', va='center', fontsize=6.8)
    lg = c.ax(36, .1, 140, 3); lg.axis('off')
    lg.legend(handles=[Patch(facecolor='#9DB3C2', edgecolor='#9DB3C2', alpha=.6, label='Bootstrap distribution'),
                       Line2D([], [], color=INK, lw=1, marker='o', ms=3.3, mfc=INK, mec=INK, label='β (95% CI), q < 0.05'),
                       Line2D([], [], color=INK, lw=1, marker='o', ms=3.3, mfc='white', mec=INK, label='q ≥ 0.05')],
              ncol=3, loc='center', frameon=False, fontsize=6.2, columnspacing=2.2, handlelength=1.6, handletextpad=.5)
    d['out_d'] = pd.DataFrame(rows)


# ---------------------------------------------------------------- supplementary: all 336 global estimates with HC3 CIs
def supplementary(root, d):
    H = 228; fig = plt.figure(figsize=(180 / 25.4, H / 25.4)); g = d['glob']; mods = d['modules']
    for i, m in enumerate(MODELS):
        for j, o in enumerate(GLOB):
            ax = fig.add_axes([(44 + j * 34) / 180, (H - 8 - (i + 1) * 30.6 + 3) / H, 31 / 180, 26 / H])
            rr = g[(g.model == m) & (g.outcome == o)].set_index('exposure').loc[mods]
            ax.axvline(0, color='#B9C6CD', lw=.6, zorder=0)
            for k, e in enumerate(mods):
                r = rr.loc[e]; dr = e.split(' | ')[1]; col = DIRCOL[dr]
                ax.plot([r.ci_low, r.ci_high], [k, k], color=col, lw=.8, alpha=.85)
                ax.scatter(r.beta, k, s=6, marker='o' if dr == 'positive' else 'D', facecolor='white' if r.q_bh >= .05 else col,
                           edgecolor=col, lw=.6, zorder=3)
            ax.set_ylim(11.6, -.6); ax.set_xlim(-.12, .12); ax.set_xticks([-.1, 0, .1])
            ax.set_xticklabels(['−0.1', '0', '0.1'] if i == 6 else [])
            ax.tick_params(axis='both', length=1.8, pad=1, labelsize=5.3); clean(ax)
            if j == 0:
                ax.set_yticks(range(12)); ax.set_yticklabels([module_name(e) for e in mods], fontsize=5.2)
                for t, e in zip(ax.get_yticklabels(), mods):
                    t.set_color(INK)
                ax.tick_params(axis='y', length=0, pad=1.5)
                fig.text(1.5 / 180, (H - 8 - (i + 1) * 30.6 + 3 + 13) / H, MLAB[m], rotation=90, ha='left', va='center',
                         fontsize=6.3, fontweight='bold', color=INK)
            else:
                ax.set_yticks([])
            if i == 0:
                ax.set_title(GLAB[o].replace('\n', ' '), fontsize=6.6, fontweight='bold', pad=4)
    fig.text((44 + 2 * 34 - 1.5) / 180, 2.6 / H, 'Standardised β (95% CI)', ha='center', va='center', fontsize=6.6)
    save(fig, root / 'figures/Supplementary_Figure_Fig5a_All336_Global_Estimates_HC3_CI')


# ---------------------------------------------------------------- io
FUN = {'a': panel_a, 'b': panel_b, 'c': panel_c, 'd': panel_d}
STEM = {'a': 'Panel_a_Global_Module_MRI', 'b': 'Panel_b_Model_Design_and_Regional_Counts',
        'c': 'Panel_c_Fixed_Module_Region_Pairs', 'd': 'Panel_d_Single_Protein_PCGMV'}


def save(fig, path):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(path.with_suffix('.' + ext), dpi=600)
    plt.close(fig); print(path.with_suffix('.png'), flush=True)


def export(root, d):
    sd = root / 'source_data'
    d['out_a'].to_csv(sd / 'Fig5a_336_global_estimates.csv', index=False)
    d['out_b'].to_csv(sd / 'Fig5b_84_counts_by_model_direction_category.csv', index=False)
    d['out_b_design'].to_csv(sd / 'Fig5b_model_covariate_design.csv', index=False)
    tot = d['cnt'].groupby(['model', 'direction']).FDR_significant.sum().unstack().reindex(MODELS)
    tot['total'] = tot.sum(axis=1); tot.insert(0, 'model_label', [MLAB[m] for m in MODELS])
    tot.rename(columns={'positive': 'higher_modules', 'negative': 'lower_modules'}).to_csv(sd / 'Fig5b_totals.csv')
    d['out_c'].to_csv(sd / 'Fig5c_105_fixed_pair_estimates.csv', index=False)
    d['out_d'].to_csv(sd / 'Fig5d_21_protein_estimates_with_bootstrap_summary.csv', index=False)
    d['boot'].to_csv(sd / 'Fig5d_21000_bootstrap_draws.csv', index=False)


def main(args):
    root = args.root; d = load(root)
    for k in ['a', 'b', 'c', 'd']:
        fig = plt.figure(figsize=(180 / 25.4, HEIGHTS[k] / 25.4))
        FUN[k](Canvas(fig, (0, 0, 1, 1), k), d); save(fig, root / 'panels' / STEM[k])
    export(root, d)
    if not args.panels_only:
        H = sum(HEIGHTS.values()); fig = plt.figure(figsize=(180 / 25.4, H / 25.4)); top = H
        for k in ['a', 'b', 'c', 'd']:
            top -= HEIGHTS[k]; FUN[k](Canvas(fig, (0, top / H, 1, HEIGHTS[k] / H), k), d)
        save(fig, root / 'figures/Figure5_Proteomic_Modules_Brain_Structure_a-d')
        supplementary(root, d)
    (root / 'plot_metadata.json').write_text(json.dumps(dict(
        n=4483, models=MLAB, model_order=MODELS, covariate_blocks=BLOCKS, design=INCL, width_mm=180,
        panel_heights_mm=HEIGHTS, composite_height_mm=sum(HEIGHTS.values()), refitted_models=False,
        direction_labels={'positive': 'Higher', 'negative': 'Lower'}), indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--panels-only', action='store_true')
    main(ap.parse_args())
