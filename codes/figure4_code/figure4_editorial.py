#!/usr/bin/env python3
"""Figure 4 editorial redesign: CKM-stage plasma proteomic differences and functional annotation.

Usage:
  python figure4_editorial.py --root PATH [--panels-only]
Inputs in ROOT/data: 05_protein_ckm_stage_contrasts.csv, Reactome_top_level_all_tests.csv,
Reactome_ORA_all_tests.csv. No estimate is modified. Source q is used for every FDR state.
For P values that underflowed in the source file (P < 1e-290), -log10(P) is recomputed from the
same beta/SE with the t distribution (df inferred from the source P values) at high precision, and is
used only as the volcano y coordinate.
"""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde
from adjustText import adjust_text

TERMS = ['stage_f1', 'stage_f2', 'stage_f3']
SC = ['#377EB8', '#D8A429', '#C44E52']          # stage colours, identical to Figures 1-3
SH = ['s', 'D', 'o']                              # stage markers, identical to Figures 1-3
HIGH, LOW, NS = '#7B4F9E', '#1E8C8C', '#CBD3D9'   # direction colours (distinct from stage colours)
MIX = '#8D9AA3'
INK, MUTED, AXIS = '#24394A', '#647B88', '#9FB0BA'
EVID = LinearSegmentedColormap.from_list('evidence', ['#EEF3F6', '#B9CCD8', '#6E92AB', '#345C78', '#1A3346'])
FE_CMAP = LinearSegmentedColormap.from_list('fe', ['#B9AC9C', '#E4DDD3', '#F8F8F6', '#BBCFDC', '#5E86A3', '#22445C'])
CATS = ['Immune System', 'Metabolism', 'Hemostasis', 'Extracellular matrix organization', 'Signal Transduction', 'Neuronal System']
CATLAB = ['Immune system', 'Metabolism', 'Hemostasis', 'ECM organization', 'Signal transduction', 'Neuronal system']
CATCOL = {'Immune System': '#6D91A8', 'Metabolism': '#CAA450', 'Hemostasis': '#A08098',
          'Extracellular matrix organization': '#5C9D98', 'Signal Transduction': '#878FB4',
          'Neuronal System': '#A6AB75', 'Disease': '#B49A86'}
SELS = ['positive_FDR', 'negative_FDR']
DF = 40489
HEIGHTS = {'a': 66, 'b': 58, 'cd': 66}
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 6.5, 'axes.labelsize': 6.8,
                     'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'axes.linewidth': .8,
                     'xtick.major.width': .8, 'ytick.major.width': .8, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': AXIS,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


class Canvas:
    """Millimetre coordinates inside one panel box of a (possibly composite) figure."""
    def __init__(self, fig, box, key):
        self.fig, self.box, self.h, self.key = fig, box, HEIGHTS[key], key
    def ax(self, x, y, w, h, **kw):
        bx, by, bw, bh = self.box
        return self.fig.add_axes([bx + x / 180 * bw, by + y / self.h * bh, w / 180 * bw, h / self.h * bh], **kw)
    def text(self, x, y, s, **kw):
        bx, by, bw, bh = self.box
        return self.fig.text(bx + x / 180 * bw, by + y / self.h * bh, s, **kw)
    def letter(self, s, x=1.5, y=None):
        self.text(x, (self.h - 1.2) if y is None else y, s, fontsize=9.5, fontweight='bold', va='top', ha='left')


def tidy(ax, left=True, bottom=True):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def _betacf(a, b, x, itmax=5000, eps=1e-15):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, dd = 1.0, 1.0 - qab * x / qap
    dd = 1.0 / (dd if abs(dd) > tiny else tiny)
    h = dd
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        dd = 1.0 + aa * dd; dd = 1.0 / (dd if abs(dd) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= dd * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        dd = 1.0 + aa * dd; dd = 1.0 / (dd if abs(dd) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        delta = dd * c; h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError('continued fraction did not converge')


def neglog10_t(beta, se):
    """Two-sided t-test -log10(P) = -log10 I_x(df/2, 1/2), x = df/(df+t^2), in log space."""
    from math import lgamma, log
    t2 = (beta / se) ** 2
    a, b = DF / 2.0, .5
    x = DF / (DF + t2)
    log_front = a * log(x) + b * log1p_safe(-x) - log(a) - (lgamma(a) + lgamma(b) - lgamma(a + b))
    return -(log_front + log(_betacf(a, b, x))) / log(10)


def log1p_safe(v):
    return float(np.log1p(v))


# ----------------------------------------------------------------------------- data
def load(root):

    p = pd.read_csv(root / 'data/05_protein_ckm_stage_contrasts.csv', float_precision='round_trip')
    p['sig'] = p.p_fdr < .05
    p['direction'] = np.select([p.sig & (p.beta > 0), p.sig & (p.beta < 0)], ['higher', 'lower'], 'not_FDR')
    under = p.p_value < 1e-290
    p['neglog10P'] = -np.log10(p.p_value.where(~under))
    p.loc[under, 'neglog10P'] = [neglog10_t(b, s) for b, s in zip(p.loc[under, 'beta'], p.loc[under, 'se'])]
    p['neglog10P_source'] = np.where(under, 'recomputed_t_df40489_source_underflow', 'source_P')
    # validation of the recomputation on finite source values
    chk = p[(~under) & (p.p_value < 1e-100)].sample(200, random_state=7)
    dev = np.abs(np.array([neglog10_t(b, s) for b, s in zip(chk.beta, chk.se)]) + np.log10(chk.p_value))
    assert np.median(dev) < 1e-3, dev
    def top_labels(k):
      labels = []
      for t in TERMS:
        g = p[p.contrast == t]
        for sign, dr in [(1, 'higher'), (-1, 'lower')]:
            s = g[g.sig & (g.beta * sign > 0)].assign(ab=lambda d: d.beta.abs()).sort_values(['ab', 'protein'], ascending=[False, True]).head(k)
            labels += [dict(contrast=t, direction=dr, rank_abs_beta=i + 1, protein=r.protein, beta=r.beta, se=r.se, q_source=r.p_fdr, neglog10P=r.neglog10P)
                       for i, (_, r) in enumerate(s.iterrows())]
      return pd.DataFrame(labels)
    labels, labels12 = top_labels(6), top_labels(12)
    b = p.pivot(index='protein', columns='contrast', values='beta')[TERMS]
    s = p.pivot(index='protein', columns='contrast', values='sig')[TERMS]
    pattern = s.apply(lambda r: ''.join('1' if r[t] else '0' for t in TERMS), axis=1)
    def direction(row):
        act = [t for t in TERMS if s.loc[row.name, t]]
        if not act:
            return 'none'
        v = row[act]
        return 'all_higher' if (v > 0).all() else ('all_lower' if (v < 0).all() else 'discordant')
    comp = b.apply(direction, axis=1)
    member = pd.DataFrame({'pattern': pattern, 'direction_class': comp})
    inter = member[member.pattern != '000'].groupby(['pattern', 'direction_class']).size().unstack(fill_value=0)
    for col in ['all_higher', 'all_lower', 'discordant']:
        if col not in inter:
            inter[col] = 0
    inter['total'] = inter[['all_higher', 'all_lower', 'discordant']].sum(axis=1)
    inter = inter.reset_index().sort_values(['total', 'pattern'], ascending=[False, True]).reset_index(drop=True)
    shared = member[(member.pattern == '111') & member.direction_class.isin(['all_higher', 'all_lower'])]
    grad = b.loc[shared.index].assign(direction_class=shared.direction_class).reset_index()
    cat = pd.read_csv(root / 'data/Reactome_top_level_all_tests.csv')
    cat = cat[cat.category.isin(CATS) & cat.contrast.isin(TERMS) & cat.selection.isin(SELS)].copy()
    ora = pd.read_csv(root / 'data/Reactome_ORA_all_tests.csv')
    ora = ora[ora.contrast.isin(TERMS) & ora.selection.isin(SELS)].copy()
    ids = []
    for _, g in ora.groupby(['contrast', 'selection']):
        ids += list(g[g.q_bh < .05].sort_values(['q_bh', 'pathway_id']).head(3).pathway_id)
    paths = ora[ora.pathway_id.isin(set(ids))].copy()
    return dict(p=p, labels=labels, labels12=labels12, member=member, inter=inter, grad=grad, cat=cat, paths=paths,
                sizes=s.sum().to_dict())


# ----------------------------------------------------------------------------- panel a
YT = [0, 10, 50, 200, 500, 1000, 1500]

def volcano(ax, g, lab, j, first, fs=5.5, ytop=1900, expand=(1.12, 1.45), force=(.2, .5)):
    ax.set_yscale('function', functions=(lambda y: np.sqrt(np.clip(y, 0, None)), lambda y: np.square(y)))
    ax.axvline(0, color='#C9D3D9', lw=.7, zorder=0)
    thr = g[g.sig].neglog10P.min()
    ax.axhline(thr, color='#8798A3', lw=.7, ls=(0, (3, 2.5)), zorder=1)
    for dr, col, s, a in [('not_FDR', NS, 2.6, .9), ('lower', LOW, 3.4, .55), ('higher', HIGH, 3.4, .5)]:
        h = g[g.direction == dr]
        ax.scatter(h.beta, h.neglog10P, s=s, c=col, edgecolors='none', alpha=a, zorder=2, rasterized=False)
    ax.set_xlim(-1.25, 1.72); ax.set_ylim(0, ytop)
    ax.set_xticks([-1, -.5, 0, .5, 1, 1.5]); ax.set_xticklabels(['−1', '', '0', '', '1', ''])
    yt = [v for v in YT if v <= ytop]
    ax.set_yticks(yt); ax.set_yticklabels([f'{v:,}' for v in yt] if first else [])
    ax.set_xlabel('Standardised β', labelpad=2)
    if first:
        ax.set_ylabel('−log$_{10}$ $P$', labelpad=2)
    ax.set_title(f'Stage {j + 1} vs Stage 0', fontsize=7.2, fontweight='bold', color=SC[j], pad=3)
    tidy(ax)
    hi, lo = int((g.direction == 'higher').sum()), int((g.direction == 'lower').sum())
    ax.text(.985, .05, f'{hi:,} higher', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3, color=HIGH, fontweight='bold')
    ax.text(.015, .05, f'{lo:,} lower', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=LOW, fontweight='bold')
    texts = []
    for _, r in lab.iterrows():
        col = HIGH if r.direction == 'higher' else LOW
        ax.scatter(r.beta, r.neglog10P, s=9, c=col, edgecolors='white', linewidths=.5, zorder=4)
        dx = -.09 if r.direction == 'higher' else .09          # place inside the V, where no points lie
        texts.append(ax.text(r.beta + dx, r.neglog10P, r.protein, fontsize=fs, color=col,
                             ha='right' if dx < 0 else 'left', va='center', zorder=5))
    sig = g[g.sig]
    adjust_text(texts, x=sig.beta.to_numpy(), y=sig.neglog10P.to_numpy(), ax=ax, expand=expand,
                force_text=force, force_static=(.2, .35), force_pull=(.02, .02), only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
                ensure_inside_axes=True, time_lim=8,
                arrowprops=dict(arrowstyle='-', color='#8C9BA5', lw=.45, shrinkA=1, shrinkB=1.5))
    return thr


def panel_a(c, d):
    c.letter('a')
    thr = {}
    for j, t in enumerate(TERMS):
        ax = c.ax(16 + j * 55.5, 14.5, 49.5, 42)
        thr[t] = volcano(ax, d['p'][d['p'].contrast == t], d['labels'][d['labels'].contrast == t], j, j == 0)
    h = [Line2D([], [], marker='o', ls='', color=HIGH, ms=3.2, label='Higher, q < 0.05'),
         Line2D([], [], marker='o', ls='', color=LOW, ms=3.2, label='Lower, q < 0.05'),
         Line2D([], [], marker='o', ls='', color=NS, ms=3.2, label='q ≥ 0.05'),
         Line2D([], [], color='#8798A3', lw=.8, ls=(0, (3, 2.5)), label='FDR = 0.05')]
    lg = c.ax(16, .2, 160, 3.4); lg.axis('off')
    lg.legend(handles=h, ncol=4, loc='center', frameon=False, fontsize=6.2, columnspacing=2.2, handletextpad=.4)
    d['thresholds'] = thr


# ----------------------------------------------------------------------------- panel b
def panel_b(c, d):
    c.letter('b')
    u, sizes = d['inter'], d['sizes']
    n = len(u)
    # intersection bars (stacked by direction class)
    bar = c.ax(44, 28, 76, 27)
    bottom = np.zeros(n)
    for col, key in [(HIGH, 'all_higher'), (LOW, 'all_lower'), (MIX, 'discordant')]:
        v = u[key].to_numpy()
        bar.bar(np.arange(n), v, bottom=bottom, width=.62, color=col, edgecolor='white', lw=.4)
        bottom += v
    for j, v in enumerate(u.total):
        bar.text(j, v + 45, f'{v:,}', ha='center', va='bottom', fontsize=6.2, color=INK)
    top = u.iloc[0]
    bar.text(0, top.all_higher / 2, f'{int(top.all_higher):,}', ha='center', va='center', fontsize=5.8, color='white', fontweight='bold')
    bar.set_xlim(-.6, n - .4); bar.set_ylim(0, 2250); bar.set_xticks([])
    bar.set_yticks([0, 1000, 2000]); bar.set_yticklabels(['0', '1,000', '2,000'])
    bar.set_ylabel('Proteins', labelpad=2); tidy(bar, bottom=False)
    lg = c.ax(62, 46, 58, 9); lg.axis('off')
    lg.legend(handles=[Patch(color=HIGH, label='Consistently higher'), Patch(color=LOW, label='Consistently lower'), Patch(color=MIX, label='Discordant direction')],
              loc='upper right', frameon=False, fontsize=6, ncol=1, handlelength=1, handleheight=.8, labelspacing=.3)
    # matrix
    mat = c.ax(44, 9.5, 76, 16)
    mat.set_xlim(-.6, n - .4); mat.set_ylim(2.6, -.6)
    for k in range(3):
        mat.axhspan(k - .38, k + .38, color='#F2F5F7', zorder=0)
    for j, ptn in enumerate(u.pattern):
        act = [k for k in range(3) if ptn[k] == '1']
        if len(act) > 1:
            mat.plot([j, j], [min(act), max(act)], color='#7F909B', lw=.9, zorder=1)
        for k in range(3):
            on = k in act
            mat.scatter(j, k, s=20 if on else 13, marker=SH[k], c=SC[k] if on else '#DCE3E7', edgecolors='white' if on else 'none', linewidths=.4, zorder=2)
    mat.axis('off')
    # set sizes
    st = c.ax(8, 9.5, 18, 16)
    st.barh(range(3), [sizes[t] for t in TERMS], height=.5, color=SC)
    st.set_xlim(3300, 0); st.set_ylim(2.6, -.6); st.set_yticks([])
    st.set_xticks([0, 2000]); st.set_xticklabels(['0', '2,000']); tidy(st, left=False)
    st.spines['bottom'].set_color(AXIS)
    for k, t in enumerate(TERMS):
        st.text(sizes[t] + 120, k, f'{int(sizes[t]):,}', ha='right', va='center', fontsize=5.8, color=SC[k])
        c.text(34.5, 9.5 + 16 * (1 - (k + .6) / 3.2), f'Stage {k + 1}', ha='center', va='center', fontsize=6.3, color=SC[k], fontweight='bold')
    st.set_xlabel('Proteins, q < 0.05', labelpad=1.5, fontsize=6.2)
    # stage gradient of shared, direction-consistent proteins
    g = d['grad']
    ax = c.ax(137, 10.5, 41, 44)
    ax.axhline(0, color='#B9C5CC', lw=.7, zorder=0)
    summ = []
    for cls, col in [('all_higher', HIGH), ('all_lower', LOW)]:
        h = g[g.direction_class == cls]
        meds = []
        for j, t in enumerate(TERMS):
            v = h[t].to_numpy()
            kde = gaussian_kde(v, bw_method='scott')
            grid = np.linspace(v.min(), v.max(), 200)
            den = kde(grid); den = den / den.max() * .36
            ax.fill_betweenx(grid, j - den, j + den, facecolor=col, edgecolor=col, alpha=.28, lw=.5, zorder=1)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([j, j], [q1, q3], color=col, lw=2.0, solid_capstyle='butt', zorder=2)
            meds.append(med)
            summ.append(dict(direction_class=cls, contrast=t, n=len(v), median=med, q1=q1, q3=q3, min=v.min(), max=v.max()))
        ax.plot(range(3), meds, color=col, lw=.9, zorder=3)
        for j, m in enumerate(meds):
            ax.scatter(j, m, s=16, marker=SH[j], facecolor='white', edgecolor=SC[j], lw=1, zorder=4)
            ax.text(j - .15, m + (.07 if cls == 'all_higher' else -.07), f'{m:.2f}'.replace('-', '−'), fontsize=5.7, color=col,
                    va='bottom' if cls == 'all_higher' else 'top', ha='right', zorder=5, fontweight='bold')
        lab = f'Consistently higher (n = {len(h):,})' if cls == 'all_higher' else f'Consistently lower (n = {len(h):,})'
        ax.text(-.62, 1.70 if cls == 'all_higher' else -1.21, lab, fontsize=5.9, color=col, va='center', ha='left', fontweight='bold')
    ax.set_xlim(-.68, 2.55); ax.set_ylim(-1.28, 1.78)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
    for j, tl in enumerate(ax.get_xticklabels()):
        tl.set_color(SC[j]); tl.set_fontweight('bold')
    ax.set_yticks([-1, -.5, 0, .5, 1, 1.5]); ax.set_yticklabels(['−1.0', '−0.5', '0', '0.5', '1.0', '1.5'])
    ax.set_ylabel('Standardised β', labelpad=2); tidy(ax)
    d['grad_summary'] = pd.DataFrame(summ)


# ----------------------------------------------------------------------------- panels c and d
def column_header(c, x0, w, ytop):
    """Two-tier header: direction group, then stage numerals in stage colours."""
    cw = w / 6
    for gi, (lab, col) in enumerate([('Higher', HIGH), ('Lower', LOW)]):
        xs = x0 + gi * 3 * cw
        c.text(xs + 1.5 * cw, ytop + 3.6, lab, ha='center', va='bottom', fontsize=6.4, color=col, fontweight='bold')
        a = c.ax(xs + .6, ytop + 3.0, 3 * cw - 1.2, .45); a.axis('off')
        a.add_patch(Rectangle((0, 0), 1, 1, color=col, transform=a.transAxes))
        for k in range(3):
            c.text(xs + (k + .5) * cw, ytop + .5, str(k + 1), ha='center', va='bottom', fontsize=6.3, color=SC[k], fontweight='bold')
    c.text(x0 - 2.6, ytop + .5, 'Stage', ha='right', va='bottom', fontsize=6, color=MUTED)


def panel_c(c, d):
    c.letter('c')
    cat = d['cat'].set_index(['category', 'contrast', 'selection'])
    x0, w, y0, rh = 30, 50, 15, 6.2
    ax = c.ax(x0, y0, w, rh * 6)
    lim = 1.2
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    rows = []
    for i, ct in enumerate(CATS):
        for k, (sel) in enumerate([s for s in SELS for _ in range(3)]):
            t = TERMS[k % 3]
            r = cat.loc[(ct, t, sel)]
            l2 = np.log2(r.fold_enrichment)
            sig = r.q_bh < .05
            ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, facecolor=FE_CMAP(norm(np.clip(l2, -lim, lim))),
                                   alpha=1 if sig else .38, edgecolor='white', lw=1.1, zorder=1))
            dark = sig and l2 > .55
            ax.text(k, i, f'{r.fold_enrichment:.2f}', ha='center', va='center', fontsize=6.1 if sig else 5.6,
                    fontweight='bold' if sig else 'normal', color='white' if dark else (INK if sig else '#8A99A2'), zorder=4)
            rows.append(dict(category=ct, contrast=t, selection=sel, query_n=r.query_n, overlap_k=r.overlap_k,
                             fold_enrichment=r.fold_enrichment, log2_fold_enrichment=l2, p_value=r.p_value, q_bh=r.q_bh,
                             testing_family_size=r.testing_family_size))
    ax.axvline(2.5, color='white', lw=3.2, zorder=2)
    ax.set_xlim(-.5, 5.5); ax.set_ylim(5.5, -.5); ax.axis('off')
    for i, (ct, lab) in enumerate(zip(CATS, CATLAB)):
        yy = y0 + rh * (5.5 - i)
        c.text(x0 - 2.6, yy, lab, ha='right', va='center', fontsize=6.4)
        s = c.ax(x0 - 1.8, yy - rh * .36, .9, rh * .72); s.axis('off')
        s.add_patch(Rectangle((0, 0), 1, 1, color=CATCOL[ct], transform=s.transAxes))
    column_header(c, x0, w, y0 + rh * 6 + .6)
    c.text(x0 + w / 2, y0 + rh * 6 + 9.6, 'Reactome top-level category', ha='center', va='bottom', fontsize=6.3, color=MUTED)
    cb = c.ax(x0 + 2, 6.2, 26, 1.6)
    cbo = ColorbarBase(cb, cmap=FE_CMAP, norm=norm, orientation='horizontal', boundaries=np.linspace(-lim, lim, 61), ticks=[-1, 0, 1])
    cbo.solids.set_rasterized(False); cbo.solids.set_edgecolor('face')
    cb.set_xticklabels(['0.5', '1', '2']); cb.tick_params(length=1.8, pad=1, labelsize=5.8)
    for sp in cb.spines.values():
        sp.set_linewidth(.4)
    c.text(x0 + 15, 1.6, 'Fold enrichment', ha='center', va='center', fontsize=6)
    for j, (al, lab) in enumerate([(1, 'q < 0.05'), (.38, 'q ≥ 0.05')]):
        q = c.ax(x0 + 30 + j * 13, 5.4, 3.2, 3.2); q.axis('off')
        q.add_patch(Rectangle((0, 0), 1, 1, facecolor=FE_CMAP(norm(1)), alpha=al, edgecolor='none', transform=q.transAxes))
        c.text(x0 + 34 + j * 13, 7, lab, ha='left', va='center', fontsize=5.9)
    d['c_source'] = pd.DataFrame(rows)


NICE = {'Innate Immune System': 'Innate immune system', 'Neutrophil degranulation': 'Neutrophil degranulation',
        'ECM proteoglycans': 'ECM proteoglycans', 'CS/DS-GAG biosynthesis': 'CS/DS-GAG biosynthesis',
        'Hemostasis': 'Hemostasis', 'Platelet activation, signaling and aggregation': 'Platelet activation,\nsignaling and aggregation',
        'Diseases associated with glycosaminoglycan metabolism': 'Diseases of GAG\nmetabolism', "Hedgehog 'on' state": "Hedgehog ‘on’ state"}
ORDER_CAT = ['Immune System', 'Metabolism', 'Hemostasis', 'Extracellular matrix organization', 'Signal Transduction', 'Disease']


def panel_d(c, d):
    c.letter('d', x=86.5)
    paths = d['paths'].copy()
    paths['top'] = paths.top_level_categories.str.split(';').str[0]
    first = paths.groupby('pathway').agg(top=('top', 'first'), minq=('q_bh', 'min')).reset_index()
    first['rank'] = first.top.map({k: i for i, k in enumerate(ORDER_CAT)})
    order = first.sort_values(['rank', 'minq', 'pathway']).pathway.tolist()
    assert len(order) == 9
    look = paths.set_index(['pathway', 'contrast', 'selection'])
    x0, w, y0 = 127, 46, 15
    rh = 37.2 / 9
    ax = c.ax(x0, y0, w, rh * 9)
    qmax = 8
    rows = []
    for i, pw in enumerate(order):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color='#F4F7F8', zorder=0)
        for k, sel in enumerate([s for s in SELS for _ in range(3)]):
            t = TERMS[k % 3]
            r = look.loc[(pw, t, sel)]
            sig = r.q_bh < .05
            sc = -np.log10(r.q_bh)
            if r.overlap_k > 0:
                ax.scatter(k, i, s=r.overlap_k * .42 + 1.2, facecolor=EVID(min(sc / qmax, 1)) if sig else 'white',
                           edgecolor=EVID(min(sc / qmax, 1)) if sig else '#B9C6CE', lw=.7 if sig else .6, zorder=3)
            rows.append(dict(pathway=pw, pathway_id=r.pathway_id, top_level=first.set_index('pathway').top[pw], contrast=t,
                             selection=sel, overlap_k=r.overlap_k, pathway_background_K=r.pathway_background_K,
                             fold_enrichment=r.fold_enrichment, p_value=r.p_value, q_bh=r.q_bh, testing_family_size=r.testing_family_size))
    ax.axvline(2.5, color='#DCE3E7', lw=.8, zorder=1)
    ax.set_xlim(-.5, 5.5); ax.set_ylim(8.5, -.5); ax.axis('off')
    top = first.set_index('pathway').top
    for i, pw in enumerate(order):
        yy = y0 + rh * (8.5 - i)
        c.text(x0 - 2.6, yy, NICE.get(pw, pw), ha='right', va='center', fontsize=6.1, linespacing=.95)
        s = c.ax(x0 - 1.8, yy - rh * .40, .9, rh * .80); s.axis('off')
        s.add_patch(Rectangle((0, 0), 1, 1, color=CATCOL[top[pw]], transform=s.transAxes))
    column_header(c, x0, w, y0 + rh * 9 + .6)
    c.text(x0 + w / 2, y0 + rh * 9 + 9.6, 'Representative Reactome pathways', ha='center', va='bottom', fontsize=6.3, color=MUTED)
    cb = c.ax(x0 - 30, 6.2, 22, 1.6)
    cbo = ColorbarBase(cb, cmap=EVID, norm=Normalize(0, qmax), orientation='horizontal', boundaries=np.linspace(0, qmax, 61), ticks=[0, 4, 8])
    cbo.solids.set_rasterized(False); cbo.solids.set_edgecolor('face')
    cb.tick_params(length=1.8, pad=1, labelsize=5.8)
    for sp in cb.spines.values():
        sp.set_linewidth(.4)
    c.text(x0 - 19, 1.6, '−log$_{10}$ q', ha='center', va='center', fontsize=6)
    for j, k in enumerate([10, 100, 300]):
        xx = x0 + 3 + j * 10.5
        sa = c.ax(xx - 2.5, 4.4, 5, 5); sa.axis('off'); sa.set_xlim(-1, 1); sa.set_ylim(-1, 1)
        sa.scatter(0, 0, s=k * .42 + 1.2, facecolor='#9DB3C2', edgecolor='#6E8796', lw=.5)
        c.text(xx + 2.6, 6.9, str(k), ha='left', va='center', fontsize=5.8)
    c.text(x0 + 17, 1.6, 'Overlapping proteins', ha='center', va='center', fontsize=6)
    hl = c.ax(x0 + 33, 4.6, 3, 4.6); hl.axis('off'); hl.set_xlim(-1, 1); hl.set_ylim(-1, 1)
    hl.scatter(0, 0, s=30, facecolor='white', edgecolor='#B9C6CE', lw=.6)
    c.text(x0 + 36.4, 6.9, 'q ≥ 0.05', ha='left', va='center', fontsize=5.8)
    d['d_source'] = pd.DataFrame(rows)
    d['d_order'] = order


def panel_cd(c, d):
    panel_c(c, d)
    panel_d(c, d)


FUN = {'a': panel_a, 'b': panel_b, 'cd': panel_cd}
STEM = {'a': 'Panel_a_Stage_Volcano_Plots', 'b': 'Panel_b_Overlap_and_Stage_Gradient', 'cd': 'Panel_c-d_Function_Category_and_Pathways'}


def supplementary_volcano(root, d):
    H = 222
    fig = plt.figure(figsize=(180 / 25.4, H / 25.4))
    for j, t in enumerate(TERMS):
        ax = fig.add_axes([16 / 180, (14 + (2 - j) * 69.5) / H, 158 / 180, 58 / H])
        g = d['p'][d['p'].contrast == t]
        volcano(ax, g, d['labels12'][d['labels12'].contrast == t], j, True, fs=5.8,
                ytop=float(g.neglog10P.max()) * 1.18, expand=(1.35, 1.9), force=(.45, .9))
    h = [Line2D([], [], marker='o', ls='', color=HIGH, ms=3.2, label='Higher, q < 0.05'),
         Line2D([], [], marker='o', ls='', color=LOW, ms=3.2, label='Lower, q < 0.05'),
         Line2D([], [], marker='o', ls='', color=NS, ms=3.2, label='q ≥ 0.05'),
         Line2D([], [], color='#8798A3', lw=.8, ls=(0, (3, 2.5)), label='FDR = 0.05')]
    fig.legend(handles=h, ncol=4, loc='lower center', bbox_to_anchor=(.53, .002), frameon=False, fontsize=6.4, columnspacing=2.2)
    save(fig, root / 'figures/Supplementary_Figure_Fig4a_Expanded_Volcano_72_labels')


def save(fig, path):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.' + ext), dpi=600)
    plt.close(fig)
    print(path.with_suffix('.png'), flush=True)


def export(root, d):
    sd = root / 'source_data'
    sd.mkdir(exist_ok=True)
    d['p'].to_csv(sd / 'Fig4a_all_8769_protein_contrasts.csv', index=False)
    d['labels'].to_csv(sd / 'Fig4a_36_labels_top6_abs_beta.csv', index=False)
    d['labels12'].to_csv(sd / 'SuppFig4a_72_labels_top12_abs_beta.csv', index=False)
    pd.Series(d['thresholds']).rename('FDR_boundary_neglog10P').to_csv(sd / 'Fig4a_FDR_boundaries.csv')
    it = d['inter'].copy()
    it.insert(0, 'intersection', it.pattern.map(lambda p: ' + '.join(f'Stage {k + 1}' for k in range(3) if p[k] == '1') + ' only' if p.count('1') < 3 else 'Stages 1, 2 and 3'))
    it['pattern'] = 'S' + it.pattern
    it.to_csv(sd / 'Fig4b_intersections_by_direction.csv', index=False)
    d['member'].to_csv(sd / 'Fig4b_membership_2923_proteins.csv')
    d['grad'].to_csv(sd / 'Fig4b_shared_direction_consistent_1834_betas.csv', index=False)
    d['grad_summary'].to_csv(sd / 'Fig4b_shared_gradient_summary.csv', index=False)
    d['c_source'].to_csv(sd / 'Fig4c_36_category_tiles.csv', index=False)
    d['d_source'].to_csv(sd / 'Fig4d_54_pathway_cells.csv', index=False)


def main(a):
    r = a.root
    for f in ['panels', 'figures', 'source_data']:
        (r / f).mkdir(exist_ok=True)
    d = load(r)
    for key in ['a', 'b', 'cd']:
        fig = plt.figure(figsize=(180 / 25.4, HEIGHTS[key] / 25.4))
        FUN[key](Canvas(fig, (0, 0, 1, 1), key), d)
        save(fig, r / 'panels' / STEM[key])
    export(r, d)
    supplementary_volcano(r, d)
    if not a.panels_only:
        h = sum(HEIGHTS.values())
        fig = plt.figure(figsize=(180 / 25.4, h / 25.4))
        top = h
        for key in ['a', 'b', 'cd']:
            ph = HEIGHTS[key]; top -= ph
            FUN[key](Canvas(fig, (0, top / h, 1, ph / h), key), d)
        save(fig, r / 'figures/Figure4_CKM_Proteomic_Profiles_a-d')
    meta = dict(n=40497, proteins=2923, set_sizes={k: int(v) for k, v in d['sizes'].items()},
                intersections=d['inter'].to_dict('records'), shared_direction_consistent=int(len(d['grad'])),
                labels=int(len(d['labels'])), pathways=d['d_order'], width_mm=180, height_mm=sum(HEIGHTS.values()),
                y_volcano='-log10 P, square-root scale; underflow P recomputed from beta/SE (t, df=40489)',
                FDR_state='source q < 0.05', estimates_modified=False)
    (r / 'plot_metadata.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--panels-only', action='store_true')
    main(ap.parse_args())
