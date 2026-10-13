#!/usr/bin/env python3
"""Figure 6 -- CKM proteomic score, neuropsychiatric risk and brain structure (publication figure).

Every number is read from results/ (aggregate tables produced by code/01-05). Participant-level
data are only used for panel a's density estimate and are never exported.
Usage: python code/figure6_editorial.py [--panels-only]
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
SC4 = ['#9AA8B2', '#377EB8', '#D8A429', '#C44E52']          # Stage 0 (reference) + Stage 1-3 as Figures 1-5
HIGH, LOW = '#7B4F9E', '#1E8C8C'
INK, MUTED, AXIS, GRID = '#24394A', '#647B88', '#9FB0BA', '#EEF2F4'
CATS = ['Immune System', 'Metabolism', 'Hemostasis', 'Extracellular matrix organization', 'Signal Transduction', 'Neuronal System']
CATLAB = ['Immune', 'Metabolism', 'Hemostasis', 'ECM', 'Signalling', 'Neuronal']
CATCOL = {'Immune System': '#6D91A8', 'Metabolism': '#CAA450', 'Hemostasis': '#A08098',
          'Extracellular matrix organization': '#5C9D98', 'Signal Transduction': '#878FB4', 'Neuronal System': '#A6AB75'}
MCOL = {'M1+PS': '#8DAEC4', 'M3': '#345C78', 'M2+F4': '#5C9D98', 'M2+GN': '#A08098'}
MMRK = {'M1+PS': 'o', 'M3': 'o', 'M2+F4': 's', 'M2+GN': '^'}
MLAB = {'M1+PS': 'CKM-PS vs M1', 'M3': 'CKM-PS vs M2', 'M2+F4': 'Figure 4-anchored score', 'M2+GN': 'GFAP + NEFL vs M2'}
DCOMP = [('M1+PS', 'M1'), ('M3', 'M2'), ('M2+GN', 'M2')]   # panel d comparisons
RISK_CMAP = LinearSegmentedColormap.from_list('risk', ['#F4F7F9', '#C9D8E2', '#8DAEC4', '#4F7C9C', '#24496A', '#13283B'])
T3COL = ['#EBD39A', '#D8A429', '#8A6410']                   # Stage 2 colour, gap tertiles 1-3
GROUPS = [('Neurodegenerative', ['Dementia', 'Alzheimers_disease', 'Parkinsons_disease']),
          ('Cerebrovascular', ['Stroke']),
          ('Psychiatric', ['Depression', 'Anxiety_disorder', 'Bipolar_disorder', 'Schizophrenia']),
          ('Sleep', ['Sleep_disorder'])]
ORDER = [o for _, g in GROUPS for o in g]
LAB = {'Dementia': 'All-cause dementia', 'Alzheimers_disease': "Alzheimer's disease", 'Parkinsons_disease': "Parkinson's disease",
       'Stroke': 'Stroke', 'Depression': 'Depression', 'Anxiety_disorder': 'Anxiety disorder', 'Bipolar_disorder': 'Bipolar disorder',
       'Schizophrenia': 'Schizophrenia', 'Sleep_disorder': 'Sleep disorder'}
GLAB = {'GMV': 'Total grey matter', 'PCGMV': 'Peripheral cortical\ngrey matter', 'WMV_matter': 'White matter', 'VCSF': 'Ventricular CSF'}
SIZE = {'a': (56, 54), 'b': (124, 54), 'cd': (180, 62), 'e': (114, 58), 'g': (66, 58), 'f': (180, 44)}
ROWS = [['a', 'b'], ['cd'], ['e', 'g'], ['f']]
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 6.5, 'axes.labelsize': 6.6,
                     'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'axes.linewidth': .8,
                     'xtick.major.width': .8, 'ytick.major.width': .8, 'xtick.major.size': 2.3,
                     'ytick.major.size': 2.3, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': AXIS,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


class Canvas:
    """Millimetre coordinates inside a panel (origin bottom-left)."""
    def __init__(self, fig, x0, y0, key, FW, FH):
        self.fig, self.x0, self.y0, self.key, self.FW, self.FH = fig, x0, y0, key, FW, FH
        self.w, self.h = SIZE[key]
    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([(self.x0 + x) / self.FW, (self.y0 + y) / self.FH, w / self.FW, h / self.FH], **kw)
    def text(self, x, y, s, **kw):
        return self.fig.text((self.x0 + x) / self.FW, (self.y0 + y) / self.FH, s, **kw)
    def letter(self, k=None, x=1.0):
        self.text(x, self.h - 1.0, k or self.key, fontsize=9.5, fontweight='bold', va='top')


def fmt(v, nd=2):
    return f'{v:.{nd}f}'.replace('-', '−')


def clean(ax, keep=('bottom',)):
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(s in keep)


def load(root):
    r = root / 'results'
    d = dict(w=pd.read_csv(r / '04_protein_weights_2923.csv'), h1=pd.read_csv(r / '04_H1_associations.csv'),
             inc=pd.read_csv(r / '05_increments.csv'), grid=pd.read_csv(r / '04_H3_risk_grid.csv'),
             km=pd.read_csv(r / '04_H3_km_curves.csv'), br=pd.read_csv(r / '04_MRI_bridge.csv'),
             summ=json.loads((r / '04_score_summary.json').read_text()))
    a = pd.read_parquet(root / 'private/analysis.parquet', columns=['Eid', 'stage'])
    s = pd.read_parquet(root / 'private/oof_scores.parquet', columns=['Eid', 'ckm_ps'])
    assert (a.Eid.to_numpy() == s.Eid.to_numpy()).all()
    z = (s.ckm_ps - s.ckm_ps.mean()) / s.ckm_ps.std()
    rows, dens = [], []
    rng = np.random.default_rng(6)
    for st in range(4):
        v = z[a.stage == st].to_numpy()
        q = np.percentile(v, [5, 25, 50, 75, 95])
        sub = v if len(v) <= 6000 else rng.choice(v, 6000, replace=False)
        yy = np.linspace(np.percentile(v, .5), np.percentile(v, 99.5), 160)
        kd = gaussian_kde(sub)(yy)
        rows.append(dict(stage=st, n=len(v), mean=v.mean(), sd=v.std(ddof=1), p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4]))
        dens.append(pd.DataFrame({'stage': st, 'ckm_ps_sd': yy, 'density': kd}))
    d['a_sum'] = pd.DataFrame(rows); d['a_kde'] = pd.concat(dens)
    # panel b: top 8 Higher + top 8 Lower by |mean weight| (prespecified display rule)
    w = d['w']
    hi = w[w.mean_weight > 0].nlargest(8, 'mean_weight'); lo = w[w.mean_weight < 0].nsmallest(8, 'mean_weight')
    d['b_top'] = pd.concat([hi, lo.iloc[::-1]]).reset_index(drop=True)
    st = w[w.selection_freq >= .8]
    conc = ((st.direction == 'Higher') & (st.f4_direction_class == 'all_higher')) | ((st.direction == 'Lower') & (st.f4_direction_class == 'all_lower'))
    inset = st.f4_direction_class.isin(['all_higher', 'all_lower'])
    d['b_bar'] = pd.DataFrame([dict(segment='Same direction as Figure 4', n=int(conc.sum())),
                               dict(segment='Opposite direction', n=int((inset & ~conc).sum())),
                               dict(segment='Not in Figure 4 shared set', n=int((~inset).sum()))])
    assert d['b_bar'].n.sum() == len(st) == d['summ']['stable_80']
    return d


# ------------------------------------------------------------------ a
def panel_a(c, d):
    c.letter()
    ax = c.ax(11, 10.5, 43, 37.5); clean(ax, ('left', 'bottom'))
    S, K = d['a_sum'], d['a_kde']
    for st in range(4):
        k = K[K.stage == st]; r = S.iloc[st]; col = SC4[st]
        wd = .40 * k.density / k.density.max()
        ax.fill_betweenx(k.ckm_ps_sd, st + .02, st + .02 + wd, color=col, alpha=.55, lw=0)
        ax.plot(st + .02 + wd, k.ckm_ps_sd, color=col, lw=.6)
        ax.plot([st - .13, st - .13], [r.p5, r.p95], color=col, lw=.7, solid_capstyle='butt')
        ax.add_patch(Rectangle((st - .2, r.p25), .14, r.p75 - r.p25, facecolor='white', edgecolor=col, lw=.8, zorder=3))
        ax.plot([st - .2, st - .06], [r.p50] * 2, color=INK, lw=.9, zorder=4)
    ax.plot(np.arange(4) - .13, S['p50'], color=INK, lw=.6, ls=(0, (2, 1.5)), zorder=2)
    ax.set_xlim(-.45, 3.55); ax.set_xticks(range(4)); ax.set_xticklabels([f'Stage {i}' for i in range(4)])
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_ylim(-3.2, 3.75); ax.set_yticks([-3, -2, -1, 0, 1, 2, 3]); ax.set_ylabel('CKM-PS (SD)', labelpad=1.5)
    for st in range(4):
        c.text(11 + 43 * (st + .45) / 4.0, 4.4, f'n = {int(S.n[st]):,}', ha='center', va='center', fontsize=5.4, color=MUTED)
    ax.text(-.38, 3.45, f"Spearman ρ = {d['summ']['spearman_psavg_stage']:.2f}", fontsize=6.0, va='top', color=INK)


# ------------------------------------------------------------------ b
def panel_b(c, d):
    c.letter()
    T = d['b_top']; n = len(T); Y = np.arange(n)[::-1].astype(float)
    Y[n // 2:] -= .7   # small gap between Higher and Lower blocks
    y0, hh = 7.5, 35.0
    ax = c.ax(20, y0, 44, hh); clean(ax, ('bottom',))
    lim = np.abs(T.mean_weight).max() * 1.12
    ax.axvline(0, color=AXIS, lw=.7, zorder=0)
    for y, r in zip(Y, T.itertuples()):
        col = HIGH if r.mean_weight > 0 else LOW
        ax.plot([0, r.mean_weight], [y, y], color=col, lw=1.0, solid_capstyle='butt', zorder=1)
        ax.scatter(r.mean_weight, y, s=6 + 26 * r.selection_freq ** 2, color=col, edgecolor='white', lw=.4, zorder=3)
    ax.set_ylim(Y.min() - .7, Y.max() + .7); ax.set_yticks(Y); ax.set_yticklabels(T.protein, fontsize=5.9)
    ax.tick_params(axis='y', length=0, pad=1.5)
    ax.set_xlim(-lim, lim); ax.set_xlabel('Mean elastic-net weight', labelpad=1.5)
    t = np.round(lim * .8, 2); ax.set_xticks([-t, 0, t]); ax.set_xticklabels([fmt(-t), '0', fmt(t)])
    # dot matrix: Reactome top-level membership + Figure 4 linkage
    m = c.ax(66, y0, 33, hh); clean(m, ()); m.set_xticks([]); m.set_yticks([])
    m.set_xlim(-.6, 7.6); m.set_ylim(Y.min() - .7, Y.max() + .7)
    for y in Y:
        m.axhline(y, color=GRID, lw=.5, zorder=0)
    for j, ct in enumerate(CATS):
        on = T['cat_' + ct].to_numpy()
        m.scatter(np.full(on.sum(), j), Y[on], s=11, color=CATCOL[ct], lw=0, zorder=2)
        m.scatter(np.full((~on).sum(), j), Y[~on], s=2.2, color='#D6DEE3', lw=0, zorder=1)
        c.text(66 + 33 * (j + .6) / 8.2, y0 + hh + .8, CATLAB[j], rotation=90, ha='center', va='bottom', fontsize=5.5, color=INK)
    for y, r in zip(Y, T.itertuples()):
        same = (r.mean_weight > 0 and r.f4_direction_class == 'all_higher') or (r.mean_weight < 0 and r.f4_direction_class == 'all_lower')
        inset = r.f4_direction_class in ('all_higher', 'all_lower')
        col = HIGH if r.mean_weight > 0 else LOW
        if inset:
            m.scatter(6.9, y, s=12, marker='D', facecolor=col if same else 'white', edgecolor=col, lw=.7, zorder=2)
        else:
            m.scatter(6.9, y, s=2.2, color='#D6DEE3', lw=0, zorder=1)
    c.text(66 + 33 * (6.9 + .6) / 8.2, y0 + hh + .8, 'Figure 4', rotation=90, ha='center', va='bottom', fontsize=5.5, color=INK, fontweight='bold')
    # stable-set composition (selection frequency >= 80%)
    B = d['b_bar']; tot = B.n.sum(); cols = ['#4E6B80', '#B7C6D0', '#E3E9ED']
    bx = c.ax(104.5, y0, 4.2, hh); clean(bx, ()); bx.set_xticks([]); bx.set_yticks([])
    bx.set_xlim(0, 1); bx.set_ylim(0, tot)
    top = tot
    for (i, r), col in zip(B.iterrows(), cols):
        bx.add_patch(Rectangle((0, top - r.n), 1, r.n, facecolor=col, edgecolor='white', lw=.6))
        c.text(109.6, y0 + hh * (top - r.n / 2) / tot, f'{r.n}', ha='left', va='center', fontsize=6.0, color=INK, fontweight='bold')
        c.text(111.2, y0 + hh * (top - r.n / 2) / tot - 2.6, ['Same', 'Opposite', 'Not in set'][i], ha='left', va='center', fontsize=5.3, color=MUTED)
        mk = c.ax(109.4, y0 + hh * (top - r.n / 2) / tot - 3.5, 1.8, 1.8); mk.axis('off'); mk.set_xlim(-1, 1); mk.set_ylim(-1, 1)
        if i < 2:
            mk.scatter(0, 0, s=7, marker='D', facecolor='#6F7F8A' if i == 0 else 'white', edgecolor='#6F7F8A', lw=.6)
        else:
            mk.scatter(0, 0, s=2.2, color='#C3CDD3', lw=0)
        top -= r.n
    c.text(106.6, y0 + hh + 1.2, f'Selected in\n≥80% of models\nn = {tot}', ha='center', va='bottom', fontsize=5.5, color=INK, linespacing=1.1)
    # legend (direction + selection frequency)
    hs = [Line2D([], [], marker='o', ls='', ms=3.6, mfc=HIGH, mec='white', mew=.4, label='Higher'),
          Line2D([], [], marker='o', ls='', ms=3.6, mfc=LOW, mec='white', mew=.4, label='Lower')]
    hs += [Line2D([], [], marker='o', ls='', ms=np.sqrt(6 + 26 * f ** 2), mfc='#9AA8B2', mec='white', mew=.3, label=lb) for f, lb in ((.5, '50%'), (1.0, '100% of models'))]
    lg = c.ax(14, y0 + hh + 2.0, 52, 8.0); lg.axis('off')
    lg.legend(handles=hs, loc='center', ncol=2, frameon=False, fontsize=5.6, handletextpad=.2, columnspacing=1.0, borderpad=0, labelspacing=.35)


# ------------------------------------------------------------------ c + d
def _rows():
    ys, labels, heads = {}, [], []
    y = 0.0
    for g, outs in GROUPS:
        heads.append((g, y)); y += 1.0
        for o in outs:
            ys[o] = y; y += 1.0
        y += .25
    return ys, heads, y - .25


def panel_cd(c, d):
    c.letter('c', 1.0); c.letter('d', 117.5)
    ys, heads, ymax = _rows(); y0, hh = 8.0, 47.0
    toy = lambda v: y0 + hh * (1 - (v + .5) / (ymax + .5))
    for g, yv in heads:
        c.text(2.0, toy(yv), g, fontsize=6.0, fontweight='bold', color=MUTED, va='center')
    H = d['h1'][d['h1'].model == 'M2: CKM-PS'].set_index('outcome')
    for o in ORDER:
        c.text(3.5, toy(ys[o]), LAB[o], fontsize=6.2, va='center')
        c.text(38.0, toy(ys[o]), f"{int(H.loc[o, 'events']):,}", fontsize=6.0, va='center', ha='right', color=MUTED)
    c.text(38.0, y0 + hh + 3.2, 'Events', fontsize=6.0, ha='right', va='center', color=MUTED)
    # c: HR per SD (M2)
    ax = c.ax(41, y0, 42, hh); clean(ax, ('bottom',))
    ax.set_xscale('log'); ax.set_xlim(.8, 2.2); ax.axvline(1, color=AXIS, lw=.7, zorder=0)
    ax.set_xticks([.8, 1, 1.25, 1.6, 2.0]); ax.set_xticklabels(['0.8', '1', '1.25', '1.6', '2.0']); ax.minorticks_off()
    ax.set_ylim(ymax + .5, -.5); ax.set_yticks([])
    for o in ORDER:
        r = H.loc[o]; sig = r.q_within_model_family < .05; y = ys[o]
        ax.plot([r.lo, min(r.hi, 2.2)], [y, y], color='#345C78', lw=1.0, solid_capstyle='butt', zorder=1)
        if r.hi > 2.2:
            ax.annotate('', xy=(2.2, y), xytext=(2.05, y), arrowprops=dict(arrowstyle='->', lw=.8, color='#345C78'))
        ax.scatter(r.HR, y, s=15, marker='o', facecolor='#345C78' if sig else 'white', edgecolor='#345C78', lw=.8, zorder=3)
        c.text(84.5, toy(y), f'{r.HR:.2f} ({r.lo:.2f}–{r.hi:.2f})', fontsize=5.9, va='center', color=INK if sig else MUTED)
        q = r.q_within_model_family
        c.text(112.5, toy(y), ('<0.001' if q < .001 else f'{q:.3f}'), fontsize=5.9, va='center', ha='right', color=INK if sig else MUTED)
    ax.set_xlabel('HR per SD of CKM-PS', labelpad=1.5)
    c.text(84.5, y0 + hh + 3.2, 'HR (95% CI)', fontsize=6.0, va='center', color=MUTED)
    c.text(112.5, y0 + hh + 3.2, 'q', fontsize=6.0, va='center', ha='right', color=MUTED, style='italic')
    # d: Delta C-index vs M2
    I = d['inc']; dx = c.ax(121, y0, 44, hh); clean(dx, ('bottom',))
    dx.axvline(0, color=AXIS, lw=.7, zorder=0); dx.set_ylim(ymax + .5, -.5); dx.set_yticks([])
    off = {'M1+PS': -.24, 'M3': 0.0, 'M2+GN': .24}
    sub = pd.concat([I[(I.new == a_) & (I.ref == b_)] for a_, b_ in DCOMP])
    lo_, hi_ = sub.dC_lo.min(), sub.dC_hi.max()
    span = max(abs(lo_), hi_); xl = (-.25 * span if lo_ > -.25 * span else lo_ * 1.15, hi_ * 1.12)
    dx.set_xlim(*xl)
    for o in ORDER:
        for mname, of in off.items():
            r = sub[(sub.outcome == o) & (sub.new == mname)].iloc[0]; y = ys[o] + of; col = MCOL[mname]
            sig = r.dC_lo > 0
            dx.plot([r.dC_lo, r.dC_hi], [y, y], color=col, lw=.9, solid_capstyle='butt', zorder=1)
            dx.scatter(r.dC, y, s=11 if mname != 'M3' else 15, marker=MMRK[mname], facecolor=col if sig else 'white', edgecolor=col, lw=.7, zorder=3)
        cm2 = sub[(sub.outcome == o) & (sub.new == 'M3')].C_ref.iloc[0]
        c.text(179.0, toy(ys[o]), f'{cm2:.3f}', fontsize=5.9, va='center', ha='right', color=MUTED)
    c.text(179.0, y0 + hh + 3.2, 'C, M2', fontsize=6.0, va='center', ha='right', color=MUTED)
    dx.set_xlabel('ΔC-index (95% CI)', labelpad=1.5)
    dx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v, 2) if abs(v) > 1e-9 else '0'))
    dx.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    hs = [Line2D([], [], marker=MMRK[k], ls='-', lw=.8, ms=3.4 if k == 'M3' else 3.0, color=MCOL[k], mfc=MCOL[k], label=MLAB[k]) for k in off]
    dx.legend(handles=hs, loc='center right', bbox_to_anchor=(1.0, .5), ncol=1, frameon=False, fontsize=5.6, handletextpad=.4,
              borderpad=0, handlelength=1.6, labelspacing=.5)


# ------------------------------------------------------------------ e
def panel_e(c, d):
    c.letter()
    G = d['grid']; G = G[G.tertile.astype(str).isin(['1', '2', '3'])].copy(); G['tertile'] = G.tertile.astype(int)
    x0, y0, w, hh = 29.5, 6.0, 72.0, 40.0
    ys, k = {}, 0.0
    for gi, (_, outs) in enumerate(GROUPS):
        for o in outs:
            ys[o] = k; k += 1.0
        k += .3
    ymax = k - .3
    ax = c.ax(x0, y0, w, hh); ax.set_xlim(0, 12); ax.set_ylim(ymax, 0); ax.axis('off')
    norm = LogNorm(vmin=.1, vmax=12)
    for o in ORDER:
        i = ys[o]
        c.text(x0 - 1.0, y0 + hh * (1 - (i + .5) / ymax), LAB[o], ha='right', va='center', fontsize=6.0)
        for st in range(4):
            for t in (1, 2, 3):
                r = G[(G.outcome == o) & (G.stage == st) & (G.tertile == t)].iloc[0]
                j = st * 3 + (t - 1); v = r.risk10_pct; lv = norm(max(v, .1)); low = r.events_10y < 10
                ax.add_patch(Rectangle((j + .04, i + .05), .92, .9, facecolor=RISK_CMAP(lv), edgecolor='none', alpha=.38 if low else 1))
                dark = lv > .47 and not low
                ax.text(j + .5, i + .52, (f'{v:.1f}' if v >= 1 else f'{v:.2f}'), ha='center', va='center', fontsize=5.2,
                        color='white' if dark else INK, alpha=.75 if low else 1)
    for st in range(4):
        xs = x0 + w * (st * 3) / 12
        c.text(x0 + w * (st * 3 + 1.5) / 12, y0 + hh + 6.0, f'Stage {st}', ha='center', va='center', fontsize=6.2, fontweight='bold', color=SC4[st] if st else MUTED)
        hb = c.ax(xs + .4, y0 + hh + 3.6, w * 3 / 12 - .8, .8); hb.axis('off'); hb.add_patch(Rectangle((0, 0), 1, 1, color=SC4[st], transform=hb.transAxes))
        for t in (1, 2, 3):
            c.text(x0 + w * (st * 3 + t - .5) / 12, y0 + hh + 1.6, f'T{t}', ha='center', va='center', fontsize=5.6, color=MUTED)
    cb = c.ax(x0 + w + 3.0, y0 + 4.5, 2.2, hh - 11.5)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=RISK_CMAP)
    cbar = plt.colorbar(sm, cax=cb); cbar.set_ticks([.1, .3, 1, 3, 10]); cbar.set_ticklabels(['0.1', '0.3', '1', '3', '10'])
    cbar.outline.set_linewidth(.5); cbar.outline.set_edgecolor(AXIS); cb.tick_params(labelsize=5.6, length=1.8, width=.6)
    cbar.solids.set_rasterized(False); cbar.solids.set_edgecolor('face'); cb.minorticks_off()
    c.text(x0 + w + 2.4, y0 + hh - 5.6, '10-year\nrisk (%)', ha='left', va='bottom', fontsize=5.6, color=MUTED, linespacing=1.05)
    sw = c.ax(x0 + w + 2.4, y0 - 1.6, 3.2, 2.4); sw.axis('off')
    sw.add_patch(Rectangle((0, 0), 1, 1, facecolor=RISK_CMAP(.55), alpha=.38, transform=sw.transAxes))
    c.text(x0 + w + 6.2, y0 - .4, '<10 events', ha='left', va='center', fontsize=5.3, color=MUTED)
    c.text(x0 + w / 2, 1.6, 'CKM-PS gap tertile within clinical stage', ha='center', va='center', fontsize=6.4)


# ------------------------------------------------------------------ g
def panel_g(c, d):
    c.letter()
    B = d['br']; outs = ['GMV', 'PCGMV', 'WMV_matter', 'VCSF']; ys = {o: i for i, o in enumerate(outs)}
    y0, hh = 8.0, 38.0
    ax = c.ax(27, y0, 36, hh); clean(ax, ('bottom',))
    ax.set_ylim(len(outs) - .4, -.6); ax.set_yticks([]); ax.axvline(0, color=AXIS, lw=.7, zorder=0)
    off = {'ckm_ps': -.15, 'f4_score': .15}; col = {'ckm_ps': MCOL['M3'], 'f4_score': MCOL['M2+F4']}; mk = {'ckm_ps': 'o', 'f4_score': 's'}
    for o in outs:
        c.text(25.5, y0 + hh * (1 - (ys[o] + .6) / (len(outs))), GLAB[o], ha='right', va='center', fontsize=6.0, linespacing=1.0)
        for e_, of in off.items():
            r = B[(B.exposure == e_) & (B.outcome == o)].iloc[0]; y = ys[o] + of; sig = r.q_family4 < .05
            ax.plot([r.lo, r.hi], [y, y], color=col[e_], lw=.9, solid_capstyle='butt')
            ax.scatter(r.beta, y, s=14 if e_ == 'ckm_ps' else 11, marker=mk[e_], facecolor=col[e_] if sig else 'white', edgecolor=col[e_], lw=.7, zorder=3)
    ax.set_xlim(-.13, .13); ax.set_xticks([-.1, 0, .1]); ax.set_xticklabels(['−0.1', '0', '0.1'])
    ax.set_xlabel('Standardised β (95% CI)', labelpad=1.5)
    hs = [Line2D([], [], marker='o', ls='-', lw=.8, ms=3.4, color=col['ckm_ps'], label='CKM-PS'),
          Line2D([], [], marker='s', ls='-', lw=.8, ms=3.0, color=col['f4_score'], label='Figure 4-anchored')]
    lg = c.ax(14, y0 + hh + 1.0, 52, 5.0); lg.axis('off')
    lg.legend(handles=hs, loc='center', ncol=2, frameon=False, fontsize=5.6, handletextpad=.3, columnspacing=.9, borderpad=0, handlelength=1.4)
    c.text(63, y0 + hh + 9.0, f"n = {int(B.n.iloc[0]):,}", ha='right', va='center', fontsize=5.6, color=MUTED)


# ------------------------------------------------------------------ f
F_OUT = ['Dementia', 'Stroke', 'Depression', 'Sleep_disorder']


def panel_f(c, d):
    c.letter()
    K = d['km']; G = d['grid']
    for i, o in enumerate(F_OUT):
        ax = c.ax(11 + i * 39.5, 8.0, 31.0, 30.0); clean(ax, ('left', 'bottom'))
        for t in (1, 2, 3):
            k = K[(K.outcome == o) & (K.stage == 2) & (K.tertile == t) & (K.time <= 12.0)]
            ax.step(k.time, k.cum_inc_pct, where='post', color=T3COL[t - 1], lw=1.05 if t != 2 else .8)
        ax.axvline(10, color=GRID, lw=.6, zorder=0)
        ax.set_xlim(0, 12.2); ax.set_xticks([0, 4, 8, 12]); ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('Years', labelpad=1.2)
        if i == 0:
            ax.set_ylabel('Cumulative incidence (%)', labelpad=1.5)
        ax.set_title(LAB[o], fontsize=6.4, fontweight='bold', pad=2.2, color=INK)
        r = G[(G.outcome == o) & (G.stage == 2) & (G.tertile.astype(str) == 'T3-T1')].iloc[0]
        sig = r.rd_lo > 0 or r.rd_hi < 0
        ax.text(.03, .97, f'10-year T3 − T1\n{fmt(r.risk10_pct, 1)}% ({fmt(r.rd_lo, 1)} to {fmt(r.rd_hi, 1)})', transform=ax.transAxes,
                ha='left', va='top', fontsize=5.6, color=INK if sig else MUTED, linespacing=1.1)
    hs = [Line2D([], [], color=T3COL[t - 1], lw=1.1, label=lb) for t, lb in ((1, 'T1 (lowest)'), (2, 'T2'), (3, 'T3 (highest)'))]
    lg = c.ax(161, 14, 17, 20); lg.axis('off')
    lg.legend(handles=hs, loc='center', ncol=1, frameon=False, fontsize=5.8, handlelength=1.3, handletextpad=.4, borderpad=0,
              title='Stage 2', title_fontsize=5.8)


FUN = {'a': panel_a, 'b': panel_b, 'cd': panel_cd, 'e': panel_e, 'g': panel_g, 'f': panel_f}
STEM = {'a': 'Panel_a_CKM-PS_by_Stage', 'b': 'Panel_b_Score_Composition_Figure4_Link', 'cd': 'Panel_c-d_Associations_and_Discrimination',
        'e': 'Panel_e_Within-Stage_Absolute_Risk', 'g': 'Panel_g_MRI_Bridge_Figure5', 'f': 'Panel_f_Stage2_Cumulative_Incidence'}


def save(fig, path):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(path.with_suffix('.' + ext), dpi=600)
    plt.close(fig); print(path.with_suffix('.png'), flush=True)


def export(root, d):
    sd = root / 'source_data'; sd.mkdir(exist_ok=True)
    d['a_sum'].to_csv(sd / 'Fig6a_CKM-PS_distribution_by_stage.csv', index=False)
    d['a_kde'].to_csv(sd / 'Fig6a_density_curves.csv', index=False)
    d['b_top'].to_csv(sd / 'Fig6b_16_displayed_proteins.csv', index=False)
    d['b_bar'].to_csv(sd / 'Fig6b_stable_set_composition.csv', index=False)
    d['w'].to_csv(sd / 'Fig6b_all_2923_protein_weights.csv', index=False)
    d['h1'][d['h1'].model == 'M2: CKM-PS'].to_csv(sd / 'Fig6c_9_HR_per_SD_M2.csv', index=False)
    pd.concat([d['inc'][(d['inc'].new == a_) & (d['inc'].ref == b_)] for a_, b_ in DCOMP]).to_csv(sd / 'Fig6d_27_deltaC.csv', index=False)
    d['grid'].to_csv(sd / 'Fig6e_risk_grid_108_cells_and_36_differences.csv', index=False)
    k = d['km']; k[(k.stage == 2) & k.outcome.isin(F_OUT)].to_csv(sd / 'Fig6f_stage2_cumulative_incidence.csv', index=False)
    d['br'].to_csv(sd / 'Fig6g_8_MRI_bridge_estimates.csv', index=False)


def main(args):
    root = args.root; d = load(root)
    for k in FUN:
        W, H = SIZE[k]; fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        FUN[k](Canvas(fig, 0, 0, k, W, H), d); save(fig, root / 'panels' / STEM[k])
    export(root, d)
    if not args.panels_only:
        FH = sum(max(SIZE[k][1] for k in row) for row in ROWS); FW = 180
        fig = plt.figure(figsize=(FW / 25.4, FH / 25.4)); top = FH
        for row in ROWS:
            rh = max(SIZE[k][1] for k in row); top -= rh; x = 0
            for k in row:
                FUN[k](Canvas(fig, x, top + rh - SIZE[k][1], k, FW, FH), d); x += SIZE[k][0]
        save(fig, root / 'figures/Figure6_CKM_Proteomic_Score_Neuropsychiatric_Risk_a-g')
        (root / 'plot_metadata.json').write_text(json.dumps(dict(width_mm=FW, height_mm=FH, panels=SIZE, rows=ROWS,
                                                                 km_outcomes=F_OUT, b_display_rule='top 8 Higher + top 8 Lower by |mean weight|'), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--panels-only', action='store_true')
    main(ap.parse_args())
