#!/usr/bin/env python3
"""Supplementary figures for Figure 6 (same visual grammar as the main figure)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure6_editorial import (ROOT, SC4, INK, MUTED, AXIS, GRID, MCOL, GROUPS, ORDER, LAB, fmt, clean, save)

R = ROOT / 'results'; OUTD = ROOT / 'figures'; SD = ROOT / 'source_data'


def mmfig(w, h):
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    return fig, (lambda x, y, ww, hh: fig.add_axes([x / w, y / h, ww / w, hh / h])), (lambda x, y, s, **k: fig.text(x / w, y / h, s, **k))


def rows():
    ys, heads, y = {}, [], 0.0
    for g, outs in GROUPS:
        heads.append((g, y)); y += 1.0
        for o in outs:
            ys[o] = y; y += 1.0
        y += .25
    return ys, heads, y - .25


def label_column(T, ys, heads, ymax, y0, hh, x=2.0):
    toy = lambda v: y0 + hh * (1 - (v + .5) / (ymax + .5))
    for g, yv in heads:
        T(x, toy(yv), g, fontsize=6.0, fontweight='bold', color=MUTED, va='center')
    for o in ORDER:
        T(x + 1.5, toy(ys[o]), LAB[o], fontsize=6.2, va='center')


# ---------------------------------------------------------------- S1 calibration
def supp_calibration():
    cal = pd.read_csv(R / '05_calibration_deciles.csv'); perf = pd.read_csv(R / '05_model_performance.csv')
    W, H = 180, 168; fig, A, T = mmfig(W, H)
    for i, o in enumerate(ORDER):
        r, cidx = divmod(i, 3)
        ax = A(14 + cidx * 57, H - 13 - (r + 1) * 50 + 8, 40, 40); clean(ax, ('left', 'bottom'))
        g = cal[cal.outcome == o]; mx = max(g.mean_pred_pct.max(), g.obs_km_pct.max()) * 1.08
        ax.plot([0, mx], [0, mx], color=AXIS, lw=.7, ls=(0, (3, 2)), zorder=0)
        for m, col, mk, fc in (('M2', '#9AA8B2', 'o', 'white'), ('M3', MCOL['M3'], 'o', MCOL['M3'])):
            gg = g[g.model == m]
            ax.plot(gg.mean_pred_pct, gg.obs_km_pct, color=col, lw=.7, zorder=1)
            ax.scatter(gg.mean_pred_pct, gg.obs_km_pct, s=10, marker=mk, facecolor=fc, edgecolor=col, lw=.7, zorder=2)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        f = lambda v, _: f'{v:g}'
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f)); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
        ax.set_title(LAB[o], fontsize=6.6, fontweight='bold', pad=2.5)
        p = perf[(perf.outcome == o)].set_index('model')
        ax.text(.04, .96, f"Slope {p.loc['M2', 'cal_slope_mean']:.2f} / {p.loc['M3', 'cal_slope_mean']:.2f}\n"
                          f"O/E {p.loc['M2', 'OE10_mean']:.2f} / {p.loc['M3', 'OE10_mean']:.2f}",
                transform=ax.transAxes, va='top', fontsize=5.6, color=MUTED, linespacing=1.15)
        if cidx == 0: ax.set_ylabel('Observed 10-year risk (%)', labelpad=1.5)
        if r == 2: ax.set_xlabel('Predicted 10-year risk (%)', labelpad=1.5)
    hs = [Line2D([], [], marker='o', color='#9AA8B2', mfc='white', lw=.7, ms=3.2, label='M2 (clinical)'),
          Line2D([], [], marker='o', color=MCOL['M3'], mfc=MCOL['M3'], lw=.7, ms=3.2, label='M3 (M2 + CKM-PS)')]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(.5, 1 - 1.5 / H), ncol=2, frameon=False, fontsize=6.2)
    cal.to_csv(SD / 'SuppFig6_1_calibration_deciles.csv', index=False)
    save(fig, OUTD / 'Supplementary_Figure_Fig6_1_Calibration_M2_vs_M3')


# ---------------------------------------------------------------- S2 increment sensitivity
def supp_increments():
    inc = pd.read_csv(R / '05_increments.csv')
    cols = [('CKM-PS added to M2 (primary)', 'C', 'Harrell ΔC'),
            ('CKM-PS added to M2 (primary)', 'C_age', 'Age-stratified ΔC'),
            ('CKM-PS added to M2 (primary)', 'C_landmark2', '2-year landmark ΔC'),
            ('CKM-PS added to M2 (primary)', 'AUC10', 'Δ10-year AUC'),
            ('CKM-PS added to M2 (primary)', 'IPA10', 'Δ10-year IPA'),
            ('Figure 4-anchored score added to M2', 'C', 'Figure 4-anchored\nscore, ΔC'),
            ('Non-deconfounded score added to M2 (repeat 0)', 'C', 'Non-deconfounded\nscore, ΔC')]
    ys, heads, ymax = rows(); W, H = 180, 66; fig, A, T = mmfig(W, H); y0, hh = 9.0, 46.0
    label_column(T, ys, heads, ymax, y0, hh)
    lim = 0
    for comp, mt, _ in cols:
        g = inc[inc.comparison == comp]; lim = max(lim, np.abs(g[[f'd{mt}_lo', f'd{mt}_hi']].to_numpy()).max())
    lim = np.ceil(lim * 1.05 * 100) / 100
    out = []
    for j, (comp, mt, lab) in enumerate(cols):
        ax = A(36 + j * 20.5, y0, 18, hh); clean(ax, ('bottom',)); ax.set_yticks([])
        ax.set_ylim(ymax + .5, -.5); ax.axvline(0, color=AXIS, lw=.6, zorder=0); ax.set_xlim(-lim, lim)
        tk = lim / 2; ax.set_xticks([-tk, 0, tk]); ax.set_xticklabels([fmt(-tk, 3), '0', fmt(tk, 3)], fontsize=5.3)
        g = inc[inc.comparison == comp].set_index('outcome')
        col = MCOL['M2+F4'] if 'Figure 4' in comp else ('#8C9AA5' if 'Non-de' in comp else MCOL['M3'])
        for o in ORDER:
            r = g.loc[o]; sig = r[f'd{mt}_lo'] > 0 or r[f'd{mt}_hi'] < 0
            ax.plot([r[f'd{mt}_lo'], r[f'd{mt}_hi']], [ys[o]] * 2, color=col, lw=.9, solid_capstyle='butt')
            ax.scatter(r[f'd{mt}'], ys[o], s=11, facecolor=col if sig else 'white', edgecolor=col, lw=.7, zorder=3)
            out.append(dict(column=lab, outcome=o, comparison=comp, metric=mt, estimate=r[f'd{mt}'], lo=r[f'd{mt}_lo'], hi=r[f'd{mt}_hi'], J=r.J))
        T(36 + j * 20.5 + 9, y0 + hh + 3.4, lab, ha='center', va='center', fontsize=5.7, fontweight='bold' if j == 0 else 'normal', linespacing=1.05)
    T(36 + 3.5 * 20.5 - 1, 2.2, 'Difference vs M2 (95% CI)', ha='center', va='center', fontsize=6.4)
    pd.DataFrame(out).to_csv(SD / 'SuppFig6_2_increment_sensitivity.csv', index=False)
    save(fig, OUTD / 'Supplementary_Figure_Fig6_2_Incremental_Performance_Sensitivity')


# ---------------------------------------------------------------- S3 within-stage risk differences
def supp_rd():
    g = pd.read_csv(R / '04_H3_risk_grid.csv'); g = g[g.tertile == 'T3-T1']
    ys, heads, ymax = rows(); W, H = 180, 66; fig, A, T = mmfig(W, H); y0, hh = 9.0, 46.0
    label_column(T, ys, heads, ymax, y0, hh)
    for st in range(4):
        gs = g[g.stage == st].set_index('outcome')
        lo_, hi_ = gs.rd_lo.min(), gs.rd_hi.max(); pad = .08 * (hi_ - lo_)
        ax = A(38 + st * 35.5, y0, 31, hh); clean(ax, ('bottom',)); ax.set_yticks([])
        ax.set_ylim(ymax + .5, -.5); ax.axvline(0, color=AXIS, lw=.6, zorder=0); ax.set_xlim(lo_ - pad, hi_ + pad)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v, 0) if abs(v - round(v)) < 1e-9 else fmt(v, 1)))
        for o in ORDER:
            r = gs.loc[o]; sig = r.rd_lo > 0 or r.rd_hi < 0
            ax.plot([r.rd_lo, r.rd_hi], [ys[o]] * 2, color=SC4[st], lw=.9, solid_capstyle='butt')
            ax.scatter(r.risk10_pct, ys[o], s=12, marker=['o', 's', 'D', 'o'][st], facecolor=SC4[st] if sig else 'white', edgecolor=SC4[st], lw=.7, zorder=3)
        T(38 + st * 35.5 + 15.5, y0 + hh + 3.0, f'Stage {st}', ha='center', va='center', fontsize=6.4, fontweight='bold', color=SC4[st] if st else MUTED)
    T(38 + 2 * 35.5 - 2, 2.2, '10-year risk difference, T3 − T1 (percentage points, 95% bootstrap CI)', ha='center', va='center', fontsize=6.4)
    g.to_csv(SD / 'SuppFig6_3_within_stage_risk_differences.csv', index=False)
    save(fig, OUTD / 'Supplementary_Figure_Fig6_3_Within-Stage_Risk_Differences')


# ---------------------------------------------------------------- S4 association sensitivity
def supp_h1():
    h = pd.read_csv(R / '04_H1_associations.csv')
    cols = [('M2: CKM-PS', 'CKM-PS, M2 (primary)', MCOL['M3']), ('M1: CKM-PS', 'CKM-PS, M1', MCOL['M1+PS']),
            ('M2: CKM-PS, 2-year landmark', 'CKM-PS, M2, landmark', MCOL['M3']),
            ('M2: Figure 4-anchored score', 'Figure 4-anchored, M2', MCOL['M2+F4']),
            ('M2: GFAP', 'GFAP, M2', MCOL['M2+GN']), ('M2: NEFL', 'NEFL, M2', MCOL['M2+GN'])]
    ys, heads, ymax = rows(); W, H = 180, 66; fig, A, T = mmfig(W, H); y0, hh = 9.0, 46.0
    label_column(T, ys, heads, ymax, y0, hh)
    lo_, hi_ = h.lo.min(), h.hi.max(); xl = (min(.6, lo_ * .95), min(max(2.4, hi_ * 1.02), 3.2))
    for j, (m, lab, col) in enumerate(cols):
        ax = A(37 + j * 23.8, y0, 21, hh); clean(ax, ('bottom',)); ax.set_yticks([])
        ax.set_xscale('log'); ax.set_xlim(*xl); ax.minorticks_off(); ax.set_xticks([.7, 1, 1.5, 2.5]); ax.set_xticklabels(['0.7', '1', '1.5', '2.5'], fontsize=5.3)
        ax.set_ylim(ymax + .5, -.5); ax.axvline(1, color=AXIS, lw=.6, zorder=0)
        g = h[h.model == m].set_index('outcome')
        for o in ORDER:
            r = g.loc[o]; sig = r.q_within_model_family < .05
            ax.plot([r.lo, min(r.hi, xl[1])], [ys[o]] * 2, color=col, lw=.9, solid_capstyle='butt')
            ax.scatter(r.HR, ys[o], s=11, facecolor=col if sig else 'white', edgecolor=col, lw=.7, zorder=3)
        T(37 + j * 23.8 + 10.5, y0 + hh + 3.0, lab, ha='center', va='center', fontsize=5.7, fontweight='bold' if j == 0 else 'normal')
    T(37 + 3 * 23.8 - 1, 2.2, 'HR per SD (95% CI)', ha='center', va='center', fontsize=6.4)
    h.to_csv(SD / 'SuppFig6_4_association_sensitivity.csv', index=False)
    save(fig, OUTD / 'Supplementary_Figure_Fig6_4_Association_Sensitivity')


if __name__ == '__main__':
    supp_calibration(); supp_increments(); supp_rd(); supp_h1()
