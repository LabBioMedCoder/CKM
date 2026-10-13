"""Figure 6 step 5: summarise outer-test-fold performance.

Increments (e.g. dC = C[M3] - C[M2]) are computed per outer test fold, averaged over the
J = 50 folds (10 repeats x 5) and given 95% intervals by the Nadeau-Bengio corrected
resampled t statistic: var = (1/J + n_test/n_train) * s^2, t with J-1 df.
Comparisons that exist only for repeat 0 (non-deconfounded score) use J = 5.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
sys.path.insert(0, '/home/ubuntu/Figure6_Prediction/code')
from f6utils import *
P = Path(R); RES = P / 'results'; OD = P / 'private/cox_folds'
fm = pd.read_csv(RES / '03_fold_metrics.csv')
assert len(fm) == 9 * (50 * 7 + 5), len(fm)
METRICS = ['C', 'C_age', 'AUC10', 'IPA10', 'C_landmark2']
COMP = [('M3', 'M2', 'CKM-PS added to M2 (primary)'), ('M2+F4', 'M2', 'Figure 4-anchored score added to M2'),
        ('M2+GN', 'M2', 'GFAP + NEFL added to M2'), ('M1+PS', 'M1', 'CKM-PS added to M1'),
        ('M1', 'M0', 'CKM stage added to M0'), ('M2', 'M1', 'Clinical risk factors added to M1'),
        ('M2+PSraw', 'M2', 'Non-deconfounded score added to M2 (repeat 0)')]


def nb_interval(d, ratio=0.25):
    d = np.asarray(d); J = len(d); m = d.mean(); s2 = d.var(ddof=1)
    se = np.sqrt((1 / J + ratio) * s2); tq = stats.t.ppf(.975, J - 1)
    p = 2 * stats.t.sf(abs(m / se), J - 1) if se > 0 else np.nan
    return m, m - tq * se, m + tq * se, p, J


rows = []
for o in OUT:
    g = fm[fm.outcome == o]
    for new, ref, lab in COMP:
        A = g[g.model == new].set_index(['rep', 'fold']); Bm = g[g.model == ref].set_index(['rep', 'fold'])
        A, Bm = A.align(Bm, join='inner', axis=0)
        r = dict(outcome=o, comparison=lab, new=new, ref=ref, J=len(A))
        for mt in METRICS:
            m, lo, hi, p, J = nb_interval(A[mt] - Bm[mt])
            r.update({f'd{mt}': m, f'd{mt}_lo': lo, f'd{mt}_hi': hi, f'd{mt}_p': p,
                      f'{mt}_new': A[mt].mean(), f'{mt}_ref': Bm[mt].mean()})
        rows.append(r)
inc = pd.DataFrame(rows)
inc['dC_q_primary'] = np.nan
pm = inc.comparison == 'CKM-PS added to M2 (primary)'
inc.loc[pm, 'dC_q_primary'] = bh(inc.loc[pm, 'dC_p'])
inc.to_csv(RES / '05_increments.csv', index=False)

# absolute performance and calibration per model
perf = fm.groupby(['outcome', 'model'])[['C', 'C_age', 'AUC10', 'IPA10', 'cal_slope', 'OE10', 'C_landmark2', 'events_test']].agg(['mean', 'std'])
perf.columns = [f'{a}_{b}' for a, b in perf.columns]
perf.reset_index().to_csv(RES / '05_model_performance.csv', index=False)

# pooled out-of-fold calibration (risk averaged over 10 repeats), deciles, M2 vs M3
a = pd.read_parquet(P / 'private/analysis.parquet'); n = len(a); cal = []
for o in OUT:
    risk = {m: np.zeros((10, n)) for m in ['M2', 'M3']}
    for rep in range(10):
        for fold in range(5):
            z = np.load(OD / f'{o}_rep{rep}_fold{fold}.npz')
            for m in risk: risk[m][rep, z['te']] = z[m]
    T = a[o + '_time'].to_numpy(); E = a[o].to_numpy()
    for m in risk:
        r = risk[m].mean(0); dec = pd.qcut(r, 10, labels=False)
        for k in range(10):
            s = dec == k
            cal.append(dict(outcome=o, model=m, decile=k + 1, n=int(s.sum()), mean_pred_pct=100 * r[s].mean(),
                            obs_km_pct=100 * km_risk(T[s], E[s]), events_10y=int(((E[s] == 1) & (T[s] <= HORIZON)).sum())))
pd.DataFrame(cal).to_csv(RES / '05_calibration_deciles.csv', index=False)
pd.set_option('display.width', 250)
show = inc[['outcome', 'comparison', 'J', 'C_ref', 'C_new', 'dC', 'dC_lo', 'dC_hi', 'dC_p', 'dC_age', 'dC_age_lo', 'dC_age_hi',
            'dAUC10', 'dIPA10', 'dC_landmark2', 'dC_landmark2_lo', 'dC_landmark2_hi']]
print(show.round(4).to_string(index=False))
