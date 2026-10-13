"""Figure 6 step 3: outcome-specific Cox models evaluated in the outer test folds.

For each outcome x repeat x outer fold the models below are fitted in the outer training fold
(CKM-PS = cross-fitted inner-validation predictions; no in-sample optimism) and evaluated in the
outer test fold. All models share the same participants (common analytic sample).
"""
import sys, json, time, numpy as np, pandas as pd, warnings
from pathlib import Path
from joblib import Parallel, delayed
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sksurv.util import Surv
sys.path.insert(0, '/home/ubuntu/Figure6_Prediction/code')
from f6utils import *
warnings.filterwarnings('ignore')
P = Path(R); FD = P / 'private/score_folds'; OD = P / 'private/cox_folds'; OD.mkdir(exist_ok=True)
PEN = 1.0   # light ridge, identical for all outcomes/models (numerical stability)
MODELS = ['M0', 'M1', 'M1+PS', 'M2', 'M3', 'M2+F4', 'M2+GN', 'M2+PSraw']
_cache = {}


def data():
    if 'a' not in _cache:
        _cache['a'] = pd.read_parquet(P / 'private/analysis.parquet')
        _cache['gn'] = np.load(P / 'private/gfap_nefl.npy')
    return _cache['a'], _cache['gn']


def metrics(Ttr, Etr, Tte, Ete, lp, risk, age):
    out = dict(C=harrell_c(Tte, Ete, lp), C_age=age_stratified_c(Tte, Ete, lp, age))
    s_tr = Surv.from_arrays(Etr.astype(bool), Ttr); s_te = Surv.from_arrays(Ete.astype(bool), Tte)
    out['AUC10'] = float(cumulative_dynamic_auc(s_tr, s_te, lp, [HORIZON])[0][0])
    b = brier_score(s_tr, s_te, (1 - risk)[:, None], [HORIZON])[1][0]
    b0 = brier_score(s_tr, s_te, np.full((len(Tte), 1), 1 - km_risk(Ttr, Etr)), [HORIZON])[1][0]
    out.update(Brier10=float(b), IPA10=float(1 - b / b0))
    out['cal_slope'] = float(cox_fit(lp[:, None], Tte, Ete)['b'][0])
    out['OE10'] = float(km_risk(Tte, Ete) / risk.mean())
    lm = Tte > 2.0   # 2-year landmark: events and follow-up after year 2 only
    out['C_landmark2'] = harrell_c(Tte[lm], Ete[lm], lp[lm])
    return out


def run(outcome, rep, fold):
    js = OD / f'{outcome}_rep{rep}_fold{fold}.json'
    if js.exists():
        return json.loads(js.read_text())
    a, gn = data()
    z = np.load(FD / f'rep{rep}_fold{fold}.npz')
    tr, te = z['tr'], z['te']
    T = a[outcome + '_time'].to_numpy(); E = a[outcome].to_numpy().astype(float); age = a.Age.to_numpy()
    extras = {'M1+PS': ('M1', z['ps_train'][:, None], z['ps_test'][:, None]),
              'M3': ('M2', z['ps_train'][:, None], z['ps_test'][:, None]),
              'M2+F4': ('M2', z['f4_train'][:, None], z['f4_test'][:, None]),
              'M2+GN': ('M2', gn[tr], gn[te])}
    if rep == 0:
        extras['M2+PSraw'] = ('M2', z['raw_train'][:, None], z['raw_test'][:, None])
    rows, oof = [], {}
    for m in MODELS:
        if m in ('M0', 'M1', 'M2'):
            base, xtr, xte = m, None, None
        elif m in extras:
            base, xtr, xte = extras[m]
        else:
            continue
        Xtr, Xte = design(a, tr, te, outcome, base, xtr, xte)
        f = cox_fit(Xtr, T[tr], E[tr], pen=PEN)
        lp = Xte @ f['b']; risk = 1 - np.exp(-f['H0'] * np.exp(lp))
        r = dict(outcome=outcome, rep=rep, fold=fold, model=m, n_test=len(te), events_test=int(E[te].sum()),
                 iters=f['iters'], **metrics(T[tr], E[tr], T[te], E[te], lp, risk, age[te]))
        if xtr is not None:  # coefficient of the added predictor(s) (per training SD)
            r['added_beta'] = float(f['b'][-xtr.shape[1]])
        rows.append(r); oof[m] = risk.astype(np.float32)
    np.savez_compressed(OD / f'{outcome}_rep{rep}_fold{fold}.npz', te=te, **{k.replace('+', '_'): v for k, v in oof.items()})
    js.write_text(json.dumps(rows))
    return rows


if __name__ == '__main__':
    a = pd.read_parquet(P / 'private/analysis.parquet')
    jobs = [(o, r, k) for o in OUT for r in range(10) for k in range(5)]
    t0 = time.time(); allrows = []
    for i, rows in enumerate(Parallel(n_jobs=6, return_as='generator')(delayed(run)(*j) for j in jobs)):
        allrows += rows
        if i % 25 == 0:
            print(f'{i + 1}/{len(jobs)} tasks, {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(P / 'results/03_fold_metrics.csv', index=False)
    print('DONE', df.shape, flush=True)
