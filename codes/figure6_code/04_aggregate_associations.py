"""Figure 6 step 4: score aggregation, H1 associations, H3 within-stage re-stratification, MRI bridge.

All participant-level objects stay in private/. results/ contains aggregate tables only.
"""
import sys, json, numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy import stats
sys.path.insert(0, '/home/ubuntu/Figure6_Prediction/code')
from f6utils import *
warnings.filterwarnings('ignore')
P = Path(R); FD = P / 'private/score_folds'; RES = P / 'results'
a = pd.read_parquet(P / 'private/analysis.parquet'); n = len(a)
names = open(P / 'private/protein_names.txt').read().split('\n')

# ---------------- A. aggregate the 50 outer score models ----------------
ps = np.zeros((10, n)); f4 = np.zeros((10, n)); raw = np.full(n, np.nan)
coefs, meta, f4sel = [], [], []
for rep in range(10):
    for fold in range(5):
        z = np.load(FD / f'rep{rep}_fold{fold}.npz')
        ps[rep, z['te']] = z['ps_test']; f4[rep, z['te']] = z['f4_test']
        if rep == 0: raw[z['te']] = z['raw_test']
        coefs.append(z['coef']); f4sel.append(z['f4_sel'])
        meta.append(dict(rep=rep, fold=fold, nnz=int(z['nnz']), alpha=float(z['alpha']), alpha_index=int(z['alpha_index']),
                         r2_inner=float(z['r2_inner']), f4_n=int(z['f4_n']), raw_nnz=int(z['raw_nnz']) if rep == 0 else np.nan))
meta = pd.DataFrame(meta); meta.to_csv(RES / '04_score_models_50.csv', index=False)
assert (ps != 0).all(axis=1).all()
C = np.vstack(coefs); F4S = np.vstack(f4sel)
ps_avg = ps.mean(0); f4_avg = f4.mean(0)
priv = pd.DataFrame({'Eid': a.Eid, 'ckm_ps': ps_avg, 'f4_score': f4_avg, 'ckm_ps_raw_rep0': raw})
priv.to_parquet(P / 'private/oof_scores.parquet', index=False)

# Figure 4 linkage
f4b = pd.read_csv('/home/ubuntu/Figure4_Editorial_Redesign/source_data/Fig4b_membership_2923_proteins.csv')
f4cls = dict(zip(f4b.protein, f4b.direction_class))
mp = pd.read_csv('/home/ubuntu/CKM_Protein_Function_Brain/tables/protein_HGNC_Reactome_mapping.csv').fillna('')
sym = dict(zip(mp.protein, mp.symbol))
pa = json.loads(open('/home/ubuntu/CKM_Protein_Function_Brain/annotations/assayed_pathways.json').read())
rootname = {k: pa['names'][k] for k in pa['roots']}
CAT6 = ['Immune System', 'Metabolism', 'Hemostasis', 'Extracellular matrix organization', 'Signal Transduction', 'Neuronal System']
catsets = {rootname[k]: set(v) for k, v in pa['roots'].items() if rootname[k] in CAT6}
assert set(catsets) == set(CAT6)
w = pd.DataFrame({'protein': names, 'selection_freq': (C != 0).mean(0), 'mean_weight': C.mean(0),
                  'sd_weight': C.std(0), 'f4_direction_class': [f4cls.get(p, 'not_in_table') for p in names],
                  'f4_anchor_freq': F4S.mean(0)})
for c in CAT6: w['cat_' + c] = [sym.get(p, '') in catsets[c] for p in names]
w['direction'] = np.where(w.mean_weight > 0, 'Higher', np.where(w.mean_weight < 0, 'Lower', 'none'))
w = w.sort_values('mean_weight', key=np.abs, ascending=False)
w.to_csv(RES / '04_protein_weights_2923.csv', index=False)
stable = w[w.selection_freq >= 0.8]
cons = stable.f4_direction_class.isin(['all_higher', 'all_lower'])
concord = ((stable.direction == 'Higher') & (stable.f4_direction_class == 'all_higher')) | \
          ((stable.direction == 'Lower') & (stable.f4_direction_class == 'all_lower'))
rho_stage = [stats.spearmanr(ps[r], a.stage)[0] for r in range(10)]
summ = dict(n=n, n_models=len(meta), nnz_median=float(meta.nnz.median()), nnz_range=[int(meta.nnz.min()), int(meta.nnz.max())],
            alpha_index_values=sorted(meta.alpha_index.unique().tolist()), r2_inner_median=float(meta.r2_inner.median()),
            spearman_ps_stage_per_repeat_median=float(np.median(rho_stage)),
            spearman_psavg_stage=float(stats.spearmanr(ps_avg, a.stage)[0]),
            pearson_psavg_age=float(np.corrcoef(ps_avg, a.Age)[0, 1]),
            pearson_psraw_age=float(np.corrcoef(raw, a.Age)[0, 1]),
            spearman_psraw_stage=float(stats.spearmanr(raw, a.stage)[0]),
            pointbiserial_psavg_sex=float(np.corrcoef(ps_avg, a.Sex)[0, 1]),
            ever_selected=int((w.selection_freq > 0).sum()), stable_80=int(len(stable)),
            stable_in_f4_consistent=int(cons.sum()), stable_direction_concordant_with_f4=int(concord.sum()),
            stable_higher=int((stable.direction == 'Higher').sum()), stable_lower=int((stable.direction == 'Lower').sum()),
            f4_anchor_n_median=float(meta.f4_n.median()), f4_anchor_n_range=[int(meta.f4_n.min()), int(meta.f4_n.max())],
            gfap_freq=float(w.set_index('protein').loc['GFAP', 'selection_freq']),
            nefl_freq=float(w.set_index('protein').loc['NEFL', 'selection_freq']),
            corr_psavg_f4avg=float(np.corrcoef(ps_avg, f4_avg)[0, 1]))
for s in range(4):
    summ[f'ps_median_stage{s}'] = float(np.median(ps_avg[a.stage == s]))

# ---------------- B. H1: associations in the full sample (unpenalised Cox) ----------------
zs = lambda v: (v - v.mean()) / v.std()
gn = np.load(P / 'private/gfap_nefl.npy')
idx = np.arange(n); rows = []
for o in OUT:
    T = a[o + '_time'].to_numpy(); E = a[o].to_numpy().astype(float)
    specs = {'M2: CKM-PS': ('M2', zs(ps_avg)[:, None]), 'M1: CKM-PS': ('M1', zs(ps_avg)[:, None]),
             'M2: Figure 4-anchored score': ('M2', zs(f4_avg)[:, None]),
             'M2: GFAP': ('M2', zs(gn[:, 0])[:, None]), 'M2: NEFL': ('M2', zs(gn[:, 1])[:, None])}
    for lab, (base, x) in specs.items():
        X, _ = design(a, idx, idx[:2], o, base)
        X = np.column_stack([X, x])
        f = cox_fit(X, T, E)
        b, se = f['b'][-1], f['se'][-1]
        rows.append(dict(outcome=o, model=lab, n=n, events=int(E.sum()), beta=b, se=se, HR=np.exp(b),
                         lo=np.exp(b - 1.959964 * se), hi=np.exp(b + 1.959964 * se), p=2 * stats.norm.sf(abs(b / se))))
    lm = T > 2.0   # 2-year landmark
    X, _ = design(a, idx[lm], idx[:2], o, 'M2'); X = np.column_stack([X, zs(ps_avg)[lm][:, None]])
    f = cox_fit(X, T[lm], E[lm]); b, se = f['b'][-1], f['se'][-1]
    rows.append(dict(outcome=o, model='M2: CKM-PS, 2-year landmark', n=int(lm.sum()), events=int(E[lm].sum()), beta=b, se=se,
                     HR=np.exp(b), lo=np.exp(b - 1.959964 * se), hi=np.exp(b + 1.959964 * se), p=2 * stats.norm.sf(abs(b / se))))
h1 = pd.DataFrame(rows)
h1['q_within_model_family'] = np.nan
for m, g in h1.groupby('model'):
    h1.loc[g.index, 'q_within_model_family'] = bh(g.p)
h1.to_csv(RES / '04_H1_associations.csv', index=False)

# ---------------- C. H3: within-stage re-stratification ----------------
# gap: CKM-PS residual after stage-specific regression on age (restricted cubic spline) and sex,
# so that tertiles within each stage are balanced on age and sex by construction
zps = zs(ps_avg); gap = np.zeros(n)
for s in range(4):
    m = (a.stage == s).to_numpy(); ag = a.Age.to_numpy()[m]
    Ds = np.column_stack([np.ones(m.sum()), rcs(ag, np.percentile(ag, [10, 50, 90])), a.Sex.to_numpy()[m]])
    gap[m] = zps[m] - Ds @ np.linalg.lstsq(Ds, zps[m], rcond=None)[0]
tert = np.zeros(n, int)
for s in range(4):
    m = (a.stage == s).to_numpy(); tert[m] = pd.qcut(gap[m], 3, labels=False)
priv['gap'] = gap; priv['gap_tertile'] = tert + 1; priv.to_parquet(P / 'private/oof_scores.parquet', index=False)
bal = pd.DataFrame([dict(stage=s, tertile=t + 1, n=int(((a.stage == s) & (tert == t)).sum()),
                         mean_age=float(a.Age[(a.stage == s) & (tert == t)].mean()),
                         pct_male=float(100 * a.Sex[(a.stage == s) & (tert == t)].mean()),
                         median_gap=float(np.median(gap[(a.stage == s).to_numpy() & (tert == t)])))
                    for s in range(4) for t in range(3)])
bal.to_csv(RES / '04_H3_tertile_balance.csv', index=False)
rng = np.random.default_rng(20261012); B = 1000; grid = []
for o in OUT:
    T = a[o + '_time'].to_numpy(); E = a[o].to_numpy()
    for s in range(4):
        cell = {}
        for t in range(3):
            m = np.where((a.stage == s).to_numpy() & (tert == t))[0]; cell[t] = m
            grid.append(dict(outcome=o, stage=s, tertile=t + 1, n=len(m), events_total=int(E[m].sum()),
                             events_10y=int(((E[m] == 1) & (T[m] <= HORIZON)).sum()), risk10_pct=100 * km_risk(T[m], E[m])))
        r1, r3 = km_risk(T[cell[0]], E[cell[0]]), km_risk(T[cell[2]], E[cell[2]])
        bs = []
        for _ in range(B):
            i1 = rng.choice(cell[0], len(cell[0])); i3 = rng.choice(cell[2], len(cell[2]))
            bs.append(km_risk(T[i3], E[i3]) - km_risk(T[i1], E[i1]))
        grid.append(dict(outcome=o, stage=s, tertile='T3-T1', n=len(cell[0]) + len(cell[2]),
                         risk10_pct=100 * (r3 - r1), rd_lo=100 * np.percentile(bs, 2.5), rd_hi=100 * np.percentile(bs, 97.5),
                         risk_ratio=r3 / r1 if r1 > 0 else np.nan))
pd.DataFrame(grid).to_csv(RES / '04_H3_risk_grid.csv', index=False)
# KM curves (step functions) for all outcomes x stage x tertile
curves = []
tg = np.round(np.arange(0, 14.01, 0.05), 2)
for o in OUT:
    T = a[o + '_time'].to_numpy(); E = a[o].to_numpy()
    for s in range(4):
        for t in range(3):
            m = (a.stage == s).to_numpy() & (tert == t)
            Ts, Es = T[m], E[m]; o_ = np.argsort(Ts); Ts, Es = Ts[o_], Es[o_]
            ut, ix = np.unique(Ts, return_index=True); d = np.add.reduceat(Es.astype(float), ix); nr = len(Ts) - ix
            S = np.cumprod(1 - d / nr); cum = 1 - S
            val = np.array([cum[ut <= g][-1] if (ut <= g).any() else 0.0 for g in tg])
            nar = np.array([(Ts >= g).sum() for g in tg])
            curves.append(pd.DataFrame({'outcome': o, 'stage': s, 'tertile': t + 1, 'time': tg, 'cum_inc_pct': 100 * val, 'n_at_risk': nar}))
pd.concat(curves).to_csv(RES / '04_H3_km_curves.csv', index=False)

# ---------------- D. MRI bridge (Figure 5 sample and CKM-adjusted design) ----------------
mri = pd.read_parquet('/home/ubuntu/CKM_Protein_Function_Brain/private/matched_modules.parquet')
mri['eid'] = pd.to_numeric(mri.eid).astype('int64')
br = mri.merge(priv[['Eid', 'ckm_ps', 'f4_score']], left_on='eid', right_on='Eid', how='inner')
GLOBAL = ['GMV', 'PCGMV', 'WMV_matter', 'VCSF']
zz = lambda v: (np.asarray(v, float) - np.mean(v)) / np.std(v, ddof=1)
def fig5_design(df):  # identical to Figure 5 'M2_CKM' (CKM-adjusted) model
    arr = pd.DataFrame({c: zz(df[c]) for c in ['Age', 'APOE4', 'SES_PCA', 'ICV']}, index=df.index)
    for c in ['Sex', 'Ethnicity', 'Smoking_status', 'Alcohol_intake_frequency', 'stage']:
        arr = pd.concat([arr, pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)], axis=1)
    arr.insert(0, 'Intercept', 1.0); return arr.to_numpy(float)
Dm = fig5_design(br); assert np.linalg.matrix_rank(Dm) == Dm.shape[1]
Q, _ = np.linalg.qr(Dm); hc = (Q * Q).sum(1); dfree = len(br) - Dm.shape[1] - 1; crit = stats.t.ppf(.975, dfree)
brows = []
for expo in ['ckm_ps', 'f4_score']:
    x = zz(br[expo]); xr = x - Q @ (Q.T @ x); ss = xr @ xr; lev = hc + xr ** 2 / ss
    for g in GLOBAL:
        y = zz(br[g]); yr = y - Q @ (Q.T @ y); b = xr @ yr / ss; e = yr - xr * b
        se = np.sqrt(((xr * e / (1 - lev)) ** 2).sum() / ss ** 2); tv = b / se
        brows.append(dict(exposure=expo, outcome=g, n=len(br), beta=b, se=se, lo=b - crit * se, hi=b + crit * se,
                          p=2 * stats.t.sf(abs(tv), dfree)))
bridge = pd.DataFrame(brows); bridge['q_family4'] = np.nan
for e_, g in bridge.groupby('exposure'): bridge.loc[g.index, 'q_family4'] = bh(g.p)
bridge.to_csv(RES / '04_MRI_bridge.csv', index=False)
summ['mri_bridge_n'] = int(len(br)); summ['mri_bridge_stage_counts'] = br.stage.value_counts().to_dict()
json.dump(summ, open(RES / '04_score_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)); print(h1[h1.model == 'M2: CKM-PS'].round(4).to_string(index=False))
print(bridge.round(4).to_string(index=False)); print(bal.round(2).to_string(index=False))
