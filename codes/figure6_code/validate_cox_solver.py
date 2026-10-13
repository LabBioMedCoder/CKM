"""Validate the custom Breslow Cox solver against lifelines on real data (Dementia, M2 design)."""
import sys, numpy as np, pandas as pd, time
sys.path.insert(0, '/home/ubuntu/Figure6_Prediction/code')
from f6utils import *
from lifelines import CoxPHFitter
a = pd.read_parquet(f'{R}/private/analysis.parquet')
idx = np.arange(len(a))
for out in ['Dementia', 'Schizophrenia']:
    X, _ = design(a, idx, idx[:5], out, 'M2')
    T = a[out + '_time'].to_numpy(); E = a[out].to_numpy()
    t0 = time.time(); f = cox_fit(X, T, E); t1 = time.time() - t0
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])]); df['T'] = T; df['E'] = E
    t0 = time.time(); cph = CoxPHFitter(baseline_estimation_method='breslow').fit(df, 'T', 'E', fit_options={'step_size': 0.25, 'max_steps': 200}); t2 = time.time() - t0
    lb = cph.params_.to_numpy(); ls = cph.standard_errors_.to_numpy()
    # baseline cumulative hazard at 10 y: lifelines centres covariates; compare predicted 10-y risks instead
    r_own = 1 - np.exp(-f['H0'] * np.exp(X @ f['b']))
    r_ll = 1 - cph.predict_survival_function(df.iloc[:2000], times=[10.0]).to_numpy().ravel()
    print(out, 'p=', X.shape[1], f'own {t1:.2f}s lifelines {t2:.2f}s',
          'max|db|=%.2e' % np.abs(f['b'] - lb).max(), 'max|dse|=%.2e' % np.abs(f['se'] - ls).max(),
          'max|drisk10|=%.2e' % np.abs(r_own[:2000] - r_ll).max(), 'iters', f['iters'])
