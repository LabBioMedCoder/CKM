"""Shared utilities for Figure 6: design matrices, fast Cox PH (Breslow), performance metrics."""
import numpy as np, pandas as pd
from lifelines.utils import concordance_index

R = '/home/ubuntu/Figure6_Prediction'
OUT = ['Dementia', 'Alzheimers_disease', 'Parkinsons_disease', 'Stroke', 'Anxiety_disorder',
       'Bipolar_disorder', 'Depression', 'Schizophrenia', 'Sleep_disorder']
LABEL = {'Dementia': 'Dementia (all-cause)', 'Alzheimers_disease': "Alzheimer's disease",
         'Parkinsons_disease': "Parkinson's disease", 'Stroke': 'Stroke', 'Anxiety_disorder': 'Anxiety disorder',
         'Bipolar_disorder': 'Bipolar disorder', 'Depression': 'Depression', 'Schizophrenia': 'Schizophrenia',
         'Sleep_disorder': 'Sleep disorder'}
APOE_OUT = {'Dementia', 'Alzheimers_disease'}
HORIZON = 10.0


def rcs(x, knots):
    """Restricted cubic spline (Harrell), 3 knots -> linear term + 1 non-linear term."""
    t1, t2, t3 = knots
    p = lambda u: np.maximum(u, 0) ** 3
    nl = (p(x - t1) - p(x - t2) * (t3 - t1) / (t3 - t2) + p(x - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2
    return np.column_stack([x, nl])


def dummies(s, levels):
    return np.column_stack([(s == l).to_numpy(float) for l in levels[1:]])


def base_blocks(a, tr, outcome, model):
    """Return list of (name, function(idx)->matrix) for the requested clinical model (M0/M1/M2)."""
    blocks = []
    def spline(col):
        k = np.percentile(a[col].to_numpy()[tr], [10, 50, 90])
        return lambda idx: rcs(a[col].to_numpy()[idx], k)
    lev = lambda col: list(a[col].iloc[tr].value_counts().index)  # reference = most frequent level
    blocks.append(('age', spline('Age')))
    blocks.append(('sex', lambda idx: a.Sex.to_numpy(float)[idx][:, None]))
    for col in ['Ethnicity4', 'Education', 'Smoking', 'Alcohol']:
        L = lev(col); blocks.append((col, (lambda c, L: (lambda idx: dummies(a[c].iloc[idx], L)))(col, L)))
    blocks.append(('townsend', lambda idx: a.townsend_index.to_numpy()[idx][:, None]))
    if model in ('M1', 'M2'):
        blocks.append(('stage', lambda idx: dummies(a.stage.iloc[idx], [0, 1, 2, 3])))
    if model == 'M2':
        for col in ['BMI', 'SBP', 'HbA1c', 'eGFR']:
            blocks.append((col, spline(col)))
        for col in ['HTN_meds', 'Diabetes', 'statin', 'LDLc', 'HDLc', 'logTG']:
            blocks.append((col, (lambda c: (lambda idx: a[c].to_numpy(float)[idx][:, None]))(col)))
        if outcome in APOE_OUT:
            blocks.append(('APOE4', lambda idx: dummies(a.APOE4.iloc[idx], [0.0, 1.0, 2.0])))
    return blocks


def design(a, tr, te, outcome, model, extra_tr=None, extra_te=None):
    """Standardised design matrices (training parameters). extra_*: additional predictors (n x k)."""
    blocks = base_blocks(a, tr, outcome, model)
    Xtr = np.column_stack([f(tr) for _, f in blocks]); Xte = np.column_stack([f(te) for _, f in blocks])
    if extra_tr is not None:
        Xtr = np.column_stack([Xtr, extra_tr]); Xte = np.column_stack([Xte, extra_te])
    m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1
    return (Xtr - m) / s, (Xte - m) / s


def _risk_sets(T):
    o = np.argsort(T, kind='mergesort'); Ts = T[o]
    # index of first element of each tie group in ascending order -> start of risk set
    _, first = np.unique(Ts, return_index=True)
    grp = np.searchsorted(Ts, Ts, side='left')
    return o, Ts, grp


def cox_fit(X, T, E, pen=0.0, maxit=60, tol=1e-9):
    """Cox PH, Breslow ties, Newton-Raphson with step-halving; ridge pen/2*||b||^2 on the sum scale."""
    n, p = X.shape
    o, Ts, grp = _risk_sets(T); Xs = X[o]; Es = E[o].astype(bool)
    b = np.zeros(p)
    def parts(b):
        eta = Xs @ b; c = eta.max(); w = np.exp(eta - c)
        S0r = np.cumsum(w[::-1])[::-1]; S1r = np.cumsum((w[:, None] * Xs)[::-1], axis=0)[::-1]
        S0 = S0r[grp]; S1 = S1r[grp]
        ll = (eta[Es] - c - np.log(S0[Es])).sum() - 0.5 * pen * b @ b
        return eta, w, S0, S1, ll
    eta, w, S0, S1, ll = parts(b)
    for it in range(maxit):
        m = S1[Es] / S0[Es][:, None]
        g = Xs[Es].sum(0) - m.sum(0) - pen * b
        inv = np.zeros(n); inv[Es] = 1 / S0[Es]
        cfw = np.cumsum(inv)  # sum over events with time <= T_j (ascending order)
        # ties: events at the same time belong to every tied subject's risk set
        last = np.searchsorted(Ts, Ts, side='right') - 1
        cj = cfw[last]
        H = (Xs * (w * cj)[:, None]).T @ Xs - m.T @ m + pen * np.eye(p)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            bn = b + t * step; r = parts(bn)
            if r[4] >= ll - 1e-12 or t < 1e-4: break
            t /= 2
        eta, w, S0, S1, lln = r; b = bn
        if abs(lln - ll) < tol * (1 + abs(ll)): ll = lln; break
        ll = lln
    # Breslow baseline cumulative hazard at HORIZON (on the eta scale of X @ b)
    eta = Xs @ b; c = eta.max(); w = np.exp(eta - c); S0 = np.cumsum(w[::-1])[::-1][grp]
    dH = np.where(Es, 1 / S0, 0.0) * np.exp(-c)
    H0h = dH[Ts <= HORIZON].sum()
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return dict(b=b, se=se, H0=H0h, ll=ll, iters=it + 1)


def km_risk(T, E, t=HORIZON):
    o = np.argsort(T); T = T[o]; E = E[o].astype(bool)
    ut, idx = np.unique(T, return_index=True)
    d = np.add.reduceat(E.astype(float), idx); nrisk = len(T) - idx
    keep = ut <= t
    return 1 - np.prod(1 - d[keep] / nrisk[keep])


def harrell_c(T, E, lp):
    return concordance_index(T, -lp, E)


def age_stratified_c(T, E, lp, age):
    bands = np.digitize(age, [50, 55, 60, 65])
    cs, ws = [], []
    for b in np.unique(bands):
        m = bands == b
        if E[m].sum() >= 1 and (~E[m].astype(bool)).sum() >= 1:
            cs.append(harrell_c(T[m], E[m], lp[m])); ws.append(E[m].sum())
    return float(np.average(cs, weights=ws)) if ws else float('nan')


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
