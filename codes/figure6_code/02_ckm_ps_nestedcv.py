"""Figure 6 step 2: CKM proteomic score (CKM-PS) by repeated nested cross-validation.

Label  : clinical CKM stage (0-3, ordinal coding) -- no outcome information is used.
Design : outer 5-fold x 10 repeats, stratified by stage. Within each outer training fold:
         (i) proteins residualised on age, age^2 and sex, then standardised (training parameters);
         (ii) elastic net (l1_ratio 0.5); alpha chosen by inner 5-fold CV (min MSE) on a 12-point grid;
         (iii) inner validation predictions at the chosen alpha = cross-fitted CKM-PS for training
               participants (used later to fit Cox models without in-sample optimism);
         (iv) refit on the whole outer training fold -> CKM-PS for the outer test fold.
         Secondary: Figure-4-anchored score (stage contrasts re-estimated in the training fold;
         proteins FDR<0.05 in all three contrasts with concordant direction; stage-3 beta weights).
         Repeat 0 also fits a non-deconfounded score (sensitivity: age leakage).
"""
import numpy as np, pandas as pd, warnings, time, sys, json
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import enet_path
from scipy import stats
warnings.filterwarnings('ignore')
R = Path('/home/ubuntu/Figure6_Prediction'); OUTD = R / 'private/score_folds'; OUTD.mkdir(parents=True, exist_ok=True)
NREP, NOUT, NIN, L1, NALPHA = 10, 5, 5, 0.5, 12


def enet_fit(Z, y, alphas):
    """Elastic-net path on standardised Z with intercept handled by centring. Returns coefs (p x k), x mean, y mean."""
    xm = Z.mean(0); ym = float(y.mean()); Zc = Z - xm; yc = (y - ym).astype(Z.dtype)
    G = Zc.T @ Zc; Xy = Zc.T @ yc
    _, coefs, _ = enet_path(Zc, yc, l1_ratio=L1, alphas=alphas, precompute=G, Xy=Xy, check_input=False,
                            max_iter=3000, tol=1e-3)
    return coefs, xm, ym


def score_model(Ztr, ytr, Zte, seed):
    n = len(ytr)
    Zc = Ztr - Ztr.mean(0)
    amax = float(np.abs(Zc.T @ (ytr - ytr.mean()).astype(Ztr.dtype)).max()) / (n * L1)
    alphas = np.geomspace(amax * 0.5, amax * 0.005, NALPHA).astype(Ztr.dtype)
    inner = StratifiedKFold(NIN, shuffle=True, random_state=seed)
    oof = np.zeros((n, NALPHA), np.float64)
    for itr, iva in inner.split(Ztr, np.round(ytr).astype(int)):
        coefs, xm, ym = enet_fit(Ztr[itr], ytr[itr], alphas)
        oof[iva] = (Ztr[iva] - xm) @ coefs + ym
    mse = ((oof - ytr[:, None]) ** 2).mean(0)
    k = int(np.argmin(mse))
    coefs, xm, ym = enet_fit(Ztr, ytr, alphas[:k + 1])
    b = coefs[:, k]
    return dict(train_oof=oof[:, k], test=(Zte - xm) @ b + ym, coef=b.astype(np.float32), alpha=float(alphas[k]),
                alpha_index=k, nnz=int((b != 0).sum()), inner_mse=mse.tolist(), r2_inner=float(1 - mse[k] / ytr.var()))


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def fig4_weights(Xs, D):
    """Protein ~ stage dummies + Figure-4 covariates (OLS, all proteins at once). Returns stage betas and q."""
    B, *_ = np.linalg.lstsq(D, Xs, rcond=None)
    res = Xs - D @ B; dfree = D.shape[0] - D.shape[1]
    s2 = (res.astype(np.float64) ** 2).sum(0) / dfree
    XtXi = np.linalg.inv(D.T @ D)
    out = {}
    for j, s in enumerate([1, 2, 3], start=1):  # columns 1..3 = stage dummies
        se = np.sqrt(s2 * XtXi[j, j]); t = B[j] / se
        out[s] = (B[j], bh(2 * stats.t.sf(np.abs(t), dfree)))
    return out


def run_fold(rep, fold, tr, te):
    t0 = time.time()
    X = np.load(R / 'private/proteins.npy', mmap_mode='r')
    a = pd.read_parquet(R / 'private/analysis.parquet')
    y = a.stage.to_numpy(np.float64)
    C = np.column_stack([np.ones(len(a)), a.Age, a.Age ** 2, a.Sex]).astype(np.float64)
    C[:, 1:3] = (C[:, 1:3] - C[tr, 1:3].mean(0)) / C[tr, 1:3].std(0)
    Xtr = np.asarray(X[tr], np.float32); Xte = np.asarray(X[te], np.float32)
    # (i) age/sex residualisation with training-fold coefficients
    Bc, *_ = np.linalg.lstsq(C[tr], Xtr.astype(np.float64), rcond=None)
    Rtr = (Xtr - C[tr] @ Bc).astype(np.float32); Rte = (Xte - C[te] @ Bc).astype(np.float32)
    mu, sd = Rtr.mean(0), Rtr.std(0)
    Ztr = (Rtr - mu) / sd; Zte = (Rte - mu) / sd
    res = score_model(Ztr, y[tr], Zte, seed=7000 + 97 * rep + fold)
    # Figure-4-anchored score
    dm = pd.get_dummies(a[['Smoking', 'Alcohol', 'Ethnicity4']].astype(str), drop_first=True).to_numpy(float)
    S = np.column_stack([(a.stage == s).to_numpy(float) for s in (1, 2, 3)])
    D = np.column_stack([np.ones(len(a)), S, a.Age, a.Sex, dm])[tr]
    mx, sx = Xtr.mean(0), Xtr.std(0)
    f4 = fig4_weights(((Xtr - mx) / sx).astype(np.float64), D)
    sig = (f4[1][1] < .05) & (f4[2][1] < .05) & (f4[3][1] < .05)
    same = (np.sign(f4[1][0]) == np.sign(f4[2][0])) & (np.sign(f4[2][0]) == np.sign(f4[3][0]))
    sel = sig & same; w = np.where(sel, f4[3][0], 0.0); w = w / np.abs(w).sum()
    out = dict(rep=rep, fold=fold, tr=tr.astype(np.int32), te=te.astype(np.int32),
               ps_train=res['train_oof'].astype(np.float32), ps_test=res['test'].astype(np.float32),
               coef=res['coef'], alpha=res['alpha'], alpha_index=res['alpha_index'], nnz=res['nnz'],
               r2_inner=res['r2_inner'], inner_mse=np.array(res['inner_mse']),
               f4_train=(Ztr @ w).astype(np.float32), f4_test=(Zte @ w).astype(np.float32),
               f4_sel=sel, f4_n=int(sel.sum()))
    if rep == 0:  # sensitivity: no age/sex de-confounding
        Ztr0 = ((Xtr - mx) / sx).astype(np.float32); Zte0 = ((Xte - mx) / sx).astype(np.float32)
        r0 = score_model(Ztr0, y[tr], Zte0, seed=9100 + fold)
        out.update(raw_train=r0['train_oof'].astype(np.float32), raw_test=r0['test'].astype(np.float32),
                   raw_nnz=r0['nnz'], raw_alpha=r0['alpha'])
    np.savez_compressed(OUTD / f'rep{rep}_fold{fold}.npz', **out)
    return dict(rep=rep, fold=fold, n_train=len(tr), n_test=len(te), nnz=res['nnz'], alpha_index=res['alpha_index'],
                r2_inner=round(res['r2_inner'], 4), f4_n=int(sel.sum()), sec=round(time.time() - t0, 1))


if __name__ == '__main__':
    a = pd.read_parquet(R / 'private/analysis.parquet'); y = a.stage.to_numpy()
    jobs = []
    for rep in range(NREP):
        skf = StratifiedKFold(NOUT, shuffle=True, random_state=2026 + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if not (OUTD / f'rep{rep}_fold{fold}.npz').exists():
                jobs.append((rep, fold, tr, te))
    print('folds to run', len(jobs), flush=True)
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for r in Parallel(n_jobs=nj, verbose=0, return_as='generator')(delayed(run_fold)(*j) for j in jobs):
        print(json.dumps(r), flush=True)
    print('DONE', flush=True)
