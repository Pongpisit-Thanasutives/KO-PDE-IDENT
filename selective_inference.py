import numpy as np
from sklearn.linear_model import Ridge, orthogonal_mp, lars_path
from sklearn.model_selection import KFold, cross_val_score
from func_timeout import func_timeout, FunctionTimedOut
from si4pipeline import (construct_pipelines, initialize_dataset, stepwise_feature_selection, PipelineManager)
from abess import LinearRegression as AbessLinearRegression
from solvel0 import MIOSR, miqp2

try:
    from parametric_si import parametric_sfs_si
except ImportError:
    print("parametric_si (https://github.com/takeuchi-lab/parametric-si) is not installed.")

def approximate_sigma(X, y, sigma=None):
    if sigma is not None:
        return sigma
        
    n, p = X.shape
    
    if n >= 2 * p:
        beta, rss, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # rss is sum of squared residuals if available; otherwise compute it
        if rss.size > 0:
            rss = rss[0]
        else:
            rss = np.linalg.norm(y - X @ beta) ** 2

        sigma = np.sqrt(rss / (n - p))
    else:
        sigma = np.std(y, ddof=1)

    return sigma

def forward_stepwise_regression(X, y, n_features=None, alpha=0, ic_type=None, cv=1, mio=False, X_norm=None):
    n, p = X.shape
    if n_features is not None:
        p = n_features

    if X_norm is not None:
        X = X / np.linalg.norm(X, ord=X_norm, axis=0)

    abess_nonzero = None
    if ic_type is not None or cv > 1:
        abess_lr = AbessLinearRegression(path_type='gs', fit_intercept=False, alpha=alpha, ic_type=ic_type, cv=cv, screening_size=0, important_search=0)
        abess_lr.fit(X, y)
        abess_nonzero = np.nonzero(abess_lr.coef_)[0]
        abess_mse = np.mean((y - abess_lr.predict(X)) ** 2)
        if mio:
            # miosr_coef = miqp2(X, y, len(abess_nonzero), alpha=alpha)
            miosr_coef = MIOSR(X, y, len(abess_nonzero), alpha=alpha)
            miosr_nonzero = np.nonzero(miosr_coef)[0]
            miosr_mse = np.mean((y - X @ miosr_coef) ** 2)
            # double mio
            tmp_nonzero = np.array(sorted(set(miosr_nonzero).union(set(abess_nonzero))))
            tmp_coef = MIOSR(X[:, tmp_nonzero], y, len(abess_nonzero), alpha=alpha)
            tmp_mse = np.mean((y - X[:, tmp_nonzero] @ tmp_coef) ** 2)
            if tmp_mse < miosr_mse:
                miosr_mse = tmp_mse
                miosr_nonzero = tmp_nonzero[np.nonzero(tmp_coef)[0]]
            if miosr_mse < abess_mse:
                abess_nonzero = miosr_nonzero

        X = X[:, abess_nonzero]
        p = len(abess_nonzero)

    selected, remaining = [], list(range(p))
    for k in range(1, p + 1):
        rss = np.empty(len(remaining))
        for j, idx in enumerate(remaining):
            cols = selected + [idx]
            X_sub = X[:, cols]
            if cv > 1:
                res = -len(y) * np.mean(cross_val_score(Ridge(alpha=alpha), X_sub, y, 
                                                        cv=KFold(n_splits=cv, shuffle=True, random_state=None), 
                                                        scoring="neg_mean_squared_error")) / cv
            else:
                beta, res, _, _ = np.linalg.lstsq(X_sub.T.dot(X_sub) + alpha * np.eye(len(cols)), X_sub.T.dot(y), rcond=None)
                res = res[0] if res.size else np.sum((y - X_sub @ beta) ** 2)

            rss[j] = res

        j_best = np.argmin(rss)
        selected.append(remaining.pop(j_best))

    if abess_nonzero is not None:
        selected = abess_nonzero[selected]

    return selected

def omp_path(X, y, n_features=None):
    if n_features is None:
        _, n_features = X.shape
    coeff_path = []
    for omp_sol in orthogonal_mp(X, y, n_nonzero_coefs=n_features, return_path=True).T:
        for coef in np.nonzero(omp_sol)[0]:
            if coef not in coeff_path:
                coeff_path.append(coef)
    return np.array(coeff_path, dtype=int)

def stepwise_regression(X, y, n_features=None, method='linear', kwargs={}):
    # Possible methods are linear, omp, lar, lasso
    if method == 'linear':
        coef_path =  forward_stepwise_regression(X, y, **kwargs, n_features=n_features)
    elif method == 'omp':
        coef_path = omp_path(X, y, **kwargs, n_features=n_features)
    else:
        if n_features is None:
            _, n_features = X.shape
        coef_path = lars_path(X, y, **kwargs, method=method)[1][:n_features]
    coef_path = np.array(list(map(int, coef_path)))
    return coef_path

def sfs_si(timeout, *args, **kwargs):
    try:
        return func_timeout(timeout, func=parametric_sfs_si, args=args, kwargs=kwargs)
    except FunctionTimedOut:
        return None

def stepwise_selective_inference(support_size) -> PipelineManager:
    return construct_pipelines(stepwise_feature_selection(*initialize_dataset(), support_size))

def subset_fdr(p_values):
    fdr = -np.mean(np.log(1-np.array(p_values)))
    return abs(fdr)

def forward_stop_rule(p_values, alpha=1.0):
    fdr = np.log(1-np.array(p_values))
    fdr = np.cumsum(fdr)
    for i in range(len(fdr)):
        fdr[i] = abs(-fdr[i]/(i+1))
    stop_at = -1
    stop_indices = np.where(fdr <= alpha)[0]
    if len(stop_indices) > 0:
        stop_at = max(stop_indices)
    return stop_at, fdr

def bonferroni_correction(alpha, n_tests):
    return alpha/n_tests

def sidak_correction(alpha, n_tests):
    return 1-((1-alpha)**n_tests)

