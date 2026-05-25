"""
Faithful implementation of ICOMP and RICOMP criteria from:

    Gueney, Y., Bozdogan, H., Arslan, O. (2021).
    Robust model selection in linear regression models using information complexity.
    Journal of Computational and Applied Mathematics 398, 113679.

Provides:
    * icomp_ifim(X, y)           -- non-robust ICOMP (Gaussian MLE)
    * ricomp_m(X, y)             -- M-based RICOMP (Eq. 14)
    * ricomp_s(X, y, ...)        -- S-based RICOMP (Eq. 29)
    * ricomp_mm(X, y, ...)       -- MM-based RICOMP (Eq. 36)
    * backward-compatible *_complexity wrappers returning only the C1 term

Two backends for S/MM:
    backend="python" (default) -- FAST-S (Salibian-Barrera & Yohai 2006) in pure
                                   NumPy, heavily tuned for speed
    backend="r"                -- robustbase::lmrob via rpy2 (reference; the
                                   de-facto gold-standard implementation)
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq


# =============================================================================
# Tukey biweight: rho, psi, psi' (paper Eq. 22-23)
# =============================================================================

def tukey_rho(u, c):
    u = np.asarray(u, dtype=float)
    out = np.full_like(u, c * c / 6.0)
    m = np.abs(u) <= c
    t2 = (u[m] / c) ** 2
    out[m] = (c * c / 6.0) * (1.0 - (1.0 - t2) ** 3)
    return out


def tukey_psi(u, c):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    m = np.abs(u) <= c
    t2 = (u[m] / c) ** 2
    out[m] = u[m] * (1.0 - t2) ** 2
    return out


def tukey_psi_deriv(u, c):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    m = np.abs(u) <= c
    t2 = (u[m] / c) ** 2
    out[m] = (1.0 - t2) * (1.0 - 5.0 * t2)
    return out


# Fast fused kernel for the M-scale inner loop: mean(tukey_rho(r/sigma, c))
# in a single pass with no intermediate-boolean-mask allocations.
def _mean_rho(r, sigma, c):
    cc_6 = c * c / 6.0
    inv_s = 1.0 / sigma
    inv_c = 1.0 / c
    u = r * inv_s
    absu = np.abs(u)
    # |u| clipped to c; t2 = (|u|/c)^2 in [0,1]
    np.clip(absu, 0.0, c, out=absu)
    t2 = (absu * inv_c) ** 2
    one_minus_t2 = 1.0 - t2
    # rho(u) = cc_6 * (1 - (1-t2)^3), which saturates to cc_6 when |u|>=c.
    return cc_6 * (1.0 - one_minus_t2 * one_minus_t2 * one_minus_t2).mean()


# =============================================================================
# M-scale
# =============================================================================

def m_scale(resid, c=1.547, bdp=0.5, tol=1e-10):
    resid = np.ascontiguousarray(np.asarray(resid, dtype=float).ravel())
    n = resid.size
    if n == 0:
        return 0.0
    abs_r = np.abs(resid)
    max_r = abs_r.max()
    if max_r < 1e-300:
        return 0.0
    b = bdp * (c * c / 6.0)

    def f(sigma):
        return _mean_rho(resid, sigma, c) - b

    mad = float(np.median(abs_r)) / 0.6745
    if mad < 1e-12:
        mad = max_r * 1e-3 + 1e-12

    lo, hi = mad * 1e-4, mad * 1e4
    f_lo, f_hi = f(lo), f(hi)
    for _ in range(40):
        if f_lo > 0:
            break
        lo *= 0.1
        f_lo = f(lo)
    for _ in range(40):
        if f_hi < 0:
            break
        hi *= 10.0
        f_hi = f(hi)
    if not (f_lo > 0 and f_hi < 0):
        return mad
    return brentq(f, lo, hi, xtol=tol, rtol=tol)


def _m_scale_fixed_point(resid, c, b, sigma_init, n_iter=2):
    """Quick fixed-point iteration used inside FAST-S screening.
    Derived from sigma_new = sigma * sqrt(mean_rho/b); converges to the M-scale."""
    sigma = sigma_init
    for _ in range(n_iter):
        mr = _mean_rho(resid, sigma, c)
        if mr <= 0:
            return sigma
        sigma = sigma * np.sqrt(mr / b)
    return sigma


# =============================================================================
# IRLS step (Tukey biweight)
# =============================================================================

def _irls_step(X, y, beta, sigma, c):
    u = (y - X @ beta) / sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        absu = np.abs(u)
        w = np.where(absu < c, (1.0 - (u / c) ** 2) ** 2, 0.0)
    sqrt_w = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta_new


# =============================================================================
# FAST-S (Salibian-Barrera & Yohai, 2006) -- optimized
# =============================================================================

def fast_s(X, y, c=1.547, bdp=0.5, N=None, k_steps=2, t_best=5,
           max_iter=200, tol=1e-7, seed=None):
    """FAST-S regression S-estimator.

    Defaults match robustbase::lmrob's KS2014: N=500 subsamples, k_steps=2,
    t_best=5. For n<=200 we use N=200 which is indistinguishable in quality
    and ~2-3x faster.
    """
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    n, p = X.shape

    if N is None:
        N = 200 if n <= 200 else 500

    b = bdp * (c * c / 6.0)

    # Phase 1: subsample screening (fast fixed-point scale, no brentq)
    cands = []
    for _ in range(N):
        idx = rng.choice(n, p, replace=False)
        try:
            beta = np.linalg.solve(X[idx], y[idx])
        except np.linalg.LinAlgError:
            continue

        resid = y - X @ beta
        abs_r = np.abs(resid)
        sigma = float(np.median(abs_r)) / 0.6745
        if sigma < 1e-12:
            sigma = abs_r.max() * 1e-3 + 1e-12
        sigma = _m_scale_fixed_point(resid, c, b, sigma, n_iter=2)

        for _ in range(k_steps):
            beta = _irls_step(X, y, beta, sigma, c)
            resid = y - X @ beta
            sigma = _m_scale_fixed_point(resid, c, b, sigma, n_iter=2)
            if sigma < 1e-300:
                break
        cands.append((sigma, beta))

    if not cands:
        raise RuntimeError("FAST-S: no feasible subsample found.")

    cands.sort(key=lambda t: t[0])
    cands = cands[:t_best]

    # Phase 2: refine top-t_best with exact brentq M-scale + early termination
    best_sigma = np.inf
    best_beta = None

    for _, beta in cands:
        resid = y - X @ beta
        sigma = m_scale(resid, c=c, bdp=bdp, tol=1e-8)
        if sigma > best_sigma * 1.05:
            continue

        prev_beta = beta
        for _ in range(max_iter):
            beta = _irls_step(X, y, beta, sigma, c)
            resid = y - X @ beta
            sigma = m_scale(resid, c=c, bdp=bdp, tol=1e-8)

            diff = np.max(np.abs(beta - prev_beta)) / max(1.0, np.max(np.abs(beta)))
            prev_beta = beta
            if diff < tol:
                break
            if sigma > best_sigma * 1.5:
                break

        if sigma < best_sigma:
            best_sigma = sigma
            best_beta = beta

    if best_beta is None:
        best_sigma, best_beta = cands[0][0], cands[0][1]

    return best_beta, best_sigma


# =============================================================================
# MM-estimator (Yohai, 1987)
# =============================================================================

def mm_estimator(X, y, c_s=1.547, c_mm=4.685, bdp=0.5, N=None, k_steps=2,
                 t_best=5, max_iter=200, tol=1e-7, seed=None):
    beta_s, sigma_s = fast_s(X, y, c=c_s, bdp=bdp, N=N, k_steps=k_steps,
                             t_best=t_best, max_iter=max_iter, tol=tol, seed=seed)
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())

    beta = beta_s.copy()
    for _ in range(max_iter):
        beta_new = _irls_step(X, y, beta, sigma_s, c_mm)
        diff = np.max(np.abs(beta_new - beta)) / max(1.0, np.max(np.abs(beta)))
        beta = beta_new
        if diff < tol:
            break
    return beta, sigma_s


# =============================================================================
# Optional R backend via rpy2 + robustbase::lmrob
# =============================================================================

_R_BACKEND_ERROR = None
try:
    import os as _os
    # Allow users to point us at a custom R library if needed
    _R_LIBS_EXTRA = _os.environ.get("ICOMP_R_LIBS", "")
    from rpy2.robjects import numpy2ri as _numpy2ri, r as _r  # noqa: E402
    from rpy2.robjects.packages import importr as _importr   # noqa: E402
    _numpy2ri.activate()
    if _R_LIBS_EXTRA:
        try:
            _paths = ", ".join(f'"{p}"' for p in _R_LIBS_EXTRA.split(_os.pathsep) if p)
            _r(f'.libPaths(c({_paths}, .libPaths()))')
        except Exception:
            pass
    _robustbase = _importr("robustbase")
    HAVE_R = True
except Exception as _e:
    HAVE_R = False
    _R_BACKEND_ERROR = repr(_e)
    _robustbase = None


def _require_r():
    if not HAVE_R:
        raise RuntimeError(
            "R backend not available. Install R + robustbase + rpy2. "
            f"Import error: {_R_BACKEND_ERROR}"
        )


def fast_s_r(X, y, c=1.547, bdp=0.5, seed=None):
    """S-stage of robustbase::lmrob. Reference implementation."""
    _require_r()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if seed is not None:
        _r(f"set.seed({int(seed)})")

    ctrl = _robustbase.lmrob_control(
        setting="KS2014",
        **{"psi": "bisquare", "tuning.chi": c, "bb": bdp}
    )
    res = _robustbase.lmrob_fit(X, y, control=ctrl)
    # lmrob.fit() stores the S-stage result under `$init` (NOT `$init.S`,
    # which is only populated by the formula interface lmrob()).
    init_S = res.rx2("init")
    beta = np.asarray(init_S.rx2("coefficients")).ravel()
    sigma = float(init_S.rx2("scale")[0])
    return beta, sigma


def mm_estimator_r(X, y, c_s=1.547, c_mm=4.685, bdp=0.5, seed=None):
    """MM-estimate from robustbase::lmrob. Reference implementation."""
    _require_r()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if seed is not None:
        _r(f"set.seed({int(seed)})")

    ctrl = _robustbase.lmrob_control(
        setting="KS2014",
        **{"psi": "bisquare",
           "tuning.chi": c_s, "bb": bdp,
           "tuning.psi": c_mm}
    )
    res = _robustbase.lmrob_fit(X, y, control=ctrl)
    beta = np.asarray(res.rx2("coefficients")).ravel()
    sigma = float(res.rx2("scale")[0])
    return beta, sigma


# =============================================================================
# Informational complexity primitives and ICOMP criteria (Bozdogan & Haughton, 1998)
# =============================================================================

def informational_complexity(Sigma, kind="C1", eps=1e-12):
    """General informational complexity of a positive-definite matrix.

    This is the primitive used to build every ICOMP-family criterion.
    Given a covariance (or inverse-Fisher-information) matrix Σ of
    dimension k, it measures how much Σ departs from a scalar multiple
    of the identity — i.e., how interdependent (non-orthogonal) the
    parameters are.

    Parameters
    ----------
    Sigma : array-like of shape (k, k)
        Positive-definite symmetric matrix.
    kind : {"C0", "C1"}, default "C1"
        Which complexity to compute.

        * "C0" — van Emden (1971) complexity, Eq. 2.2 of Bozdogan &
          Haughton (1998). NOT invariant under orthonormal
          transformations; generally not recommended for model selection.
          C0(Σ) = (1/2) Σⱼ log(σ²ⱼⱼ) − (1/2) log|Σ|

        * "C1" — Bozdogan's maximal informational complexity, Eq. 2.3.
          Invariant under scalar multiplication and orthonormal
          transforms; monotonically increasing in k. Zero iff Σ ∝ I.
          C1(Σ) = (k/2) log(tr(Σ)/k) − (1/2) log|Σ|
    eps : float
        Floor on diagonal entries for C0, preventing log of non-positive
        values on ill-conditioned inputs.

    Returns
    -------
    C : float
        The complexity. Nonnegative for C1 on a positive-definite Σ.
        Returns +inf if Σ is not positive-definite.

    References
    ----------
    Bozdogan, H. & Haughton, D. M. A. (1998). Informational complexity
    criteria for regression models. Comp. Stat. & Data Anal. 28, 51-76.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square; got shape {Sigma.shape}")
    k = Sigma.shape[0]
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    if kind == "C0":
        diag = np.maximum(np.diag(Sigma), eps)
        return 0.5 * (float(np.sum(np.log(diag))) - logdet)
    if kind == "C1":
        tr = float(np.trace(Sigma))
        if tr <= 0:
            return np.inf
        return 0.5 * (k * np.log(tr / k) - logdet)
    raise ValueError(f"Unknown kind={kind!r}; expected 'C0' or 'C1'")


def _complexity_C1(Sigma, k=None):
    """Internal thin wrapper; k kept for backward compat, not used."""
    return informational_complexity(Sigma, kind="C1")


def _block_cov(cov_beta, var_sigma):
    """Assemble block-diagonal [cov_beta, var_sigma]."""
    p = cov_beta.shape[0]
    return np.block([[cov_beta, np.zeros((p, 1))],
                     [np.zeros((1, p)), np.array([[var_sigma]])]])


def _ols_fit(X, y):
    """OLS fit returning (beta, resid, sigma2_MLE, XtX_inv)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = y.size
    sigma2 = float(resid @ resid) / n  # MLE (paper Eq. 4); NOT the unbiased RSS/(n-p)
    XtX_inv = np.linalg.inv(X.T @ X)
    return beta, resid, sigma2, XtX_inv


def _neg2_gaussian_loglik(n, sigma2):
    """-2 log L at the Gaussian MLE: n log(2π) + n log σ² + n."""
    return n * np.log(2.0 * np.pi) + n * np.log(sigma2) + n


# ---- ICOMP flavors for linear regression ---------------------------------
#
# Three criteria from Bozdogan & Haughton (1998), each parameterised by an
# optional sequence a_n controlling the penalty strength:
#     a_n = 1       -> AIC-like asymptotic behavior (Proposition 2)
#     a_n = log(n)  -> BIC-like consistency
#
#     icomp_c1    -- Eq. 3.3  -- complexity of (X'X)⁻¹ alone
#     icomp_ifim  -- Eq. 3.7  -- complexity of asymptotic IFIM
#     icomp_cov   -- Eq. 3.8  -- complexity of finite-sample Q = cov(β̂, σ̂²)
#
# All three reduce to -2 log L + 2 a_n · C₁(·) where the matrix inside C₁
# differs. The scale-invariance of C₁ means σ² factors out of the top block,
# giving the compact forms in the paper.

def icomp_c1(X, y, a_n=1.0):
    """ICOMP_{a_n} criterion, Eq. 3.3 of Bozdogan & Haughton (1998).

    Uses the C1 complexity of (X'X)⁻¹ alone. This is the Approach-1
    formulation in which the residual complexity on the (n-q)-dimensional
    projected subspace is treated as zero.

    Lower is better.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, _ = X.shape
    _, _, sigma2, XtX_inv = _ols_fit(X, y)
    C1 = informational_complexity(XtX_inv, kind="C1")
    return _neg2_gaussian_loglik(n, sigma2) + 2.0 * a_n * C1


def icomp_ifim(X, y, a_n=1.0):
    """ICOMP_IFIM_{a_n} criterion, Eq. 3.7 of Bozdogan & Haughton (1998).

    Uses the C1 complexity of the asymptotic inverse Fisher information
    matrix F⁻¹ = diag(σ²(X'X)⁻¹, 2σ⁴/n). Because C1 is scale-invariant,
    this equals C1(diag((X'X)⁻¹, 2σ²/n)).

    Lower is better.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, _ = X.shape
    _, _, sigma2, XtX_inv = _ols_fit(X, y)
    cov_beta = sigma2 * XtX_inv
    var_sigma = 2.0 * sigma2 ** 2 / n
    F_inv = _block_cov(cov_beta, var_sigma)
    C1 = informational_complexity(F_inv, kind="C1")
    return _neg2_gaussian_loglik(n, sigma2) + 2.0 * a_n * C1


def icomp_cov(X, y, a_n=1.0):
    """ICOMP_COV_{a_n} criterion, Eq. 3.8 of Bozdogan & Haughton (1998).

    Uses the finite-sample covariance matrix
        Q = diag(σ²(X'X)⁻¹, 2σ⁴ (n-q)/n²)
    with q = p (the number of regressor columns in X). Asymptotically
    equivalent to ICOMP_IFIM but can differ noticeably for small n.

    Lower is better.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    q = p
    _, _, sigma2, XtX_inv = _ols_fit(X, y)
    cov_beta = sigma2 * XtX_inv
    var_sigma = 2.0 * sigma2 ** 2 * (n - q) / (n * n)
    Q = _block_cov(cov_beta, var_sigma)
    C1 = informational_complexity(Q, kind="C1")
    return _neg2_gaussian_loglik(n, sigma2) + 2.0 * a_n * C1


def icomp_all(X, y, a_n=1.0):
    """Compute all four ICOMP-family criteria at once.

    Returns
    -------
    dict with keys {'C0', 'C1', 'IFIM', 'COV'}:
        'C0'   : -2 log L + 2 a_n · C0((X'X)⁻¹)   (non-invariant; for reference)
        'C1'   : -2 log L + 2 a_n · C1((X'X)⁻¹)   Eq. 3.3
        'IFIM' : -2 log L + 2 a_n · C1(F⁻¹)       Eq. 3.7
        'COV'  : -2 log L + 2 a_n · C1(Q)         Eq. 3.8

    'C0' is included for completeness only; prefer C1-based variants.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    q = p
    _, _, sigma2, XtX_inv = _ols_fit(X, y)
    neg2ll = _neg2_gaussian_loglik(n, sigma2)

    cov_beta = sigma2 * XtX_inv
    F_inv = _block_cov(cov_beta, 2.0 * sigma2 ** 2 / n)
    Q     = _block_cov(cov_beta, 2.0 * sigma2 ** 2 * (n - q) / (n * n))

    return {
        "C0":   neg2ll + 2.0 * a_n * informational_complexity(XtX_inv, kind="C0"),
        "C1":   neg2ll + 2.0 * a_n * informational_complexity(XtX_inv, kind="C1"),
        "IFIM": neg2ll + 2.0 * a_n * informational_complexity(F_inv,   kind="C1"),
        "COV":  neg2ll + 2.0 * a_n * informational_complexity(Q,       kind="C1"),
    }


# ---- Complexity-only wrappers (penalty term only; no -2 log L) ----------

def icomp_c1_complexity(X, y=None):
    """C1((X'X)⁻¹) penalty alone (no likelihood term).

    `y` is accepted for signature symmetry with other *_complexity
    functions but is not used.
    """
    X = np.asarray(X, dtype=float)
    return informational_complexity(np.linalg.inv(X.T @ X), kind="C1")


def icomp_ifim_complexity(X_pre, y_pre, use_sm=True):
    """C1(F⁻¹) penalty alone (no likelihood). Backward-compatible signature."""
    X = np.asarray(X_pre, dtype=float)
    y = np.asarray(y_pre, dtype=float).ravel()
    n, _ = X.shape
    if use_sm:
        m_res = sm.OLS(y, X).fit()
        sigma2 = float(np.sum(m_res.resid ** 2)) / n
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = y - X @ beta
        sigma2 = float(err @ err) / n
    cov = sigma2 * np.linalg.inv(X.T @ X)
    var_sigma = 2.0 * sigma2 ** 2 / n
    return informational_complexity(_block_cov(cov, var_sigma), kind="C1")


def icomp_cov_complexity(X, y):
    """C1(Q) penalty alone (no likelihood term), Eq. 3.8's finite-sample Q."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    _, _, sigma2, XtX_inv = _ols_fit(X, y)
    cov_beta = sigma2 * XtX_inv
    var_sigma = 2.0 * sigma2 ** 2 * (n - p) / (n * n)
    return informational_complexity(_block_cov(cov_beta, var_sigma), kind="C1")


# --- RICOMP_M ---------------------------------------------------------------

def _m_fit(X, y, c=4.685):
    norm = sm.robust.norms.TukeyBiweight(c=c)
    res = sm.RLM(y, X, M=norm).fit(scale_est="mad")
    return res.params, float(res.scale), np.asarray(res.resid).ravel()


def _ricomp_m_core(X, y, c=4.685):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    k = p + 1
    beta_m, sigma_m, resid = _m_fit(X, y, c=c)
    u = resid / sigma_m
    mean_psi2 = np.mean(tukey_psi(u, c) ** 2)
    mean_psi_prime = np.mean(tukey_psi_deriv(u, c))
    if mean_psi_prime ** 2 < 1e-300:
        return beta_m, sigma_m, resid, None, k
    scaling = (n / (n - p)) * mean_psi2 / mean_psi_prime ** 2
    cov_beta = sigma_m ** 2 * scaling * np.linalg.inv(X.T @ X)
    var_sigma = 2.0 * sigma_m ** 4 / n
    Sigma = _block_cov(cov_beta, var_sigma)
    return beta_m, sigma_m, resid, Sigma, k


def ricomp_m(X, y, c=4.685):
    beta_m, sigma_m, resid, Sigma, k = _ricomp_m_core(X, y, c=c)
    if Sigma is None:
        return np.inf
    C1 = _complexity_C1(Sigma, k)
    loss = 2.0 * np.sum(tukey_rho(resid / sigma_m, c))
    return loss + 2.0 * C1


def ricomp_m_complexity(X, y, c=4.685):
    _, _, _, Sigma, k = _ricomp_m_core(X, y, c=c)
    if Sigma is None:
        return np.inf
    return _complexity_C1(Sigma, k)


# --- RICOMP_S ---------------------------------------------------------------

def _fit_s(X, y, c=1.547, bdp=0.5, backend="python", seed=None, **kw):
    if backend == "r":
        return fast_s_r(X, y, c=c, bdp=bdp, seed=seed)
    return fast_s(X, y, c=c, bdp=bdp, seed=seed, **kw)


def _ricomp_s_core(X, y, c=1.547, bdp=0.5, backend="python", seed=None, **kw):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    k = p + 1

    beta_s, sigma_s = _fit_s(X, y, c=c, bdp=bdp, backend=backend, seed=seed, **kw)
    resid = y - X @ beta_s
    u = resid / sigma_s

    mean_psi2 = np.mean(tukey_psi(u, c) ** 2)
    mean_psi_prime = np.mean(tukey_psi_deriv(u, c))
    if mean_psi_prime ** 2 < 1e-300:
        return beta_s, sigma_s, resid, None, k

    scaling_beta = (n / (n - p)) * mean_psi2 / mean_psi_prime ** 2
    cov_beta = sigma_s ** 2 * scaling_beta * np.linalg.inv(X.T @ X)
    rho_vals = tukey_rho(u, c)
    num = np.mean((rho_vals - np.mean(rho_vals)) ** 2)
    den = np.mean(u * tukey_psi(u, c)) ** 2
    if den < 1e-300:
        return beta_s, sigma_s, resid, None, k
    var_sigma = num / (n * den)
    Sigma = _block_cov(cov_beta, var_sigma)
    return beta_s, sigma_s, resid, Sigma, k


def ricomp_s(X, y, c=1.547, bdp=0.5, backend="python", seed=None, **kw):
    _, sigma_s, _, Sigma, k = _ricomp_s_core(X, y, c=c, bdp=bdp,
                                             backend=backend, seed=seed, **kw)
    if Sigma is None:
        return np.inf
    C1 = _complexity_C1(Sigma, k)
    n = np.asarray(X).shape[0]
    return 2.0 * n * np.log(sigma_s) + 2.0 * C1


def ricomp_s_complexity(X, y, c=1.547, bdp=0.5, backend="python", seed=None, **kw):
    _, _, _, Sigma, k = _ricomp_s_core(X, y, c=c, bdp=bdp,
                                       backend=backend, seed=seed, **kw)
    if Sigma is None:
        return np.inf
    return _complexity_C1(Sigma, k)


# --- RICOMP_MM --------------------------------------------------------------

def _fit_mm(X, y, c_s=1.547, c_mm=4.685, bdp=0.5, backend="python", seed=None, **kw):
    if backend == "r":
        return mm_estimator_r(X, y, c_s=c_s, c_mm=c_mm, bdp=bdp, seed=seed)
    return mm_estimator(X, y, c_s=c_s, c_mm=c_mm, bdp=bdp, seed=seed, **kw)


def _ricomp_mm_core(X, y, c_s=1.547, c_mm=4.685, bdp=0.5,
                    backend="python", seed=None, **kw):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    k = p + 1

    beta_mm, sigma_mm = _fit_mm(X, y, c_s=c_s, c_mm=c_mm, bdp=bdp,
                                backend=backend, seed=seed, **kw)
    resid = y - X @ beta_mm
    u = resid / sigma_mm

    mean_psi2 = np.mean(tukey_psi(u, c_mm) ** 2)
    mean_psi_prime = np.mean(tukey_psi_deriv(u, c_mm))
    if mean_psi_prime ** 2 < 1e-300:
        return beta_mm, sigma_mm, resid, None, k

    scaling = (n / (n - p)) * mean_psi2 / mean_psi_prime ** 2
    cov_beta = sigma_mm ** 2 * scaling * np.linalg.inv(X.T @ X)
    var_sigma = 2.0 * sigma_mm ** 4 / n
    Sigma = _block_cov(cov_beta, var_sigma)
    return beta_mm, sigma_mm, resid, Sigma, k


def ricomp_mm(X, y, c_s=1.547, c_mm=4.685, bdp=0.5,
              backend="python", seed=None, **kw):
    _, sigma_mm, resid, Sigma, k = _ricomp_mm_core(
        X, y, c_s=c_s, c_mm=c_mm, bdp=bdp, backend=backend, seed=seed, **kw
    )
    if Sigma is None:
        return np.inf
    C1 = _complexity_C1(Sigma, k)
    loss = 2.0 * np.sum(tukey_rho(resid / sigma_mm, c_mm))
    return loss + 2.0 * C1


def ricomp_mm_complexity(X, y, c_s=1.547, c_mm=4.685, bdp=0.5,
                         backend="python", seed=None, **kw):
    _, _, _, Sigma, k = _ricomp_mm_core(
        X, y, c_s=c_s, c_mm=c_mm, bdp=bdp, backend=backend, seed=seed, **kw
    )
    if Sigma is None:
        return np.inf
    return _complexity_C1(Sigma, k)
