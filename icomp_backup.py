# Implemented by Pongpisit Thanasutives
# Ref Robust model selection in linear regression models using information complexity (2021)
import numpy as np
import statsmodels.api as sm
from statsmodels.robust import scale
from statsmodels.robust.norms import TukeyBiweight
from astropy.stats import biweight_scale


class FixedBiweightScale(scale.HuberScale):
    def __init__(self, fixed_scale):
        super().__init__()
        self.fixed_scale = fixed_scale

    def __call__(self, df_resid, nobs, resid):
        return self.fixed_scale


class BiweightScale(scale.HuberScale):
    def __init__(self, c=1.55, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    def __call__(self, df_resid, nobs, resid):
        return biweight_scale(resid, c=self.c)


### Old / Based on https://github.com/Pongpisit-Thanasutives/ICOMP ###
# def icomp_complexities(X_pre, y_pre, beta=None, eps=1e-12, verbose=False):
#     N = len(y_pre)

#     # -------------------------------------------------
#     # Fit model or use provided beta
#     # -------------------------------------------------
#     if beta is None:
#         model = sm.OLS(y_pre, X_pre)
#         m_res = model.fit()

#         q = model.rank
#         resid = m_res.resid
#         rss = float(np.sum(resid**2))

#         # Inverse Fisher / covariance-like matrix
#         S_inv = m_res.cov_params(scale=1)
#     else:
#         beta = beta.reshape(-1, 1)
#         q = np.linalg.matrix_rank(X_pre)

#         resid = y_pre.reshape(-1, 1) - X_pre @ beta
#         rss = float(np.sum(resid**2))

#         XtX = X_pre.T @ X_pre
#         S_inv = np.linalg.inv(XtX)

#     # -------------------------------------------------
#     # Eigenvalues of S_inv (for log + trace stability)
#     # -------------------------------------------------
#     eigvals = np.linalg.eigvalsh(S_inv)

#     if np.any(eigvals <= 0):
#         if verbose:
#             print("Warning: S_inv is not positive definite; clipping eigenvalues.")
#         eigvals = np.clip(eigvals, eps, None)

#     trace_Sinv = float(np.sum(eigvals))

#     # -------------------------------------------------
#     # log(det(S_inv)) via slogdet
#     # -------------------------------------------------
#     sign, logdet_Sinv = np.linalg.slogdet(S_inv)

#     if sign <= 0:
#         if verbose:
#             print("inf covariance complexity (non-positive determinant)")
#         return np.full(4, np.inf)

#     # -------------------------------------------------
#     # Complexity measures
#     # -------------------------------------------------
#     C0 = np.sum(np.log(np.maximum(np.diag(S_inv), eps))) - logdet_Sinv

#     C1 = q * np.log(trace_Sinv / q) - logdet_Sinv

#     C_IFIM = (
#         (q + 1) * np.log((trace_Sinv + 2 * rss / N) / (q + 1))
#         - logdet_Sinv
#         - np.log(2 * rss / N)
#     )

#     C_COV = (
#         (q + 1) * np.log((trace_Sinv + 2 * rss * (N - q) / (N**2)) / (q + 1))
#         - logdet_Sinv
#         - np.log(2 * rss * (N - q) / (N**2))
#     )

#     return np.array([C0, C1, C_IFIM, C_COV]) / 2


def icomp_ifim_complexity(X_pre, y_pre, use_sm=True):
    y_pre = y_pre.ravel()
    n, p = X_pre.shape
    # assert n == len(y_pre)
    k = p + 1

    if use_sm:
        m_res = sm.OLS(y_pre, X_pre).fit()
        var = m_res.scale
        cov = m_res.cov_params()
    else:
        err = y_pre - X_pre @ np.linalg.lstsq(X_pre, y_pre, rcond=None)[0]
        var = float(np.dot(err, err)) / n
        cov = var * np.linalg.inv(X_pre.T @ X_pre)

    F_inv = np.block(
        [
            [cov, np.zeros((p, 1))],
            [np.zeros((1, p)), 2 * var**2 / n],
        ]
    )
    _, log_det = np.linalg.slogdet(F_inv)
    # assert _ > 0

    com = k * np.log(np.trace(F_inv) / k) - log_det
    return com / 2


def ricomp_m_complexity(X, y):
    n, p = X.shape
    k = p + 1

    # Fit Robust Model
    m_norm = sm.robust.norms.TukeyBiweight(c=4.685)  # [cite: 362]
    m_res = sm.RLM(y, X, M=m_norm).fit()

    # sigma_m IS the standard deviation (scale) in statsmodels RLM
    sigma_m = m_res.scale

    # Calculate robust scaling factor
    resids_scaled = m_res.resid / sigma_m
    numerator = np.mean(m_norm.psi(resids_scaled) ** 2)
    denominator = np.mean(m_norm.psi_deriv(resids_scaled)) ** 2

    # Cov(beta) block uses sigma_m squared
    scaling_factor = (n / (n - p)) * (numerator / denominator)
    cov_beta_m = (sigma_m**2) * scaling_factor * np.linalg.inv(X.T @ X)  #

    # Var(sigma) block uses sigma_m to the fourth power
    var_sigma_m = (2 * (sigma_m**4)) / n

    # Assemble Sigma_M
    Sigma_M = np.block(
        [[cov_beta_m, np.zeros((p, 1))], [np.zeros((1, p)), var_sigma_m]]
    )  #

    # Calculate C1 Complexity
    tr_S = np.trace(Sigma_M)
    _, log_det_S = np.linalg.slogdet(Sigma_M)

    c1 = k * np.log(tr_S / k) - log_det_S  # [cite: 101, 176]
    return c1 / 2


def ricomp_s_complexity(X, y):
    """
    Implements RICOMP based on S-estimation (Eq 29, 30).
    S-estimators are robust against outliers in both y and X directions.
    """
    y = y.ravel()
    n, p = X.shape
    k = p + 1  # Total free parameters (p coefficients + 1 scale)

    # 1. Fit S-estimator (Standard in statsmodels via RLM with specific setup)
    # The paper uses Tukey's biweight function with c=1.547 for 50% BDP
    s_norm = sm.robust.norms.TukeyBiweight(c=1.547)

    # We use RLM but we must extract the S-scale specifically
    # In practice, S-estimators are often used to initialize MM-estimators.
    model = sm.RLM(y, X, M=s_norm).fit()

    beta_s = model.params
    sigma_s = model.scale  # This is the S-estimate of scale

    # 2. Residuals and Influence Function components
    resids_scaled = model.resid / sigma_s
    psi_vals = s_norm.psi(resids_scaled)
    psi_prime_vals = s_norm.psi_deriv(resids_scaled)
    rho_vals = s_norm.rho(resids_scaled)

    # 3. Covariance of Beta_S
    # Factor: ave{psi^2} / [ave{psi'}]^2
    scaling_beta = (n / (n - p)) * (
        np.mean(psi_vals**2) / (np.mean(psi_prime_vals) ** 2)
    )
    cov_beta_s = (sigma_s**2) * scaling_beta * np.linalg.inv(X.T @ X)

    # 4. Variance of Sigma_S
    # This term is specific to the S-estimator's asymptotic distribution
    # Numerator: ave{(rho - ave{rho})^2}
    # Denominator: {ave{u * psi(u)}}^2 where u is the scaled residual
    numerator_sigma = np.mean((rho_vals - np.mean(rho_vals)) ** 2)
    denominator_sigma = np.mean(resids_scaled * psi_vals) ** 2

    var_sigma_s = (1 / n) * (numerator_sigma / denominator_sigma)

    # 5. Construct the Robust IFIM Matrix Sigma_S (Eq 28)
    #
    Sigma_S = np.block(
        [[cov_beta_s, np.zeros((p, 1))], [np.zeros((1, p)), var_sigma_s]]
    )

    # 6. Calculate C1 Complexity (Eq 30)
    # C1 = (k/2) * log(tr(Sigma_S)/k) - (1/2) * log|Sigma_S|
    tr_S = np.trace(Sigma_S)
    _, log_det_S = np.linalg.slogdet(Sigma_S)

    c1_s = (k / 2) * np.log(tr_S / k) - 0.5 * log_det_S
    return c1_s


def ricomp_mm_complexity(X, y, update_scale=False):
    """
    Implements RICOMP based on MM-estimation (Eq 36).
    Combines high efficiency and high breakdown point
    """
    y = y.ravel()
    n, p = X.shape
    k = p + 1  # Total free parameters (p coefficients + 1 scale)

    # 1. Fit MM-estimator
    if update_scale:
        scale_est = BiweightScale(c=1.55)
    else:
        scale_est = FixedBiweightScale(biweight_scale(sm.OLS(y, X).fit().resid, c=1.55))
    mm_norm = sm.robust.norms.TukeyBiweight(c=4.685)
    mm_res = sm.RLM(y, X, M=mm_norm).fit(scale_est=scale_est, update_scale=update_scale)

    # 2. Extract Robust Estimates
    # statsmodels mm_res.scale is the MM-estimate of scale (sigma_hat)
    sigma_mm = mm_res.scale

    # 3. Calculate scaling factor for the robust covariance block
    # ave_i{psi^2} / [ave_i{psi'}]^2
    resids_scaled = mm_res.resid / sigma_mm
    psi_vals = mm_norm.psi(resids_scaled)
    psi_prime_vals = mm_norm.psi_deriv(resids_scaled)

    # Paper uses 'ave_i' (mean) for these scaling terms
    scaling_factor = (n / (n - p)) * (
        np.mean(psi_vals**2) / (np.mean(psi_prime_vals) ** 2)
    )

    # Top-left block: Estimated Covariance of beta_mm
    cov_beta_mm = (sigma_mm**2) * scaling_factor * np.linalg.inv(X.T @ X)

    # 4. Construct the Robust IFIM
    # Bottom-right term: 2 * sigma_mm^4 / n
    var_sigma_mm = (2 * (sigma_mm**4)) / n

    # Sigma_MM is a block diagonal matrix
    Sigma_MM = np.block(
        [[cov_beta_mm, np.zeros((p, 1))], [np.zeros((1, p)), var_sigma_mm]]
    )

    # 5. Calculate C1 Complexity
    # C1 = (k/2) * log(tr(Sigma_MM)/k) - (1/2) * log|Sigma_MM|
    tr_S = np.trace(Sigma_MM)
    _, log_det_S = np.linalg.slogdet(Sigma_MM)

    c1_mm = (k / 2) * np.log(tr_S / k) - 0.5 * log_det_S

    return c1_mm
