import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Tuple


class BayesianLinearRegression(LinearRegression):
    """
    Bayesian Linear Regression model.

    Implements Bayesian linear regression using a Gaussian prior for the weights.
    The posterior distribution of the weights is calculated analytically.

    Parameters
    ----------
    ridge_lambda : float, default=0
        The precision parameter for the Gaussian prior. When set to a value > 0,
        it acts as a regularizer. When set to 0, a weak (uninformative) prior
        is used.

    uninformative_prior_variance : float, default=1.0
        The variance used for the weak (uninformative) prior when `ridge_lambda`
        is set to 0. A larger value makes the prior less influential.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        The posterior mean of the weights (coefficients).

    intercept_ : float
        The posterior mean of the intercept term.

    sigma_ : array of shape (n_features_including_intercept, n_features_including_intercept)
        The posterior covariance matrix of the weights, capturing the uncertainty.

    noise_variance_ : float
        The estimated variance of the noise in the data (sigma^2).
    """

    def __init__(
        self,
        ridge_lambda: float = 0,
        uninformative_prior_variance: float = 1.0,
        fit_intercept: bool = False,
    ):
        super().__init__(fit_intercept=fit_intercept)
        self.ridge_lambda = ridge_lambda
        self.uninformative_prior_variance = uninformative_prior_variance

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Bayesian Linear Regression model.
        """
        X, y = check_X_y(X, y)
        yy = y.reshape(-1, 1)

        Phi = np.c_[np.ones(X.shape[0]), X] if self.fit_intercept else X
        n_samples, n_features = Phi.shape

        # Estimate noise variance from an OLS solution (MLE of sigma^2)
        w_mle, _, _, _ = np.linalg.lstsq(Phi, yy, rcond=None)
        self.noise_variance_ = np.mean((yy - Phi @ w_mle) ** 2)

        # Define the prior distribution based on ridge_lambda
        if self.ridge_lambda > 0:
            prior_mean = np.zeros((n_features, 1))
            prior_cov = (self.noise_variance_ / self.ridge_lambda) * np.identity(
                n_features
            )
        else:
            prior_mean = w_mle
            prior_cov = self.uninformative_prior_variance * np.identity(n_features)

        prior_cov_inv = np.linalg.inv(prior_cov)

        # Calculate Posterior
        self.sigma_ = self.noise_variance_ * np.linalg.pinv(
            self.noise_variance_ * prior_cov_inv + Phi.T @ Phi
        )

        posterior_mean = self.sigma_ @ (
            prior_cov_inv @ prior_mean + (Phi.T @ yy) / self.noise_variance_
        )

        # Set scikit-learn compatible attributes
        if self.fit_intercept:
            self.intercept_ = posterior_mean[0, 0]
            self.coef_ = posterior_mean[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = posterior_mean.flatten()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the posterior mean of the weights."""
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def predict_dist(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the full predictive distribution (mean and standard deviation)."""
        check_is_fitted(self)
        X = check_array(X)

        Phi = np.c_[np.ones(X.shape[0]), X] if self.fit_intercept else X

        # Combine intercept and coefficients for matrix multiplication
        full_coef = np.hstack([self.intercept_, self.coef_]).reshape(-1, 1)
        y_mean = Phi @ full_coef

        # Variance of the predictive distribution
        variance_from_params = np.sum((Phi @ self.posterior_cov_) * Phi, axis=1)
        predictive_variance = self.noise_variance_ + variance_from_params
        y_std = np.sqrt(predictive_variance)

        return y_mean.flatten(), y_std.flatten()
