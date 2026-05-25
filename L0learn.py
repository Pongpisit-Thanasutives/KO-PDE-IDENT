from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import l0learn
from bayesian_model_evidence import log_evidence

class L0Learn(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for the L0Learn regression model.
    This version uses L0L2 penalty and supports hyperparameter tuning.

    Parameters:
    -----------
    penalty : str
        Penalty type, one of {'L0', 'L0L2', 'L0L1'}
    max_support_size : int
        Maximum number of non-zero coefficients allowed.
    intercept : bool
        Whether to fit the intercept.
    num_gamma : int
        Number of gamma values to try in the regularization path.
    gamma_min : float
        Minimum gamma value.
    gamma_max : float
        Maximum gamma value.
    algorithm : str
        Algorithm to use, one of {'CD', 'CDPSI'}
    """

    def __init__(self,
                 num_folds=5,
                 penalty="L0L2",
                 max_support_size=10,
                 intercept=False,
                 num_gamma=20,
                 gamma_min=1e-5,
                 gamma_max=1,
                 algorithm="CDPSI"):
        self.num_folds = num_folds
        self.penalty = penalty
        self.max_support_size = max_support_size
        self.intercept = intercept
        self.num_gamma = num_gamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.algorithm = algorithm

    def fit(self, X, y):
        if self.num_folds > 1:
            self.model_ = l0learn.cvfit(
                X, y, num_folds=5, 
                penalty=self.penalty,
                max_support_size=self.max_support_size,
                intercept=self.intercept,
                num_gamma=self.num_gamma,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                algorithm=self.algorithm
            )
        else:
            self.model_ = l0learn.fit(
                X, y, 
                penalty=self.penalty,
                max_support_size=self.max_support_size,
                intercept=self.intercept,
                num_gamma=self.num_gamma,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                algorithm=self.algorithm
            )
            
        # Select according to an BIC or Bayesian model evidence
        coeffs = self.model_.coeffs[0].toarray()
        log_evidences = [log_evidence(X, y.reshape(-1, 1), effective_indices=np.nonzero(coeffs[:, _])[0]) 
                         for _ in range(coeffs.shape[1]) if np.count_nonzero(coeffs[:, _]) <= self.max_support_size]
        self.coef_ = coeffs[:, np.argmax(log_evidences)]
        return self

    def predict(self, X):
        """Make predictions."""
        return X @ self.coef_

    def get_support(self):
        """Get indices of non-zero features."""
        return np.nonzero(self.coef_)[0]

    def get_support_feature_names(self, feature_names):
        """Return feature names of selected features."""
        return np.array(feature_names)[self.get_support()]

