import jax.numpy as jnp
from sklearn.base import BaseEstimator, RegressorMixin
from skscope import ScopeSolver
from skscope import utilities as sco_utilities

class SCO(BaseEstimator, RegressorMixin):
    def __init__(self, path_type='gs', sparsity=10, ic_method="LinearSIC"):
        self.path_type = path_type
        self.sparsity = range(1, sparsity+1)
        self.ic_method = getattr(sco_utilities, ic_method)
        self.solver = None
        self.coef_ = None

    def fit(self, X, y):
        n, p = X.shape
        self.solver = ScopeSolver(dimensionality=p, path_type=self.path_type, sparsity=self.sparsity, sample_size=n, ic_method=self.ic_method)
        self.coef_ = self.solver.solve(lambda params: jnp.mean((y-X@params)**2), jit=True)
        return self

    def predict(self, X):
        return jnp.dot(X, self.coef_)

