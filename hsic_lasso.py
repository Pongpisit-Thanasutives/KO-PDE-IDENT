"""
HSICLassoSelector: A scikit-learn compatible feature selector based on pyHSICLasso.

Wraps the HSICLasso algorithm (https://github.com/riken-aip/pyHSICLasso) as a
scikit-learn estimator, compatible with Pipeline, GridSearchCV, and other
sklearn utilities.

Usage:
    from hsic_lasso import HSICLassoSelector

    selector = HSICLassoSelector(num_feat=6, B=50, task="regression")
    selector.fit(X, y)
    X_selected = selector.transform(X)

    # Or use in a Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    pipe = Pipeline([
        ("selector", HSICLassoSelector(num_feat=10, B=50)),
        ("model", Ridge()),
    ])
    pipe.fit(X_train, y_train)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y


class HSICLassoSelector(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature selector using HSIC Lasso.

    HSIC Lasso (Hilbert-Schmidt Independence Criterion Lasso) selects features
    by maximising the dependence between selected features and the target,
    while minimising redundancy among selected features.

    Parameters
    ----------
    num_feat : int, default=5
        Number of features to select.

    B : int, default=0
        Block size for block HSIC Lasso. 0 means no blocking (full HSIC).
        Larger B increases accuracy at the cost of memory/time.
        Recommended: 20–100 for large datasets.

    M : int, default=1
        Number of permutations (only used when B > 0).

    task : str, default="regression"
        Either "regression" or "classification".
        Controls which HSIC Lasso variant is applied internally.

    covtype : str, default="Gaussian"
        Kernel type for the feature covariance. Options: "Gaussian", "Delta".
        Use "Delta" for discrete/categorical features.

    kerneltype : str, default="Gaussian"
        Kernel type for the target. Options: "Gaussian", "Delta".
        Use "Delta" for classification targets.

    n_jobs : int, default=1
        Number of parallel jobs (passed to pyHSICLasso internally).

    discrete_x : bool, default=False
        If True, uses the Delta kernel for all input features.
        Overrides `covtype` to "Delta".

    max_neighbors : int, default=5
        Max neighbors for k-NN graph (used internally by pyHSICLasso
        for certain kernel configurations).

    feature_names : array-like of str or None, default=None
        Optional feature names passed to pyHSICLasso for interpretability.
        If None, names are auto-generated ("x0", "x1", ...).

    random_state : int or None, default=None
        Seed for reproducibility (pyHSICLasso uses numpy random internally).

    Attributes
    ----------
    selected_indices_ : ndarray of shape (num_feat,)
        Indices of selected features in the original feature space.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Importance scores for all features; zero for unselected features,
        HSIC score for selected features.

    hsic_scores_ : ndarray of shape (num_feat,)
        HSIC scores for the selected features (ordered as returned by
        pyHSICLasso).

    beta_ : ndarray
        Raw beta coefficients from pyHSICLasso.

    hsic_lasso_ : HSICLasso
        The fitted pyHSICLasso HSICLasso object (for advanced inspection).

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from hsic_lasso import HSICLassoSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 20))
    >>> y = X[:, 0] * 2 + X[:, 3] - X[:, 7] + rng.standard_normal(200) * 0.1
    >>> sel = HSICLassoSelector(num_feat=3, B=50, random_state=42)
    >>> sel.fit(X, y)
    HSICLassoSelector(B=50, num_feat=3, random_state=42)
    >>> sel.selected_indices_
    array([0, 3, 7])  # (may vary)
    >>> X_sel = sel.transform(X)
    >>> X_sel.shape
    (200, 3)
    """

    def __init__(
        self,
        num_feat: int = 5,
        B: int = 0,
        M: int = 1,
        task: str = "regression",
        covtype: str = "Gaussian",
        kerneltype: str = "Gaussian",
        n_jobs: int = 1,
        discrete_x: bool = False,
        max_neighbors: int = 5,
        feature_names=None,
        random_state=None,
    ):
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.task = task
        self.covtype = covtype
        self.kerneltype = kerneltype
        self.n_jobs = n_jobs
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.feature_names = feature_names
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature_names(self, n_features: int):
        """Return feature name list, falling back to auto-generated names."""
        if self.feature_names is not None:
            names = list(self.feature_names)
            if len(names) != n_features:
                raise ValueError(
                    f"feature_names has length {len(names)} but X has "
                    f"{n_features} features."
                )
            return names
        return [f"x{i}" for i in range(n_features)]

    def _seed(self):
        """Seed numpy if random_state is set."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the HSIC Lasso selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : HSICLassoSelector
            Fitted estimator.
        """
        try:
            from pyHSICLasso import HSICLasso
        except ImportError as exc:
            raise ImportError(
                "pyHSICLasso is required. Install it with:\n"
                "    pip install pyHSICLasso"
            ) from exc

        X, y = check_X_y(X, y, ensure_2d=True, y_numeric=(self.task == "regression"))
        y = y.ravel()

        self.n_features_in_ = X.shape[1]
        feat_names = self._get_feature_names(self.n_features_in_)

        self._seed()

        # Initialise and fit pyHSICLasso
        hl = HSICLasso()
        hl.input(X, y, featname=feat_names)

        run_kwargs = dict(
            num_feat=self.num_feat,
            B=self.B,
            M=self.M,
            n_jobs=self.n_jobs,
            covtype=self.covtype if not self.discrete_x else "Delta",
            kerneltype=self.kerneltype,
        )

        if self.task == "regression":
            hl.regression(**run_kwargs)
        elif self.task == "classification":
            hl.classification(**run_kwargs)
        else:
            raise ValueError(
                f"task must be 'regression' or 'classification', got '{self.task}'."
            )

        # Persist the fitted object and derived attributes
        self.hsic_lasso_ = hl
        self.selected_indices_ = np.array(hl.get_index(), dtype=int)
        self.hsic_scores_ = np.array(hl.get_index_score(), dtype=float)
        self.beta_ = hl.beta.ravel()

        # Build full-length importance vector (zero for unselected features)
        importances = np.zeros(self.n_features_in_, dtype=float)
        importances[self.selected_indices_] = self.hsic_scores_
        self.feature_importances_ = importances

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_selected : ndarray of shape (n_samples, num_feat)
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}."
            )
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_support(self, indices: bool = False):
        """
        Return a mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices; otherwise return a boolean mask.

        Returns
        -------
        support : ndarray
        """
        check_is_fitted(self)
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """
        Return feature names for selected features (sklearn ≥ 1.0).

        Parameters
        ----------
        input_features : array-like of str or None
            If provided, used as feature names; otherwise uses the names
            supplied at construction (or auto-generated names).

        Returns
        -------
        feature_names_out : ndarray of str
        """
        check_is_fitted(self)
        if input_features is not None:
            names = np.asarray(input_features)
        elif self.feature_names is not None:
            names = np.asarray(self.feature_names)
        else:
            names = np.array([f"x{i}" for i in range(self.n_features_in_)])
        return names[self.selected_indices_]

    def _more_tags(self):
        """Tell sklearn this estimator requires y during fit."""
        return {"requires_y": True}

    # ------------------------------------------------------------------
    # Convenience display
    # ------------------------------------------------------------------

    def summary(self, input_features=None):
        """
        Print a ranked summary of selected features.

        Parameters
        ----------
        input_features : array-like of str or None
        """
        check_is_fitted(self)
        names = self.get_feature_names_out(input_features)
        print(f"{'Rank':<6} {'Feature':<30} {'HSIC Score':>12}")
        print("-" * 52)
        order = np.argsort(self.hsic_scores_)[::-1]
        for rank, idx in enumerate(order, 1):
            print(f"{rank:<6} {names[idx]:<30} {self.hsic_scores_[idx]:>12.6f}")
