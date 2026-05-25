from decimal import Decimal
import itertools

import numpy as np
from sklearn import linear_model
from sklearn.inspection import permutation_importance

from kneed import KneeLocator
from kneefinder import KneeFinder

import shap

import yaml

def read_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    f.close()
    return content

def knee(x, y, S=0.95, interp_method='linear', degree=7, direction='decreasing'):
    if direction == 'decreasing':
        curve = 'convex'
    elif direction == 'increasing':
        curve = 'concave'

    if interp_method == 'polynomial':
        opt = KneeLocator(x, y, S=S, interp_method='polynomial', polynomial_degree=degree, 
                          curve=curve, direction=direction).knee
    elif interp_method == 'linear':
        opt = KneeLocator(x, y, S=S, curve=curve, direction=direction).knee
    else:
        raise Exception("Unknown interp_method...")

    if direction == 'decreasing':
        argopt = x[np.argmin(y)]
    elif direction == 'increasing':
        argopt = x[np.argmax(y)]

    if opt is not None:
        opt = min(opt, argopt)
    else:
        opt = argopt
    return opt

def knee_finder(y, decreasing=False):
    y = np.array(y)
    if decreasing:
        decreasing_indices = range(0, len(y))
    else:
        decreasing_indices = decreasing_values_indices(y)
    if len(decreasing_indices) == 2 and y[1] < y[0]:
        return 1
    kf = KneeFinder(decreasing_indices, y[decreasing_indices])
    return int(kf.find_knee()[0])

def colvec(arr):
    return arr.reshape(-1, 1)

def normalize_lp(arr, p=2, axis=0):
    arr_norm = np.linalg.norm(arr, p, axis=axis)
    return arr/arr_norm, arr_norm

def select_column(arr, idx):
    assert len(arr.shape) == 2
    return arr[:, idx:idx+1]

def decreasing_values_indices(arr):
    mini = max(arr)+1; out = []
    for i, e in enumerate(arr):
        if e < mini:
            mini = e
            out.append(i)
    return np.array(out)

def distribute_order(n_poly, n_vars):
    out = []
    for n in range(n_poly+1):
        distribution = []
        for event in itertools.product(range(n+1), repeat=n_vars):
            if sum(event) == n:
                distribution.append(event)
        out.extend(sorted(distribution, reverse=True))
    return out

def biggest_superset(sets_list):
    max_superset = None
    max_count = 0
    for s in sets_list:
        count = sum(1 for other in sets_list if other.issubset(s))
        if count > max_count:
            max_count = count
            max_superset = s
    return max_superset

def MSE(arr1, arr2=None):
    D = arr1
    if arr2 is not None:
        D = D - arr2
    return (D**2).mean()

def sci_format(n):
    sf = '%.2E' % Decimal(n)
    sf = sf.split('E')
    return float(sf[0]), int(sf[1])

def permutation_linear_importance(X_pre, y_pre, n_repeats=30, scale=False, full=False, alpha=0):
    alpha = max(alpha, 0)
    if alpha > 0:
        lm = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    else:
        lm = linear_model.LinearRegression(fit_intercept=False)
    lm.fit(X_pre, y_pre)
    pi = permutation_importance(lm, X_pre, y_pre, n_repeats=30)
    if not full:
        pi = pi['importances_mean']
        if scale:
            pi = pi/pi.sum()
    else:
        pi = pi['importances'].T
    return pi

def shap_linear_importance(X_pre, y_pre, scale=True, full=False, alpha=0):
    alpha = max(alpha, 0)
    if alpha > 0:
        lm = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    else:
        lm = linear_model.LinearRegression(fit_intercept=False)
    lm.fit(X_pre, y_pre)
    explainer = shap.explainers.Linear(lm, X_pre)
    feature_importance = explainer(X_pre).values
    if not full:
        feature_importance = abs(feature_importance).mean(axis=0)
        if scale:
            feature_importance = feature_importance/feature_importance.sum()
    return feature_importance

def shap_model_selection(X_pre, y_pre, threshold=0.99):
    full_importance_values = shap_linear_importance(X_pre, y_pre, scale=False, full=True)
    importance_values = abs(full_importance_values).mean(axis=0)
    importance_values = importance_values/importance_values.sum()
    importance_ranking = np.argsort(-importance_values)
    importance_ranking = importance_ranking[:np.searchsorted(np.cumsum(importance_values[importance_ranking]), threshold)+1]
    return importance_ranking, full_importance_values[:, importance_ranking]

def extract_unique_candidates(pareto_optimal_models):
    unique_candidates = frozenset()
    for i in range(len(pareto_optimal_models)):
        unique_candidates = unique_candidates.union(pareto_optimal_models[i][0])
    return sorted(unique_candidates)

# Linear regression with feature importances
class LinearRegressionWithFI(linear_model.LinearRegression):
    def __init__(self, fit_intercept=False):
        super().__init__(fit_intercept=fit_intercept)
        self.feature_importances_ = None

    def fit(self, X, y):
        # Fit the original LinearRegression model
        super().fit(X, y)

        # Compute SHAP values
        explainer = shap.explainers.Linear(self, X)
        shap_values = explainer(X).values

        # Mean absolute SHAP value
        self.feature_importances_ = np.abs(shap_values).mean(axis=0)
        return self

