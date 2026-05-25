from itertools import chain, combinations
from math import factorial
import numpy as np
from tqdm import trange


def get_all_subsets(items):
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


def get_all_other_feature_subsets(n_features, feature_of_interest):
    all_other_features = [j for j in range(n_features) if j != feature_of_interest]
    return get_all_subsets(all_other_features)


def subset_model(model, X_train, y_train, feature_subset, instance):
    assert len(instance.shape) == 1, "Instance must be a 1D array"
    if len(feature_subset) == 0:
        return y_train.mean()  # a model with no features predicts E[y]
    X_subset = X_train.take(feature_subset, axis=1)
    model.fit(X_subset, y_train)
    return model.predict(instance.take(feature_subset).reshape(1, -1))[0]


def permutation_factor(n_features, n_subset):
    return (
        factorial(n_subset)
        * factorial(n_features - n_subset - 1)
        / factorial(n_features)
    )


def single_shap_value(untrained_model, X_train, y_train, feature_of_interest, instance):
    "Compute a single SHAP value (equation 4)"
    n_features = X_train.shape[1]
    shap_value = 0
    for subset in get_all_other_feature_subsets(n_features, feature_of_interest):
        n_subset = len(subset)
        prediction_without_feature = subset_model(
            untrained_model, X_train, y_train, subset, instance
        )
        prediction_with_feature = subset_model(
            untrained_model, X_train, y_train, subset + (feature_of_interest,), instance
        )
        factor = permutation_factor(n_features, n_subset)
        shap_value += factor * (prediction_with_feature - prediction_without_feature)
    return shap_value

# from shapley_regression_value import shapley_regression_value
# shap_values = shapley_regression_value(LinearRegression(fit_intercept=fit_intercept), X_pre_top, y_pre, full=False)
# shap_values /= shap_values.sum()
# shap_values
def shapley_regression_value(model, X, y, full=True):
    n_samples, n_features = X.shape
    shap_values_raw = np.zeros((n_samples, n_features))
    y_flat = y.ravel()

    for i in trange(n_samples):
        for j in range(n_features):
            shap_values_raw[i, j] = single_shap_value(model, X, y_flat, j, X[i, :])

    if not full:
        shap_values_raw = abs(shap_values_raw).mean(axis=0)

    return shap_values_raw
