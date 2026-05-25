from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from shapley_regression_value import single_shap_value

X, y = make_regression(n_samples=50, n_features=3)

val = single_shap_value(
    untrained_model=LinearRegression(),
    X_train=X,
    y_train=y,
    feature_of_interest=2,
    instance=X[0, :],
)
print(val)
