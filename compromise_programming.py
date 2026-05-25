# Basics
import numpy as np
import statsmodels.api as sm
from collections import Counter
from scipy.stats import rankdata

# MCDM/MCDA by pymcdm
from pymcdm import weights as obj_w
from pymcdm.methods import TOPSIS, MABAC, COMET, SPOTIS, VIKOR, COPRAS, EDAS, PROMETHEE_II, COCOSO
from pymcdm.methods.comet_tools import MethodExpert
from pymcdm import normalizations

from icomp import ricomp_mm_complexity

try:
    # MCDM/MCDA by pyDecision [EXTRA]
    from pyDecision.algorithm import (
        topsis_method,
        vikor_method,
        copras_method,
        mabac_method,
        marcos_method,
    )
except ImportError:
    print("pyDecision is not installed in the environment.")

# Rank aggregation methods for MCDA problems
from pyRankMCDA.algorithm import rank_aggregation

# Sparse regression
from sklearn.linear_model import ARDRegression

# from bayesian_linear_regression import BayesianLinearRegression
from bayesian_model_evidence import log_evidence
from UBIC import BIC_AIC


# OPTION-I: sparse_regressor=ARDRegression, regressor_kwargs={"fit_intercept": False, "compute_score": True, "threshold_lambda": 1e6}
# OPTION-II: sparse_regressor=BayesianLinearRegression, regressor_kwargs={}
# generate_alternatives + recursive_mcdm = optimal_decision
def generate_alternatives(
    best_subsets,
    dataset,
    sparse_regressor=ARDRegression,
    regressor_kwargs={"fit_intercept": False, "compute_score": True, "threshold_lambda": 1e6},
    data_normalization=lambda _: _,
    ssr_normalization=lambda _: _,
    ic_normalization=lambda _: _ - np.min(_),  # relative ic
    uncertainty_normalization=lambda _: _ / np.min(_),  # relative uncertainty
    include_uncertainty=True,
):
    X, y = dataset
    y = y.ravel()

    F = {}
    BS = {}
    for efi in best_subsets:
        nX = data_normalization(X[:, efi])
        um = sparse_regressor(**regressor_kwargs)
        um.fit(nX, y)
        pred = um.predict(nX)

        # number of effective parameters
        um_n_params = np.count_nonzero(um.coef_)

        # SSR
        ssr = np.sum((pred - y) ** 2)

        # Information criterion
        bic, aic = BIC_AIC(pred, y, um_n_params)
        ic = bic

        # PDE uncertainty
        pde_uncertainty = np.linalg.norm(
            np.sqrt(np.diag(um.sigma_)), 1
        ) / np.linalg.norm(um.coef_, 1)

        # Track PDE stat
        pde_stat = (ssr, pde_uncertainty)
        if um_n_params not in F or F[um_n_params] < pde_stat:
            F[um_n_params] = pde_stat
            BS[um_n_params] = efi

    F = np.column_stack((list(F.keys()), list(F.values())))
    if not include_uncertainty:
        F = F[:, :-1]
    else:
        F[:, -1] = F[:, -1] / np.min(F[:, -1])

    BS = np.array([BS[_] for _ in sorted(BS)], dtype=object)

    return F, BS


def recursive_mcdm(
    F,
    weight="entropy_weights",
    types=None,
    rank_aggregation_methods=["bd", "rrf", "sc"],
    verbose=True,
):
    if types is None:
        types = np.array(
            [-1 for _ in range(F.shape[-1])]
        )  # Default types if not provided

    # Compute objective weights based on Gini index or any other method
    obj_weights = getattr(obj_w, weight)(F, types=np.array(types))
    if verbose:
        print("Weights:", obj_weights)

    filtered_F = F.copy()  # Create a copy of F to start the filtering process

    # Perform the recursive MCDM filtering process
    for _ in range(20):
        if len(filtered_F) <= 2:
            break

        # Rank the alternatives based on the current weights and types
        ranks, prefs = mcdm(filtered_F, obj_weights, types)
        # ranks, prefs = pydecision_mcdm(filtered_F, obj_weights, types)

        # Find the most common ranks and sort them by frequency and rank index
        most_common = ranks2decision(ranks, methods=rank_aggregation_methods)
        keep_until = most_common[0][0]

        if verbose:
            print(most_common, filtered_F)

        # Update filtered_F based on the rank criteria
        filtered_F = filtered_F[: keep_until + 1]

        if len(most_common) == 1:
            # If only one most common rank remains, exit the loop
            break

    return most_common, filtered_F


def optimal_decision(
    best_subsets,
    dataset,
    sparse_regressor=ARDRegression,
    regressor_kwargs={"fit_intercept": False, "compute_score": True, "threshold_lambda": 1e6},
    data_normalization=lambda _: _,
    ssr_normalization=lambda _: _,
    ic_normalization=lambda _: _ - np.min(_),  # relative ic
    uncertainty_normalization=lambda _: _ / np.min(_),  # relative uncertainty
    include_uncertainty=True,
    weight="entropy_weights",
    types=None,
    rank_aggregation_methods=["bd", "rrf", "sc"],
    verbose=True,
):
    F, BS = generate_alternatives(
        best_subsets,
        dataset,
        sparse_regressor=sparse_regressor,
        regressor_kwargs=regressor_kwargs,
        data_normalization=data_normalization,
        ssr_normalization=ssr_normalization,
        ic_normalization=ic_normalization,
        uncertainty_normalization=uncertainty_normalization,
        include_uncertainty=include_uncertainty,
    )

    if verbose:
        print("Generate alternatives:", BS)
        print(F)
        print()
        print("Recursive MCDM")

    most_common, filtered_F = recursive_mcdm(
        F,
        weight=weight,
        types=types,
        rank_aggregation_methods=rank_aggregation_methods,
        verbose=verbose,
    )

    return most_common, filtered_F


def compromise_programming(
    best_subsets,
    dataset,
    weight="entropy_weights",
    criterion="bic",
    ssr_normalization=None,
    ic_normalization=None,
    ic_kwargs={},  # for criterion="bme"
    rank_aggregation_methods=["bd", "rrf", "sc"],
    verbose=True,
):
    assert criterion in {"bic", "bme"}
    XX, yy = dataset

    ssr = []
    ic = []
    complexity = []
    for efi in best_subsets:
        # SSR and Complexity
        ols_result = sm.OLS(yy, XX[:, efi]).fit()
        ssr.append(ols_result.ssr)
        # complexity.append(ricomp_mm_complexity(XX[:, efi], yy))
        complexity.append(len(ols_result.params))
        # Criterion: BIC or Bayesian Model Evidence (BME)
        if criterion == "bme":
            ic.append(-log_evidence(XX[:, efi], yy, **ic_kwargs))
        else:
            ic.append(ols_result.bic)
    ic = np.array(ic)

    F = np.stack((ssr, complexity), axis=1)
    n_alternatives, n_criteria = F.shape
    types = np.array([-1 for _ in range(n_criteria)])  # default to minimization
    if ssr_normalization is not None:
        if np.any(F[:, 0:1] < 0):
            F[:, 0:1] = -getattr(normalizations, ssr_normalization)(-F[:, 0:1])
        else:
            F[:, 0:1] = getattr(normalizations, ssr_normalization)(F[:, 0:1])
    obj_weights = getattr(obj_w, weight)(F, types=np.array(types))

    filtered_F = F.copy()
    filtered_F[:, 0:1] = ic.reshape(-1, 1)
    if ic_normalization is not None:
        if np.any(filtered_F[:, 0:1] < 0):
            filtered_F[:, 0:1] = -getattr(normalizations, ic_normalization)(
                -filtered_F[:, 0:1]
            )
        else:
            filtered_F[:, 0:1] = getattr(normalizations, ic_normalization)(
                filtered_F[:, 0:1]
            )

    decision_seq = []
    preference_seq = []
    while len(filtered_F) > 2:
        ranks, prefs = mcdm(filtered_F, obj_weights, types)
        # ranks, prefs = pydecision_mcdm(filtered_F, obj_weights, types)
        most_common = ranks2decision(ranks, methods=rank_aggregation_methods)
        balance_point = most_common[0][0]
        if verbose:
            print(filtered_F, most_common)

        filtered_F = filtered_F[: balance_point + 1]

        decision_seq.append(most_common)
        preference_seq.append(prefs)

        if len(most_common) == 1:
            break

    return F, decision_seq, preference_seq


def mcdm(F, obj_weights, types):
    """Ensemble MCDM using 5 paradigm-diverse methods.

    Methods and their decision paradigms:
        TOPSIS        — distance to ideal/anti-ideal (Euclidean geometry)
        VIKOR         — compromise: group utility vs individual regret (min-max)
        COMET         — pairwise characteristic objects (learned fuzzy model)
        PROMETHEE II  — outranking via pairwise preference flows
        COCOSO        — hybrid additive + multiplicative aggregation (WSM+WPM)

    Parameters
    ----------
    F : ndarray, shape (n_alternatives, n_criteria)
        Decision matrix.
    obj_weights : ndarray, shape (n_criteria,)
        Criteria weights (should sum to 1).
    types : ndarray, shape (n_criteria,)
        Criteria types: +1 for profit, -1 for cost.

    Returns
    -------
    ranks : ndarray, shape (n_methods, n_alternatives)
        Ranking from each method (1 = best).
    prefs : ndarray, shape (n_methods, n_alternatives)
        Raw preference values from each method.
    """
    cvalues = None
    try:
        cvalues = COMET.make_cvalues(F)
    except ValueError:
        pass

    candidates = [
        ('TOPSIS', lambda: TOPSIS()),                            # distance to ideal/anti-ideal
        ('VIKOR', lambda: VIKOR()),                              # compromise: group utility vs individual regret
        ('COMET', lambda: COMET(cvalues,                         # pairwise characteristic objects (fuzzy model)
                    MethodExpert(TOPSIS(), obj_weights, types))),
        ('PROMETHEE_II', lambda: PROMETHEE_II('usual')),         # outranking via pairwise preference flows
        ('COCOSO', lambda: COCOSO()),                            # hybrid additive + multiplicative aggregation
    ]

    names, prefs_list, ranks_list = [], [], []
    for name, build in candidates:
        try:
            if name == 'COMET' and cvalues is None:
                continue
            method = build()
            p = method(F, obj_weights, types)
            r = method.rank(p)
            names.append(name)
            prefs_list.append(p)
            ranks_list.append(r)
        except (ValueError, ZeroDivisionError):
            continue

    return np.array(ranks_list), np.array(prefs_list)


def pydecision_mcdm(F, obj_weights, types):
    # types
    type_dict = {-1: "min", 1: "max"}
    criterion_type = [type_dict[_] for _ in types]

    # topsis
    topsis_pref = topsis_method(
        F, obj_weights, criterion_type, graph=False, verbose=False
    )
    # vikor
    vikor_pref = vikor_method(
        F,
        obj_weights,
        criterion_type,
        strategy_coefficient=0.5,
        graph=False,
        verbose=False,
    )
    vikor_pref = vikor_pref[0]
    vikor_pref = 1 - vikor_pref[vikor_pref[:, 0].argsort()][:, 1]
    # mabac
    mabac_pref = mabac_method(F, obj_weights, criterion_type, verbose=False)
    # marcos
    marcos_pref = marcos_method(
        F, obj_weights, criterion_type, graph=False, verbose=False
    )
    # copras
    copras_pref = copras_method(
        F, obj_weights, criterion_type, graph=False, verbose=False
    )[:, 1]

    # ranking
    prefs = np.vstack([topsis_pref, vikor_pref, mabac_pref, marcos_pref, copras_pref])
    ranks = rankdata(-prefs, axis=1)

    return ranks, prefs


# rank aggregation
def ranks2decision(ranks, methods=["bd", "rrf", "sc"]):
    ranks = (
        rank_aggregation(np.array(ranks).T.astype(np.int32))
        .run_methods(methods=methods)
        .values
    )
    ranks = (np.argsort(ranks, axis=0) + 1).T
    most_common = Counter(np.argmin(ranks, axis=1)).most_common()
    most_common = sorted(most_common, key=lambda _: (_[1], _[0]), reverse=True)
    return most_common
