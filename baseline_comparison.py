#!/usr/bin/env python3
"""
baseline_comparison.py
──────────────────────
Compares PySINDy sparse regression baselines (STLSQ, SR3, SSR)
against ground-truth governing equations.

Protocol:
  1. Hyperparameters are tuned via 3-fold CV-MSE on the full dataset
     (no ground-truth access during tuning).
  2. The best configuration is refit on the full dataset.
  3. eFDR, Power, |S|, and %CE (when eFDR=0 and Power=1) are computed.

Supports both single-target (e.g. Burgers, KdV, KS) and multi-target
(e.g. RD, GS) datasets automatically.

Usage:
    python baseline_comparison.py --data path/to/dataset.mat
    python baseline_comparison.py --data path/to/dataset.mat --methods STLSQ SR3 SSR
"""

import argparse
import warnings
import numpy as np
from scipy import io as sio
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
from itertools import product
import pysindy as ps

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════
#  Wrapper to make PySINDy optimizers sklearn-compatible
# ═════════════════════════════════════════════════════════════════════

class SINDyOptWrapper(BaseEstimator, RegressorMixin):
    """Wraps a PySINDy optimizer class for use with sklearn cross_val_score."""

    def __init__(self, optimizer_cls, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.opt_kwargs = kwargs

    def fit(self, X, y):
        self.optimizer_ = self.optimizer_cls(**self.opt_kwargs)
        self.optimizer_.fit(X, y)
        self.coef_ = self.optimizer_.coef_
        return self

    def predict(self, X):
        return X @ self.coef_.T

    def get_params(self, deep=True):
        return {"optimizer_cls": self.optimizer_cls, **self.opt_kwargs}

    def set_params(self, **params):
        self.optimizer_cls = params.pop("optimizer_cls", self.optimizer_cls)
        self.opt_kwargs.update(params)
        return self


# ═════════════════════════════════════════════════════════════════════
#  Hyperparameter grids
# ═════════════════════════════════════════════════════════════════════

HYPERPARAMS = {
    "STLSQ": {
        "optimizer_cls": [ps.STLSQ],
        "threshold": [0.01, 0.05, 0.1, 0.5, 1.0],
        "alpha": [0, 1e-5, 1e-3],
        "unbias": [True],
    },
    "SR3": {
        "optimizer_cls": [ps.SR3],
        "reg_weight_lam": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        "regularizer": ["L0"],
        "relax_coeff_nu": [1.0, 10.0],
        "unbias": [True],
    },
    "SSR": {
        "optimizer_cls": [ps.SSR],
        "alpha": [0, 1e-5],
        "criteria": ["coefficient_value", "model_residual"],
        "unbias": [True],
    },
}


# ═════════════════════════════════════════════════════════════════════
#  Metrics
# ═════════════════════════════════════════════════════════════════════

def compute_metrics(coef_row, ground_indices, ground_coefficients, n_features):
    """
    Compute eFDR, Power, |S|, and %CE for a single target.

    Parameters
    ----------
    coef_row : array of shape (n_features,)
    ground_indices : array of shape (n_true,)
    ground_coefficients : array of shape (n_true,)
    n_features : int

    Returns
    -------
    dict with keys: eFDR, Power, support_size, pct_ce (or None)
    """
    discovered = set(np.nonzero(coef_row)[0])
    true_set = set(ground_indices)
    n_true = len(true_set)

    tp = len(discovered & true_set)
    fp = len(discovered - true_set)
    support_size = len(discovered)

    efdr = fp / max(support_size, 1)
    power = tp / max(n_true, 1)

    # %CE only when exact recovery
    pct_ce = None
    if efdr == 0.0 and power == 1.0 and support_size > 0:
        discovered_coefs = coef_row[ground_indices]
        pct_ce_per_term = np.abs(
            (discovered_coefs - ground_coefficients) / ground_coefficients
        ) * 100
        pct_ce = float(np.mean(pct_ce_per_term))

    return {
        "eFDR": round(efdr, 4),
        "Power": round(power, 4),
        "support_size": support_size,
        "pct_ce": round(pct_ce, 4) if pct_ce is not None else None,
    }


# ═════════════════════════════════════════════════════════════════════
#  CV-based tuning and evaluation
# ═════════════════════════════════════════════════════════════════════

def tune_and_evaluate(X, y, ground_indices, ground_coefficients,
                      method_name, param_grid, cv=3,
                      tuning_split=None, seed=42):
    """
    Tune hyperparameters via CV-MSE, then refit on full data.

    Parameters
    ----------
    X : array (n, p)
    y : array (n,)
    ground_indices : array (n_true,)
    ground_coefficients : array (n_true,)
    method_name : str
    param_grid : dict of lists
    cv : int
    tuning_split : float or None
        If not None (e.g. 0.5), CV is performed on a random subset of
        this fraction; evaluation is still on the full dataset.
    seed : int
        Random seed for the tuning split.

    Returns
    -------
    dict with metrics and best hyperparameters
    """
    y_flat = y.ravel()
    n_features = X.shape[1]

    # Optionally restrict CV to a tuning subset
    if tuning_split is not None:
        from sklearn.model_selection import train_test_split
        X_tune, _, y_tune, _ = train_test_split(
            X, y_flat, test_size=(1 - tuning_split), random_state=seed
        )
    else:
        X_tune, y_tune = X, y_flat

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    configs = [dict(zip(keys, combo)) for combo in product(*values)]

    best_score = -np.inf
    best_config = None

    for config in configs:
        try:
            wrapper = SINDyOptWrapper(**config)
            scores = cross_val_score(
                wrapper, X_tune, y_tune, cv=cv,
                scoring="neg_mean_squared_error"
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_config = config.copy()
        except Exception:
            continue

    if best_config is None:
        return {
            "method": method_name,
            "eFDR": None, "Power": None, "support_size": None,
            "pct_ce": None, "best_params": None, "cv_mse": None,
        }

    # Refit on full data
    wrapper = SINDyOptWrapper(**best_config)
    wrapper.fit(X, y_flat)
    coef_row = wrapper.coef_.ravel()

    metrics = compute_metrics(
        coef_row, ground_indices, ground_coefficients, n_features
    )

    display_params = {
        k: v for k, v in best_config.items() if k != "optimizer_cls"
    }

    return {
        "method": method_name,
        **metrics,
        "best_params": display_params,
        "cv_mse": round(-best_score, 8),
        "coef": coef_row,
    }


# ═════════════════════════════════════════════════════════════════════
#  Dataset loader
# ═════════════════════════════════════════════════════════════════════

def load_dataset(path):
    """Load a .mat dataset and return a standardised dictionary."""
    data = sio.loadmat(path, squeeze_me=False)
    X = np.asarray(data["candidate_library"], dtype=np.float64)
    feature_names = np.array(
        [s.strip() for s in data["feature_names"]], dtype=str
    )

    targets = np.asarray(data["targets"], dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    n_targets = targets.shape[1]

    raw_gi = np.asarray(data["ground_indices"])
    raw_gc = np.asarray(data["ground_coefficients"])

    # Handle two storage formats:
    #   1. Regular (n_targets, n_true) int/float array (equal-length supports)
    #   2. Object array of nested arrays (ragged supports, e.g. GS: 6 vs 5)
    def _unpack(raw, n_targets):
        if raw.dtype == object:
            flat = raw.flat
            return [np.asarray(flat[i]).ravel() for i in range(n_targets)]
        else:
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            return [raw[i] for i in range(n_targets)]

    ground_indices = _unpack(raw_gi, n_targets)
    ground_coefficients = _unpack(raw_gc, n_targets)

    assert len(ground_indices) == n_targets, (
        f"Mismatch: targets has {n_targets} columns but "
        f"ground_indices has {len(ground_indices)} entries"
    )

    return {
        "X": X,
        "targets": targets,
        "feature_names": feature_names,
        "ground_indices": ground_indices,
        "ground_coefficients": ground_coefficients,
        "n_targets": n_targets,
    }


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def run_baselines(data_path, methods=None, cv=3, tuning_split=None, seed=42, verbose=True):
    """Run baseline comparison on a single dataset."""
    if methods is None:
        methods = ["STLSQ", "SR3", "SSR"]

    dataset = load_dataset(data_path)
    X = dataset["X"]
    n, p = X.shape
    n_targets = dataset["n_targets"]
    feature_names = dataset["feature_names"]

    if verbose:
        print(f"Dataset: {data_path}")
        print(f"  X: ({n}, {p}), targets: {n_targets}, cv: {cv}"
              + (f", tuning_split: {tuning_split}" if tuning_split else ""))
        for t in range(n_targets):
            gi = dataset["ground_indices"][t]
            gc = dataset["ground_coefficients"][t]
            terms = ", ".join(
                f"{gc[i]:+.4f}*{feature_names[gi[i]]}"
                for i in range(len(gi))
            )
            print(f"  Target {t}: |S*|={len(gi)}, {terms}")
        print()

    all_results = []

    for method_name in methods:
        if method_name not in HYPERPARAMS:
            print(f"  Unknown method: {method_name}, skipping.")
            continue

        param_grid = HYPERPARAMS[method_name]

        for t in range(n_targets):
            y = dataset["targets"][:, t]
            gi = dataset["ground_indices"][t]
            gc = dataset["ground_coefficients"][t]

            target_label = "" if n_targets == 1 else f" (target {t})"
            if verbose:
                print(f"  {method_name}{target_label}:")

            result = tune_and_evaluate(
                X, y, gi, gc, method_name, param_grid, cv=cv,
                tuning_split=tuning_split, seed=seed
            )
            result["target_idx"] = t

            if verbose:
                ce_str = (
                    f"{result['pct_ce']:.2f}%"
                    if result["pct_ce"] is not None
                    else "N/A"
                )
                print(
                    f"    eFDR={result['eFDR']:.3f}  Power={result['Power']:.3f}  "
                    f"|S|={result['support_size']}  %CE={ce_str}"
                )
                print(f"    params: {result['best_params']}")
                print(f"    CV-MSE: {result['cv_mse']}")

                if result.get("coef") is not None:
                    coef = result["coef"]
                    nz = np.nonzero(coef)[0]
                    terms = ", ".join(
                        f"{coef[j]:+.4f}*{feature_names[j]}" for j in nz
                    )
                    print(f"    discovered: {terms}")
                print()

            all_results.append(result)

    # ── Summary table ───────────────────────────────────────────────
    if verbose:
        print("=" * 72)
        print(f"{'Method':<12} {'Target':>6} {'eFDR':>6} {'Power':>6} "
              f"{'|S|':>4} {'%CE':>8}")
        print("-" * 72)
        for r in all_results:
            ce = f"{r['pct_ce']:.2f}" if r["pct_ce"] is not None else "N/A"
            tgt = r["target_idx"] if n_targets > 1 else ""
            print(f"{r['method']:<12} {tgt:>6} {r['eFDR']:>6.3f} "
                  f"{r['Power']:>6.3f} {r['support_size']:>4} {ce:>8}")
        print("=" * 72)

        # LaTeX rows
        print("\nLaTeX rows:")
        for r in all_results:
            ce = f"{r['pct_ce']:.2f}" if r["pct_ce"] is not None else "---"
            tgt_label = f" ({r['target_idx']})" if n_targets > 1 else ""
            print(
                f" & {r['method']}{tgt_label:6s} & {r['eFDR']:.3f} "
                f"& {r['Power']:.3f} & {ce} & {r['support_size']} \\\\"
            )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PySINDy baseline comparison (3-fold CV tuning)."
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=["STLSQ", "SR3", "SSR"])
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--split", type=float, default=None,
        help="Tuning split fraction (e.g. 0.5). Default: None (CV on full data)."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_baselines(args.data, methods=args.methods, cv=args.cv,
                  tuning_split=args.split, seed=args.seed)
