"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Robustness metrics.

This module computes scalar robustness metrics from episodic
evaluation results. These metrics are used for quantitative
comparison of attacks, defenses, and training strategies.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Core utilities
# ------------------------------------------------------------

def aggregate_by_epsilon(
    returns: np.ndarray,
    epsilons: np.ndarray,
) -> Dict[float, np.ndarray]:
    """
    Group returns by epsilon value.
    """
    groups: Dict[float, list] = {}
    for r, e in zip(returns, epsilons):
        groups.setdefault(float(e), []).append(float(r))

    return {e: np.asarray(v) for e, v in groups.items()}


# ------------------------------------------------------------
# Mean robustness curve
# ------------------------------------------------------------

def mean_return_curve(
    returns: np.ndarray,
    epsilons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean return as a function of epsilon.
    """
    grouped = aggregate_by_epsilon(returns, epsilons)
    eps = np.array(sorted(grouped.keys()))
    mean_returns = np.array([grouped[e].mean() for e in eps])
    return eps, mean_returns


# ------------------------------------------------------------
# Area under curve (AUC-ε)
# ------------------------------------------------------------

def auc_epsilon(
    eps: np.ndarray,
    mean_returns: np.ndarray,
) -> float:
    """
    Compute area under the return-vs-epsilon curve using
    trapezoidal integration.
    """
    return float(np.trapz(mean_returns, eps))


# ------------------------------------------------------------
# Relative performance drop
# ------------------------------------------------------------

def relative_drop(
    clean_return: float,
    attacked_return: float,
) -> float:
    """
    Relative performance degradation.
    """
    denom = max(abs(clean_return), 1e-8)
    return (clean_return - attacked_return) / denom


# ------------------------------------------------------------
# Worst-case return
# ------------------------------------------------------------

def worst_case_return(
    returns: np.ndarray,
) -> float:
    """
    Minimum episodic return observed.
    """
    return float(np.min(returns))


# ------------------------------------------------------------
# CVaR (Conditional Value at Risk)
# ------------------------------------------------------------

def cvar(
    returns: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Compute CVaR at level alpha.

    CVaR_alpha = mean of the worst alpha fraction of returns.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    sorted_returns = np.sort(returns)
    k = max(1, int(alpha * len(sorted_returns)))
    return float(sorted_returns[:k].mean())


# ------------------------------------------------------------
# Convenience: compute all metrics from CSV
# ------------------------------------------------------------

def compute_metrics_from_csv(
    csv_path: str,
    *,
    cvar_alpha: float = 0.1,
) -> Dict[str, float]:
    """
    Compute standard robustness metrics directly from an episode CSV.
    """
    df = pd.read_csv(csv_path)

    returns = df["total_return"].to_numpy()
    epsilons = df["epsilon"].to_numpy()

    eps, mean_curve = mean_return_curve(returns, epsilons)

    metrics = {
        "auc_epsilon": auc_epsilon(eps, mean_curve),
        "worst_case_return": worst_case_return(returns),
        f"cvar_{cvar_alpha}": cvar(returns, alpha=cvar_alpha),
    }

    return metrics
