"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Risk-sensitive metrics.

This module implements tail-risk metrics such as CVaR and
worst-percentile return, which are critical for evaluating
robustness under adversarial perturbations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from road_rl.metrics.base import Metric, MetricResult


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _group_by_epsilon(
    episodes: Iterable[Mapping[str, Any]],
) -> Dict[float, List[float]]:
    """
    Group episodic returns by epsilon.
    """
    groups: Dict[float, List[float]] = defaultdict(list)
    for ep in episodes:
        eps = float(ep["epsilon"])
        ret = float(ep["return"])
        groups[eps].append(ret)
    return dict(groups)


def _cvar(values: np.ndarray, alpha: float) -> float:
    """
    Compute CVaR at level alpha (lower tail).
    """
    if values.size == 0:
        return float("nan")

    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    q = np.quantile(values, alpha)
    tail = values[values <= q]
    return float(np.mean(tail)) if tail.size > 0 else float(q)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

class CVaRReturn(Metric):
    """
    Conditional Value at Risk (CVaR) of episodic returns.

    CVaR is computed over the lower alpha-quantile of returns.
    """

    def __init__(self, alpha: float = 0.1):
        self._alpha = float(alpha)

    @property
    def name(self) -> str:
        return f"cvar_return_alpha_{self._alpha:.2f}"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        returns = np.array([float(ep["return"]) for ep in episodes], dtype=float)

        value = _cvar(returns, self._alpha)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "alpha": self._alpha,
                "n_episodes": int(returns.size),
            },
        )


class CVaRReturnPerEpsilon(Metric):
    """
    CVaR of returns computed separately for each epsilon.
    """

    def __init__(self, alpha: float = 0.1):
        self._alpha = float(alpha)

    @property
    def name(self) -> str:
        return f"cvar_return_per_epsilon_alpha_{self._alpha:.2f}"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        groups = _group_by_epsilon(episodes)

        per_eps = {
            eps: _cvar(np.array(vals, dtype=float), self._alpha)
            for eps, vals in groups.items()
        }

        # Aggregate value: mean CVaR across epsilons
        value = (
            float(np.mean(list(per_eps.values())))
            if per_eps
            else float("nan")
        )

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "alpha": self._alpha,
                "per_epsilon": per_eps,
            },
        )


class WorstPercentileReturn(Metric):
    """
    Mean return of the worst p-percent of episodes.

    Example: p=10 computes the mean of the worst 10% returns.
    """

    def __init__(self, percentile: float = 10.0):
        self._percentile = float(percentile)

    @property
    def name(self) -> str:
        return f"worst_{self._percentile:.1f}_percent_return"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        returns = np.array([float(ep["return"]) for ep in episodes], dtype=float)

        if returns.size == 0:
            value = float("nan")
        else:
            cutoff = np.percentile(returns, self._percentile)
            worst = returns[returns <= cutoff]
            value = float(np.mean(worst)) if worst.size > 0 else float(cutoff)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "percentile": self._percentile,
                "n_episodes": int(returns.size),
            },
        )
