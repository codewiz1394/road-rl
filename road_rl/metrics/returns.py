"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Return-based metrics.

This module implements standard performance metrics computed from
episodic returns, including aggregation across attack strengths.
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


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

class MeanReturn(Metric):
    """
    Mean episodic return across all epsilons and seeds.
    """

    @property
    def name(self) -> str:
        return "mean_return"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        returns = [float(ep["return"]) for ep in episodes]
        value = float(np.mean(returns)) if returns else float("nan")

        return MetricResult(
            name=self.name,
            value=value,
            details={"n_episodes": len(returns)},
        )


class MedianReturn(Metric):
    """
    Median episodic return across all epsilons and seeds.
    """

    @property
    def name(self) -> str:
        return "median_return"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        returns = [float(ep["return"]) for ep in episodes]
        value = float(np.median(returns)) if returns else float("nan")

        return MetricResult(
            name=self.name,
            value=value,
            details={"n_episodes": len(returns)},
        )


class MeanReturnPerEpsilon(Metric):
    """
    Mean return computed separately for each epsilon.
    """

    @property
    def name(self) -> str:
        return "mean_return_per_epsilon"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        groups = _group_by_epsilon(episodes)

        per_eps = {
            eps: float(np.mean(vals)) for eps, vals in groups.items()
        }

        # Global mean over epsilons (simple average)
        value = float(np.mean(list(per_eps.values()))) if per_eps else float("nan")

        return MetricResult(
            name=self.name,
            value=value,
            details={"per_epsilon": per_eps},
        )


class NormalizedReturnDrop(Metric):
    """
    Normalized performance drop relative to epsilon=0 baseline.

    Drop is defined as:
        (R_0 - R_eps) / |R_0|
    """

    @property
    def name(self) -> str:
        return "normalized_return_drop"

    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        groups = _group_by_epsilon(episodes)

        if 0.0 not in groups:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                details={"error": "No epsilon=0 baseline found"},
            )

        baseline = np.mean(groups[0.0])
        drops: Dict[float, float] = {}

        for eps, vals in groups.items():
            mean_eps = np.mean(vals)
            drops[eps] = float((baseline - mean_eps) / (abs(baseline) + 1e-8))

        # Average drop over attacked epsilons (excluding eps=0)
        attacked_drops = [v for k, v in drops.items() if k != 0.0]
        value = float(np.mean(attacked_drops)) if attacked_drops else 0.0

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "baseline": float(baseline),
                "per_epsilon_drop": drops,
            },
        )
