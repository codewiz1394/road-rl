"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Safety metrics.

Safety signals in RL environments are not standardized. This module
provides generic, signal-driven safety metrics that operate on episode
records when such signals are available.

Episode records are expected to be dictionaries that may include:
- "terminated": bool
- "truncated": bool
- "termination_reason": str (optional)
- "violations": int (optional)
- other boolean flags such as "collision", "offroad", "crash" (optional)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Set

import numpy as np

from road_rl.metrics.base import Metric, MetricResult


def _count_bool(episodes: Iterable[Mapping[str, Any]], key: str) -> tuple[int, int]:
    n = 0
    k = 0
    for ep in episodes:
        if key in ep:
            n += 1
            if bool(ep[key]):
                k += 1
    return k, n


def _mean_int(episodes: Iterable[Mapping[str, Any]], key: str) -> tuple[float, int]:
    vals = []
    for ep in episodes:
        if key in ep:
            try:
                vals.append(int(ep[key]))
            except Exception:
                continue
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


class TerminationRates(Metric):
    """
    Termination/truncation rates across all episodes.
    """

    @property
    def name(self) -> str:
        return "termination_rates"

    def compute(self, episodes: Iterable[Mapping[str, Any]]) -> MetricResult:
        episodes = list(episodes)
        n_total = len(episodes)

        term = sum(bool(ep.get("terminated", False)) for ep in episodes)
        trunc = sum(bool(ep.get("truncated", False)) for ep in episodes)

        value = float(term / n_total) if n_total else float("nan")

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "n_episodes": n_total,
                "terminated_rate": float(term / n_total) if n_total else float("nan"),
                "truncated_rate": float(trunc / n_total) if n_total else float("nan"),
            },
        )


class TerminationReasonHistogram(Metric):
    """
    Histogram of termination reasons if present.

    This does not invent reasons; it only aggregates what is logged.
    """

    @property
    def name(self) -> str:
        return "termination_reason_hist"

    def compute(self, episodes: Iterable[Mapping[str, Any]]) -> MetricResult:
        reasons = []
        for ep in episodes:
            r = ep.get("termination_reason", None)
            if r is not None:
                reasons.append(str(r))

        hist = dict(Counter(reasons))

        # Primary scalar value: fraction of episodes with a recorded reason
        n_total = len(list(episodes)) if not isinstance(episodes, list) else len(episodes)
        n_with = sum(hist.values())
        value = float(n_with / n_total) if n_total else float("nan")

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "n_with_reason": n_with,
                "hist": hist,
            },
        )


class ViolationRate(Metric):
    """
    Average number of violations per episode if a 'violations' field exists.
    """

    @property
    def name(self) -> str:
        return "violation_rate"

    def compute(self, episodes: Iterable[Mapping[str, Any]]) -> MetricResult:
        mean_v, n = _mean_int(episodes, "violations")
        return MetricResult(
            name=self.name,
            value=mean_v,
            details={"n_episodes_with_violations": n},
        )


class SafetyFlagRates(Metric):
    """
    Generic boolean safety flag rates.

    Example flags: collision, crash, offroad, lane_violation, etc.
    Only computes rates for keys that actually appear.
    """

    def __init__(self, candidate_flags: Set[str] | None = None):
        self._flags = candidate_flags or {
            "collision", "crash", "offroad", "lane_violation", "near_miss"
        }

    @property
    def name(self) -> str:
        return "safety_flag_rates"

    def compute(self, episodes: Iterable[Mapping[str, Any]]) -> MetricResult:
        episodes = list(episodes)
        n_total = len(episodes)

        rates: Dict[str, float] = {}
        coverage: Dict[str, int] = {}

        for flag in sorted(self._flags):
            k, n = _count_bool(episodes, flag)
            if n > 0:
                rates[flag] = float(k / n)
                coverage[flag] = int(n)

        # Primary scalar value: collision/crash/offroad average if present
        primary_keys = [k for k in ["collision", "crash", "offroad"] if k in rates]
        if primary_keys:
            value = float(np.mean([rates[k] for k in primary_keys]))
        else:
            value = float("nan")

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "n_episodes": n_total,
                "rates": rates,
                "coverage": coverage,
            },
        )
