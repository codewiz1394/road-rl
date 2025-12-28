"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Metric base definitions.

This module defines the common interface and data structures used
by all evaluation metrics in RoAd-RL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


# ---------------------------------------------------------------------
# Metric result container
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MetricResult:
    """
    Output of a metric computation.

    Attributes
    ----------
    name:
        Metric identifier (e.g. "mean_return", "cvar_return").

    value:
        Primary scalar value of the metric.

    details:
        Optional structured details (e.g. per-epsilon values).
    """
    name: str
    value: float
    details: Mapping[str, Any]


# ---------------------------------------------------------------------
# Metric interface
# ---------------------------------------------------------------------

class Metric(ABC):
    """
    Abstract base class for all metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique metric name.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        episodes: Iterable[Mapping[str, Any]],
    ) -> MetricResult:
        """
        Compute metric from episode records.

        Parameters
        ----------
        episodes:
            Iterable of episode dictionaries, typically loaded
            from episodes.csv.

        Returns
        -------
        MetricResult
        """
        raise NotImplementedError
