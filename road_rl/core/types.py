"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Core data structures.

This module defines immutable data containers used to pass results
between evaluation, logging, and metrics layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Episode-level result
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeResult:
    """
    Result of a single evaluation episode.
    """
    episode: int
    seed: int
    epsilon: float

    return_: float
    length: int

    terminated: bool
    truncated: bool

    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Experiment-level result
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentResult:
    """
    Aggregated result of an evaluation experiment.
    """
    env_id: str
    algorithm: str

    attack_name: Optional[str]
    defense_name: Optional[str]

    episodes: List[EpisodeResult]

    metadata: Dict[str, Any] = field(default_factory=dict)
