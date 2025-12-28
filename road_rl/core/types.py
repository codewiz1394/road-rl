"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Core data structures shared across the entire framework.
These types define what information flows between environments,
policies, attacks, defenses, and evaluation code.

The goal of this module is stability: these definitions should
rarely change, even as the library grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepContext:
    """
    Context available at a single environment step.

    This object is passed to attacks and defenses so they can
    make decisions based on time, epsilon, or environment feedback.
    """
    step: int
    episode: int
    seed: int
    epsilon: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """
    Result of executing a single episode under a fixed
    attack/defense configuration.
    """
    episode: int
    seed: int
    epsilon: float
    total_return: float
    length: int
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """
    Aggregated results from an evaluation sweep across
    multiple episodes and epsilon values.
    """
    env_id: str
    algorithm: str
    attack_name: Optional[str]
    defense_name: Optional[str]
    epsilons: List[float]
    episode_results: List[EpisodeResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
