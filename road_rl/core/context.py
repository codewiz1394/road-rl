"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Execution context definitions.

This module defines immutable context objects that carry metadata
throughout evaluation and attack/defense pipelines. Contexts are
explicitly passed rather than relying on hidden global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Step-level context
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class StepContext:
    """
    Context available at each environment step.

    This context is passed to attacks and defenses and provides
    all relevant metadata needed for reproducible, transparent
    perturbation logic.
    """
    step: int
    episode: int
    seed: int
    epsilon: float
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Episode-level context
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeContext:
    """
    Context describing a single episode.

    EpisodeContext is useful for higher-level evaluation, logging,
    and reproducibility tracking.
    """
    episode: int
    seed: int
    epsilon: float
    env_id: Optional[str] = None
    algorithm: Optional[str] = None
