"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Normalize-and-clip defense.

This defense applies simple, deterministic preprocessing to observations:
1) Optional normalization
2) Clipping to valid observation bounds

It serves as a lightweight inference-time baseline defense and is intentionally
simple and transparent.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from road_rl.core.types import StepContext
from road_rl.defenses.base import Defense


class NormalizeClipDefense(Defense):
    """
    Normalize-and-clip observation defense.

    Parameters
    ----------
    mean:
        Optional mean used for normalization.

    std:
        Optional standard deviation used for normalization.

    clip_low:
        Optional lower bound for clipping.

    clip_high:
        Optional upper bound for clipping.
    """

    def __init__(
        self,
        *,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        clip_low: Optional[Any] = None,
        clip_high: Optional[Any] = None,
    ):
        self.mean = mean
        self.std = std
        self.clip_low = clip_low
        self.clip_high = clip_high

    def apply(
        self,
        observation: Any,
        context: StepContext,
    ) -> Any:
        """
        Apply normalization and clipping to the observation.
        """
        # Convert to NumPy for lightweight preprocessing
        obs = np.asarray(observation, dtype=np.float32)

        # Normalize if statistics are provided
        if self.mean is not None and self.std is not None:
            obs = (obs - self.mean) / (self.std + 1e-8)

        # Clip to bounds if provided
        if self.clip_low is not None or self.clip_high is not None:
            low = self.clip_low if self.clip_low is not None else -np.inf
            high = self.clip_high if self.clip_high is not None else np.inf
            obs = np.clip(obs, low, high)

        # Return same type as input where possible
        if isinstance(observation, torch.Tensor):
            return torch.as_tensor(obs, dtype=observation.dtype)

        return obs
