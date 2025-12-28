"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Gymnasium environment adapter.

This module provides lightweight normalization and consistency
wrappers for standard Gymnasium environments. The goal is not to
change environment dynamics, but to ensure stable observation
types and predictable behavior during evaluation.
"""

from __future__ import annotations

from typing import Any, Tuple, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError(
        "gymnasium is required for RoAd-RL. Install with: pip install gymnasium"
    ) from e


class Float32ObservationWrapper(gym.ObservationWrapper):
    """
    Ensure observations are returned as np.float32.

    This avoids silent dtype mismatches when computing gradients
    or logging numerical results.
    """

    def observation(self, observation: Any) -> Any:
        return np.asarray(observation, dtype=np.float32)


class GymAdapter(gym.Wrapper):
    """
    Base adapter for Gymnasium environments.

    This wrapper:
    - enforces float32 observations
    - exposes observation_space and action_space unchanged
    - does not alter rewards, termination, or truncation
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Wrap observation dtype
        self.env = Float32ObservationWrapper(env)

        # Expose spaces explicitly for clarity
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        """
        Reset environment with optional seed.
        """
        if seed is not None:
            obs, info = self.env.reset(seed=int(seed), options=options)
        else:
            obs, info = self.env.reset(options=options)

        return obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, dict]:
        """
        Step the environment.

        Action is passed through unchanged. This allows both
        discrete and continuous control policies.
        """
        return self.env.step(action)


def make_gym_env(
    env_id: str,
    *,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Create and adapt a Gymnasium environment.

    This function is intentionally minimal. All environment
    creation for standard Gym envs should go through here.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = GymAdapter(env)
    return env
