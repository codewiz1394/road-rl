"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Highway-Env adapter.

This module provides a consistent interface to environments from
the highway-env package. It standardizes observation format and
configuration while preserving original environment dynamics.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError(
        "gymnasium is required for highway environments."
    ) from e

try:
    import highway_env  # noqa: F401
except ImportError as e:
    raise ImportError(
        "highway-env is not installed. Install with:\n"
        "  pip install highway-env"
    ) from e


def _default_highway_config() -> dict:
    """
    Default configuration for Highway environments.

    These settings are chosen to:
    - use kinematic observations (vector-based, low-dimensional)
    - avoid image observations (simpler, attack-friendly)
    - keep rewards interpretable
    """
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,
        "policy_frequency": 1,
        "simulation_frequency": 15,
    }


class HighwayAdapter(gym.Wrapper):
    """
    Adapter wrapper for highway-env environments.

    Ensures observations are returned as np.float32 arrays and
    hides highway-specific internals from the rest of RoAd-RL.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            obs, info = self.env.reset(seed=int(seed), options=options)
        else:
            obs, info = self.env.reset(options=options)

        return self._process_obs(obs), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), float(reward), terminated, truncated, info

    @staticmethod
    def _process_obs(obs: Any) -> Any:
        """
        Convert observation to a float32 numpy array.

        Highway kinematic observations are already arrays,
        but this enforces dtype consistency.
        """
        return np.asarray(obs, dtype=np.float32)


def make_highway_env(
    env_id: str,
    *,
    render_mode: Optional[str] = None,
    config: Optional[dict] = None,
) -> gym.Env:
    """
    Create and configure a Highway-Env environment.

    Parameters
    ----------
    env_id:
        Highway environment ID (e.g., "highway-v0", "merge-v0").

    render_mode:
        Optional render mode ("human", "rgb_array").

    config:
        Optional config dict overriding defaults.

    Returns
    -------
    env:
        Adapted highway environment.
    """
    env = gym.make(env_id, render_mode=render_mode)

    # Apply configuration
    cfg = _default_highway_config()
    if config is not None:
        cfg.update(config)

    try:
        env.unwrapped.configure(cfg)
    except Exception as e:
        raise RuntimeError(
            "Failed to configure highway environment. "
            "Check config keys and highway-env version."
        ) from e

    env.reset()
    env = HighwayAdapter(env)
    return env
