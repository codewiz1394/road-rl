"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Environment factory.

This module provides a single, consistent entry point for creating
evaluation environments. A centralized env factory is critical for
reproducibility: it ensures wrappers, seeding, and observation formats
are applied uniformly across experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError(
        "gymnasium is required for RoAd-RL environments. "
        "Install with: pip install gymnasium"
    ) from e


@dataclass(frozen=True)
class EnvSpec:
    """
    Configuration for environment creation.

    Keep this small and explicit. Anything that affects evaluation
    results should be controlled here.
    """
    env_id: str
    render_mode: Optional[str] = None
    max_episode_steps: Optional[int] = None
    # Optional adapter hint: "gym", "highway", "atari"
    adapter: str = "gym"


def make_env(
    spec: EnvSpec,
    *,
    seed: Optional[int] = None,
) -> Any:
    """
    Create a fresh environment instance based on EnvSpec.

    Parameters
    ----------
    spec:
        Env specification describing which environment to create.

    seed:
        Optional seed passed to env.reset(seed=...).

    Returns
    -------
    env:
        Gymnasium-compatible environment.
    """
    if spec.adapter == "gym":
        env = gym.make(spec.env_id, render_mode=spec.render_mode)

    elif spec.adapter == "highway":
        from road_rl.envs.highway_adapter import make_highway_env
        env = make_highway_env(spec.env_id, render_mode=spec.render_mode)

    elif spec.adapter == "atari":
        from road_rl.envs.atari_adapter import make_atari_env
        env = make_atari_env(spec.env_id, render_mode=spec.render_mode)

    else:
        raise ValueError(
            f"Unknown adapter '{spec.adapter}'. Use one of: gym, highway, atari."
        )

    # Enforce max episode steps if requested (Gymnasium TimeLimit wrapper)
    if spec.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(spec.max_episode_steps))

    # Seed deterministically if provided
    if seed is not None:
        try:
            env.reset(seed=int(seed))
        except TypeError:
            # Some envs may not support seed in reset; still return env
            pass

    return env


def make_env_factory(
    spec: EnvSpec,
) -> Callable[[], Any]:
    """
    Return a zero-argument factory that constructs a fresh env each call.
    This is the preferred way to pass environments into sweep_runner.
    """
    def _factory() -> Any:
        return make_env(spec)
    return _factory
