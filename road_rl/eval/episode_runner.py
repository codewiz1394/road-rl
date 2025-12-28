"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Episode execution logic.

This module defines the canonical procedure for running a single
episode under a fixed attack/defense configuration. All evaluation
results in RoAd-RL are derived from this execution path.
"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np

from road_rl.core.context import StepContext
from road_rl.core.types import EpisodeResult
from road_rl.policies.base import Policy
from road_rl.attacks.base import Attack
from road_rl.defenses.base import Defense


def run_episode(
    env: Any,
    policy: Policy,
    *,
    attack: Optional[Attack] = None,
    defense: Optional[Defense] = None,
    epsilon: float = 0.0,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> EpisodeResult:
    """
    Run a single episode in the environment.

    Parameters
    ----------
    env:
        Environment instance following the Gymnasium API.

    policy:
        Policy used to select actions.

    attack:
        Optional adversarial attack applied to observations.

    defense:
        Optional defense applied after the attack.

    epsilon:
        Perturbation budget used by the attack.

    seed:
        Optional random seed for reproducibility.

    max_steps:
        Optional hard limit on the number of steps.

    Returns
    -------
    EpisodeResult
        Summary of the episode outcome.
    """
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    obs, info = env.reset()
    done = False
    terminated = False
    truncated = False

    total_return = 0.0
    step_idx = 0

    while not done:
        if max_steps is not None and step_idx >= max_steps:
            truncated = True
            break

        # Construct step context
        ctx = StepContext(
            step=step_idx,
            episode=0,  # filled by sweep runner if needed
            seed=seed if seed is not None else -1,
            epsilon=epsilon,
            info=info if info is not None else {},
        )

        obs_in = obs

        # Apply attack if present
        if attack is not None:
            obs_in = attack.apply(
                obs_in,
                policy,
                ctx,
            )

        # Apply defense if present
        if defense is not None:
            obs_in = defense.apply(
                obs_in,
                ctx,
            )

        # Policy action
        action = policy.act(obs_in)

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_return += float(reward)
        step_idx += 1

    return EpisodeResult(
        episode=0,  # filled by sweep runner
        seed=seed if seed is not None else -1,
        epsilon=epsilon,
        total_return=total_return,
        length=step_idx,
        terminated=terminated,
        truncated=truncated,
        info={},
    )
