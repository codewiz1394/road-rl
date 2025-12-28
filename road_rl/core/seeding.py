"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Seeding utilities.

This module centralizes all randomness control used throughout
RoAd-RL to ensure reproducible evaluation and training runs.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_everything(
    seed: int,
    *,
    deterministic_torch: bool = True,
) -> None:
    """
    Seed Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed:
        Global random seed.

    deterministic_torch:
        If True, configure PyTorch for deterministic behavior
        (may reduce performance).
    """
    seed = int(seed)

    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch (optional dependency)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch not installed; ignore silently
        pass


def seed_env(
    env,
    seed: int,
) -> None:
    """
    Seed a Gymnasium-compatible environment.

    This function tries to seed both the environment and its
    action/observation spaces when supported.
    """
    seed = int(seed)

    try:
        env.reset(seed=seed)
    except Exception:
        pass

    # Some envs expose action_space / observation_space seeding
    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    try:
        env.observation_space.seed(seed)
    except Exception:
        pass
