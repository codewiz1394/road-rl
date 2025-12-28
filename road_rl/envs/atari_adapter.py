"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Atari (ALE) adapter.

This module provides standard Atari preprocessing wrappers suitable for
benchmarking. It uses Gymnasium wrappers where available and falls back
to a minimal, explicit preprocessing pipeline.

Expected dependencies:
- gymnasium[atari] or gymnasium[accept-rom-license]
- ale-py

Install example:
  pip install "gymnasium[atari,accept-rom-license]"
"""

from __future__ import annotations

from typing import Optional, Any, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("gymnasium is required for Atari environments.") from e

# Check ALE availability early for clearer errors
try:
    import ale_py  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ale-py is not installed. Install with:\n"
        '  pip install "gymnasium[atari,accept-rom-license]"'
    ) from e


def _is_atari_env_id(env_id: str) -> bool:
    # Gymnasium Atari IDs typically look like: "ALE/Pong-v5"
    return env_id.startswith("ALE/")


def make_atari_env(
    env_id: str,
    *,
    render_mode: Optional[str] = None,
    frameskip: int = 4,
    noop_max: int = 30,
    terminal_on_life_loss: bool = False,
    grayscale: bool = True,
    resize_shape: Tuple[int, int] = (84, 84),
    frame_stack: int = 4,
) -> gym.Env:
    """
    Create and wrap an Atari environment with standard preprocessing.

    Parameters
    ----------
    env_id:
        Atari environment ID, typically "ALE/<Game>-v5".

    render_mode:
        Optional render mode.

    frameskip:
        Number of frames to skip per step (standard is 4).

    noop_max:
        Max no-op actions on reset.

    terminal_on_life_loss:
        If True, losing a life ends an episode (common for training).
        For evaluation, many setups keep this False.

    grayscale:
        Whether to convert frames to grayscale.

    resize_shape:
        Output frame size (default 84x84).

    frame_stack:
        Number of frames to stack.

    Returns
    -------
    env:
        Gymnasium environment with Atari preprocessing applied.
    """
    if not _is_atari_env_id(env_id):
        raise ValueError(
            f"Unsupported Atari env_id '{env_id}'. Expected Gymnasium ALE IDs like 'ALE/Pong-v5'."
        )

    # Create base env (frameskip handled by wrapper below)
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)

    # --- Standard Gymnasium Atari wrappers ---
    # These wrappers exist in gymnasium.wrappers.atari_preprocessing in modern Gymnasium.
    try:
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    except Exception as e:
        raise ImportError(
            "Gymnasium AtariPreprocessing wrapper not available. "
            'Install with: pip install "gymnasium[atari,accept-rom-license]"'
        ) from e

    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frameskip,
        screen_size=resize_shape[0],
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=grayscale,
        grayscale_newaxis=False,  # we keep (84,84) not (84,84,1)
        scale_obs=False,          # keep uint8 values; policies can scale explicitly
    )

    # Frame stack (channel-last stacking). For grayscale: (84,84,stack)
    if frame_stack and frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=int(frame_stack))

    return env


def atari_obs_to_float32(obs: Any, scale: bool = True) -> np.ndarray:
    """
    Convert Atari observation to float32.

    Parameters
    ----------
    obs:
        Observation from env (may be LazyFrames / stacked).

    scale:
        If True, scale uint8 pixels to [0, 1].

    Returns
    -------
    np.ndarray:
        Float32 observation array.
    """
    arr = np.asarray(obs, dtype=np.float32)
    if scale:
        arr /= 255.0
    return arr
