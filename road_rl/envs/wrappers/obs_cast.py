"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Observation casting wrapper.

Ensures observations are cast to a consistent dtype (default: float32).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class ObsCastWrapper(gym.ObservationWrapper):
    """
    Cast observations to a specified NumPy dtype.
    """

    def __init__(self, env: gym.Env, dtype=np.float32):
        super().__init__(env)
        self._dtype = dtype

        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=self._dtype,
            )

    def observation(self, observation):
        return np.asarray(observation, dtype=self._dtype)
