"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Observation normalization wrapper.

Applies running mean / standard deviation normalization to observations.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's algorithm.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.mean.shape) else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count


class ObsNormalizeWrapper(gym.ObservationWrapper):
    """
    Normalize observations using running mean and std.
    """

    def __init__(self, env: gym.Env, clip_range: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        self._clip = clip_range
        self._eps = epsilon

        assert isinstance(env.observation_space, gym.spaces.Box), \
            "ObsNormalizeWrapper only supports Box observation spaces."

        self._rms = RunningMeanStd(env.observation_space.shape)

    def observation(self, observation):
        self._rms.update(observation)
        obs = (observation - self._rms.mean) / np.sqrt(self._rms.var + self._eps)
        return np.clip(obs, -self._clip, self._clip)
