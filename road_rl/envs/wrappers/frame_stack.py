"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Frame stacking wrapper.

Stacks the last K observations along the channel dimension.
"""

from __future__ import annotations

from collections import deque
import numpy as np
import gymnasium as gym


class FrameStackWrapper(gym.Wrapper):
    """
    Stack the last `num_frames` observations.
    """

    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        self.num_frames = int(num_frames)
        self.frames = deque(maxlen=self.num_frames)

        assert isinstance(env.observation_space, gym.spaces.Box), \
            "FrameStackWrapper only supports Box observation spaces."

        low = np.repeat(env.observation_space.low, self.num_frames, axis=0)
        high = np.repeat(env.observation_space.high, self.num_frames, axis=0)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)
