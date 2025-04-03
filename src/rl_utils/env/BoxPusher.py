from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class BoxPusher(gym.Env):
    def __init__(self):
        self.max_speed = 10
        self.max_torque = 2.0
        self.dt = 0.05
        self.mu = 0.1
        self.m = 1.0

        self.state = np.array([0.0, 0.0])

        high = np.array([10.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        x, v = self.state
        mu = self.mu
        m = self.m
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        costs = x**2 + 0.1 * v**2 + 0.001 * (u**2)

        a = (u / m) - mu * v
        a = np.clip(a, -self.max_speed, self.max_speed)

        new_v = v + (a * dt)
        new_x = x + (v * dt)

        if new_x <= -10:
            new_x = -10
            new_v = 0
        elif new_x >= 10:
            new_x = 10
            new_v = 0

        self.state = np.array([new_x, new_v])

        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        high = np.array([10, 0])

        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        return self._get_obs(), {}

    def _get_obs(self):
        x, v = self.state
        return np.array([x, v], dtype=np.float32)


gym.envs.register(id="BoxPusher-v0", entry_point=BoxPusher, max_episode_steps=300)
