from typing import Any
import gymnasium as gym
import torch
from torch.autograd.functional import jacobian
import numpy as np
from ..solvers import rk4


class LinearCartPoleEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.dt = 0.01

        self.m_cart = 1.0
        self.f_cart = 0.1

        self.l_pole = 1.0
        self.m_pole = 0.1
        self.f_pole = 0.0

        self.g = 9.81  # Gravity

        # New state order: [x, theta, x_dot, theta_dot]
        self._state = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rand_range = torch.tensor([10, np.pi / 4, 0, 0])
        self._state = 2 * torch.rand_like(self._state) * rand_range - rand_range
        return self._state.cpu().numpy(), {}

    def get_observation(self):
        return self._state.detach().cpu().numpy(), {}

    def _dynamics(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        state:  [4] tensor -> [x, theta, x_dot, theta_dot]
        control: [1] tensor -> [u]
        returns: [4] tensor -> time derivative of state
        """

        x, theta, x_dot, theta_dot = state
        u = control[0]

        m_c = self.m_cart
        m_p = self.m_pole
        l = self.l_pole
        f_c = self.f_cart
        f_p = self.f_pole
        g = self.g

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # Mass matrix
        M = torch.stack(
            [
                torch.stack([torch.tensor(m_c + m_p), m_p * l * cos_theta]),
                torch.stack([m_p * l * cos_theta, torch.tensor(m_p * l**2)]),
            ]
        )

        # Force terms
        C = torch.stack(
            [
                -m_p * l * theta_dot**2 * sin_theta - f_c * x_dot + u,
                -m_p * g * l * sin_theta - f_p * theta_dot,
            ]
        )

        # Solve for accelerations
        acc = torch.linalg.solve(M, C)  # [xddot, thetaddot]

        # Return state derivative in reordered state layout
        dxdt = torch.stack([x_dot, theta_dot, acc[0], acc[1]])
        return dxdt

    def linearize(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor([0.0], requires_grad=True)
        A = jacobian(lambda state: self._dynamics(state, u), x)
        B = jacobian(lambda action: self._dynamics(x, action), u)
        return A.detach().cpu().numpy(), B.detach().cpu().numpy()

    def step(self, action):
        # Not implemented yet
        self._state = rk4(lambda s: self._dynamics(s, action), self._state, self.dt)
        return self.get_observation()
