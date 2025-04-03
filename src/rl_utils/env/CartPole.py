import gymnasium as gym
from gymnasium import spaces
import pygame

import numpy as np


class CartPoleMotor:
    def __init__(
        self,
        v_max=12.0,
        r=0.5,
        k_t=0.05,
        k_e=0.05,
        gear_radius=0.02,
        efficiency=0.85,
        current_limit=np.inf,
    ):
        self.V_max = v_max
        self.R = r
        self.k_t = k_t
        self.k_e = k_e
        self.gear_radius = gear_radius
        self.efficiency = efficiency
        self.current_limit = current_limit

    def compute_force(self, velocity, u_cmd):
        """
        Compute the force applied to the cart given:
        - velocity: cart linear velocity (m/s)
        - u_cmd: normalized control input [-1, 1]
        """
        # Convert to motor angular velocity
        omega = velocity / self.gear_radius
        V = np.clip(u_cmd, -1.0, 1.0) * self.V_max

        I = (V - self.k_e * omega) / self.R
        I = np.clip(I, -self.current_limit, self.current_limit)

        torque = self.efficiency * self.k_t * I
        force = torque / self.gear_radius
        return force


class CustomCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, dt=0.02):
        self.dt = dt
        self.g = 9.81
        self.mc = 1.0 / 3.0  # cart mass
        self.mp = 1.0 / 3.0  # pole mass
        self.L = 0.2032  # pole length (8 inches in meters)
        self.l = self.L / 2  # distance to center of mass
        self.I = (
            (1 / 3) * self.mp * self.L**2
        )  # moment of inertia of pole (uniform rod)

        # Kinetic friction coefficients (N·s/m and N·m·s/rad)
        self.cart_friction = 0
        self.pole_friction = 0.05

        self.reset()

        self.motor = CartPoleMotor(
            v_max=24.0,
            r=0.6857,
            k_t=0.018779,
            k_e=0.018779,
            gear_radius=0.0075,
            efficiency=0.618,
            current_limit=17.5,
        )

        # Observation/action space
        high = np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.state = None
        self.reset()

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.screen_size = (600, 400)
        self.scale = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(
            [0.0, 0.0, -np.pi, 0.0],
            dtype=np.float32,
        )
        return self.state.copy(), {}

    def dynamics(self, state, u):
        x, x_dot, theta, theta_dot = state
        mc, mp, l, I, g = self.mc, self.mp, self.l, self.I, self.g

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Inertia matrix M(q)
        M = np.array(
            [[mc + mp, mp * l * cos_theta], [mp * l * cos_theta, I + mp * l**2]]
        )

        # Coriolis and gravity vector
        C = np.array([-mp * l * theta_dot**2 * sin_theta, mp * g * l * sin_theta])

        # Kinetic friction (always opposes velocity)
        F_friction = np.array(
            [-self.cart_friction * x_dot, -self.pole_friction * theta_dot]
        )

        # Generalized force vector: Q = [u; 0]
        Q = np.array([u, 0])

        # Total RHS
        rhs = Q + F_friction - C

        # Solve for accelerations: M * q_ddot = rhs
        q_ddot = np.linalg.solve(M, rhs)
        x_ddot, theta_ddot = q_ddot

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float32)

    def get_observation(self):
        return self.state.copy()

    def step(self, action):
        u_cmd = np.clip(action[0], -1.0, 1.0)
        x_dot = self.state[1]
        u = self.motor.compute_force(x_dot, u_cmd)
        self.state = self.rk4(self.state, u)

        terminated = False
        reward = 1.0
        return self.get_observation(), reward, terminated, False, {}

    def rk4(self, state, u):
        dt = self.dt
        f = lambda s: self.dynamics(s, u)

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("CartPole")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))
        cart_y = self.screen_size[1] // 2
        origin = np.array([self.screen_size[0] // 2, cart_y])

        # extract state
        x, _, theta, _ = self.state
        cart_x = int(origin[0] + x * self.scale)
        cart_width = 60
        cart_height = 30

        # Draw cart
        cart_rect = pygame.Rect(
            cart_x - cart_width // 2, cart_y - cart_height // 2, cart_width, cart_height
        )
        pygame.draw.rect(self.window, (0, 0, 0), cart_rect)

        # Draw pole
        pole_length = self.l * self.scale
        pole_x = cart_x + int(pole_length * np.sin(theta))
        pole_y = cart_y - int(pole_length * np.cos(theta))
        pygame.draw.line(
            self.window, (200, 0, 0), (cart_x, cart_y), (pole_x, pole_y), 6
        )

        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


gym.envs.register(id="MyCartPole", entry_point=CustomCartPoleEnv, max_episode_steps=300)
