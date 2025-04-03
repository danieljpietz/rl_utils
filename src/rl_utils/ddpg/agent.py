import torch
from torch import optim, nn
import numpy as np


import gymnasium as gym
import copy as _copy

from .noise import OUNoise
from .replay_buffer import PrioritizedReplayBuffer


class Agent:
    def __init__(
        self,
        env: gym.Env,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        device="cpu",
    ):
        self.env = env
        self.device = torch.device(device)

        self.state_dim = env.observation_space.shape[-1]
        self.action_dim = env.action_space.shape[-1]
        self.max_action = float(env.action_space.high[-1])

        self.actor = actor
        self.actor_target = _copy.deepcopy(actor)

        self.critic = critic
        self.critic_target = _copy.deepcopy(critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Other components
        self.replay_buffer = PrioritizedReplayBuffer()
        self.noise = OUNoise(self.action_dim)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if explore:
            action += self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.batch_size, beta=0.4)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = weights.to(self.device)

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = self.critic_target(torch.cat((next_states, next_actions), dim=1))
            q_target = rewards + (1 - dones) * self.gamma * q_target

        q_val = self.critic(torch.cat((states, actions), dim=1))
        td_errors = q_val - q_target
        critic_loss = (td_errors.pow(2) * weights).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Actor loss
        actor_loss = -self.critic(torch.cat((states, self.actor(states)), dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target updates
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
