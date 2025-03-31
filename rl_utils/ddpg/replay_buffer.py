import torch
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, size=1_000_000, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.max_size = size
        self.alpha = alpha
        self.pos = 0

    def add(self, transition, td_error=1.0):
        priority = float(abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** -beta
        weights = weights / max(weights)

        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            torch.FloatTensor(weights).unsqueeze(1),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(td_err) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)
