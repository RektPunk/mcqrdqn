import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from common.buffer import ReplayBuffer
from common.env import device
from common.loss import quantile_huber_loss


class QRDQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_quantiles: int,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return out.view(-1, self.num_actions, self.num_quantiles)

    def action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            quantiles = self.forward(state)
            q_values = quantiles.mean(dim=2)
            action = q_values.argmax().item()
        return int(action)


class QRDQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_quantiles: int,
        lr: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        **kwargs,
    ):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.tau = tau
        self.policy_net = QRDQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)
        self.target_net = QRDQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.quantiles = torch.linspace(
            1 / (2 * num_quantiles),
            1 - 1 / (2 * num_quantiles),
            num_quantiles,
            device=device,
        ).view(1, -1)

    def select_action(self, state: NDArray[np.float32], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.policy_net.eval()
        return self.policy_net.action(state_t)

    def update(self, buffer: ReplayBuffer, batch_size: int):
        if len(buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        actions = actions.view(-1, 1, 1).expand(-1, -1, self.num_quantiles)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        self.policy_net.train()

        curr_dist = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_target_dist = self.target_net(next_states)
            next_target_q = next_target_dist.mean(dim=2)
            best_actions = (
                next_target_q.argmax(dim=1)
                .view(-1, 1, 1)
                .expand(-1, -1, self.num_quantiles)
            )
            best_next_dist = next_target_dist.gather(1, best_actions).squeeze(1)
            target_dist = rewards + self.gamma * (1 - dones) * best_next_dist

        loss = quantile_huber_loss(curr_dist, target_dist, self.quantiles)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters(),
                strict=True,
            ):
                target_param.copy_(
                    self.tau * policy_param + (1.0 - self.tau) * target_param
                )
