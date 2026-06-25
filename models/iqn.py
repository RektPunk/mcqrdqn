import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from common.buffer import ReplayBuffer
from common.env import device
from common.loss import quantile_huber_loss


class IQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_cosines: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.state_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cosine_net = nn.Sequential(
            nn.Linear(num_cosines, hidden_dim),
            nn.ReLU(),
        )
        self.merge_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.cosine_basis: torch.Tensor
        self.register_buffer(
            "cosine_basis",
            torch.arange(
                1,
                num_cosines + 1,
                dtype=torch.float32,
            ),
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_net(x)
        cos_features = torch.cos(
            tau.unsqueeze(-1) * self.cosine_basis.view(1, 1, -1) * math.pi
        )
        tau_emb = self.cosine_net(cos_features)
        combined = state_emb.unsqueeze(1) * tau_emb
        out = self.merge_net(combined)

        return out.permute(0, 2, 1)

    def action(self, state: torch.Tensor, num_quantiles: int) -> int:
        with torch.no_grad():
            batch_size = state.size(0)
            tau = torch.rand(batch_size, num_quantiles, device=state.device)
            dist = self.forward(state, tau)
            q_values = dist.mean(dim=2)
            action = q_values.argmax().item()

        return int(action)


class IQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_quantiles: int,
        num_cosines: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        **kwargs,
    ):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.tau = tau

        self.policy_net = IQNet(
            state_dim,
            hidden_dim,
            num_actions,
            num_cosines,
            **kwargs,
        ).to(device)
        self.target_net = IQNet(
            state_dim,
            hidden_dim,
            num_actions,
            num_cosines,
            **kwargs,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: NDArray[np.float32], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.policy_net.eval()
        return self.policy_net.action(state_t, self.num_quantiles)

    def update(self, buffer: ReplayBuffer, batch_size: int):
        if len(buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        actions = actions.view(-1, 1, 1).expand(-1, -1, self.num_quantiles)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        tau = torch.rand(batch_size, self.num_quantiles, device=device)
        tau_next = torch.rand(batch_size, self.num_quantiles, device=device)

        self.policy_net.train()

        curr_dist = self.policy_net(states, tau).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_policy_dist = self.policy_net(next_states, tau_next)
            next_policy_q = next_policy_dist.mean(dim=2)
            best_actions = (
                next_policy_q.argmax(dim=1)
                .view(-1, 1, 1)
                .expand(-1, -1, self.num_quantiles)
            )
            next_target_dist = self.target_net(next_states, tau_next)
            best_next_dist = next_target_dist.gather(1, best_actions).squeeze(1)
            target_dist = rewards + self.gamma * (1 - dones) * best_next_dist

        loss = quantile_huber_loss(curr_dist, target_dist, tau)

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
