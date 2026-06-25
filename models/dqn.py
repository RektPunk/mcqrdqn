import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from common.buffer import ReplayBuffer
from common.env import device


class DQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.argmax().item()

        return int(action)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int,
        lr: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        **kwargs,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        self.policy_net = DQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            **kwargs,
        ).to(device)
        self.target_net = DQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            **kwargs,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

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
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        self.policy_net.train()

        curr_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1 - dones) * next_q

        loss = self.loss_fn(curr_q, target_q)

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
