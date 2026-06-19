import random

import torch
import torch.nn as nn
import torch.optim as optim

from common.env import device


class DQNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
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
        hidden_dim: int,
        num_actions: int,
        lr: float,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        self.policy_net = DQNet(state_dim, hidden_dim, num_actions).to(device)
        self.target_net = DQNet(state_dim, hidden_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.policy_net.eval()
        return self.policy_net.action(state_t)

    def update(self, buffer, batch_size: int) -> tuple[float, float | None] | None:
        if len(buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        rewards_t = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        self.policy_net.train()

        current_q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_t).max(
                dim=1,
                keepdim=True,
            )[0]
            target_q_values = rewards_t + self.gamma * (1 - dones_t) * max_next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

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

        return loss.item(), None
