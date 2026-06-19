import random

import torch
import torch.nn as nn
import torch.optim as optim

from common.env import device
from common.loss import quantile_huber_loss


class QRDQNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_actions: int, num_quantiles: int
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
        # Output shape: (batch_size, num_actions, num_quantiles)
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
        hidden_dim: int,
        num_actions: int,
        num_quantiles: int,
        lr: float,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.tau = tau

        self.policy_net = QRDQNet(state_dim, hidden_dim, num_actions, num_quantiles).to(
            device
        )
        self.target_net = QRDQNet(state_dim, hidden_dim, num_actions, num_quantiles).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Cumulative probabilities for the quantile Huber loss
        self.quantiles = torch.linspace(
            1 / (2 * num_quantiles),
            1 - 1 / (2 * num_quantiles),
            num_quantiles,
            device=device,
        ).view(1, -1)

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

    def update(self, buffer, batch_size: int):
        if len(buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = (
            torch.as_tensor(actions, dtype=torch.long, device=device)
            .view(-1, 1, 1)
            .expand(-1, -1, self.num_quantiles)
        )
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(
            1
        )

        self.policy_net.train()
        all_quantiles = self.policy_net(states_t)
        with torch.no_grad():
            diff = all_quantiles[..., 1:] - all_quantiles[..., :-1]
            crossing_rate = (diff < 0).float().mean().item()

        current_quantiles = all_quantiles.gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_quantiles_target = self.target_net(next_states_t)
            next_q_values = next_quantiles_target.mean(dim=2)
            best_actions = (
                next_q_values.argmax(dim=1)
                .view(-1, 1, 1)
                .expand(-1, -1, self.num_quantiles)
            )
            best_next_quantiles = next_quantiles_target.gather(1, best_actions).squeeze(
                1
            )

            target_quantiles = (
                rewards_t + self.gamma * (1 - dones_t) * best_next_quantiles
            )

        loss = quantile_huber_loss(current_quantiles, target_quantiles, self.quantiles)

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

        return loss.item(), crossing_rate
