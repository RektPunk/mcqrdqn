import random

import torch
import torch.nn as nn
import torch.optim as optim

from common.env import device
from common.loss import quantile_huber_loss


class L1PMLinear(nn.Module):
    def __init__(self, input_dim: int, num_quantiles: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_quantiles = num_quantiles
        self.delta_coef = nn.Parameter(torch.randn(input_dim, num_quantiles) * 0.05)
        self.delta_bias = nn.Parameter(torch.randn(1, num_quantiles) * 0.05)
        self.log_scale = nn.Parameter(torch.zeros(1))
        # Placeholder to dynamically store the L1 penalty during the forward pass
        self.register_buffer("_penalty", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate intercepts and weights: (h + 1, r)
        delta = torch.cat([self.delta_bias, self.delta_coef], dim=0)
        coef = torch.cumsum(delta, dim=1)

        # Slice out the weight and intercept variations for constraint validation
        delta_coef = delta[1:, 1:]  # (h, r - 1)
        delta_bias = delta[0:1, 1:]  # (1, r - 1)

        # Sum the negative weight across all hidden neurons (h) for each quantile
        delta_minus_sum = torch.sum(torch.clamp(-delta_coef, min=0.0), dim=0)

        # Clip intercept to be greater than or equal to the sum of negative weights
        # Since the hidden layer outputs feature representation x in [0, 6],
        # the bound guarantees that predicted quantiles never cross (Monotone).
        delta_bias_clipped = torch.clamp(delta_bias, min=6.0 * delta_minus_sum)

        scale = torch.exp(self.log_scale)
        if self.training:
            # Store the L1 penalty internally to pass it up to the optimizer
            self._penalty = torch.mean(torch.abs(delta_bias - delta_bias_clipped))

            # Return unconstrained prediction for smoother training landscape
            return (torch.matmul(x, coef[1:, :]) + coef[0, :]) * scale
        else:
            term1 = torch.matmul(x, coef[1:, :])
            term2 = torch.cumsum(
                torch.cat([coef[0:1, 0:1], delta_bias_clipped], dim=1),
                dim=1,
            )
            return (term1 + term2) * scale

    @property
    def penalty(self) -> torch.Tensor:
        return self._penalty


class MCQRDQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_quantiles: int,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
        )

        self.output_heads = nn.ModuleList(
            [L1PMLinear(hidden_dim, num_quantiles) for _ in range(num_actions)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch_size, num_actions, num_quantiles)
        features = self.feature_extractor(x)
        quantiles = torch.stack([head(features) for head in self.output_heads], dim=1)
        return quantiles

    def get_penalty(self) -> torch.Tensor:
        return torch.stack(
            [head.penalty for head in self.output_heads if isinstance(head, L1PMLinear)]
        ).sum()

    def action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            quantiles = self.forward(state)
            q_values = quantiles.mean(dim=2)
            action = q_values.argmax().item()

        return int(action)


class MCQRDQNAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_quantiles: int,
        lr: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        l1_penalty_weight: float = 5.0,
    ):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.tau = tau
        self.l1_penalty_weight = l1_penalty_weight
        self.policy_net = MCQRDQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
        ).to(device)
        self.target_net = MCQRDQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
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

    def select_action(self, state, epsilon: float):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.policy_net.eval()
        return self.policy_net.action(state_t)

    def update(self, buffer, batch_size: int) -> tuple[float, float] | None:
        if len(buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = (
            torch.as_tensor(actions, dtype=torch.long, device=device)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, -1, self.num_quantiles)
        )
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

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
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, -1, self.num_quantiles)
            )
            best_next_quantiles = next_quantiles_target.gather(
                1,
                best_actions,
            ).squeeze(1)
            target_quantiles = (
                rewards_t + self.gamma * (1 - dones_t) * best_next_quantiles
            )

        loss = quantile_huber_loss(current_quantiles, target_quantiles, self.quantiles)
        mcqr_penalty = self.policy_net.get_penalty()
        total_loss = loss + self.l1_penalty_weight * mcqr_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
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

        return total_loss.item(), crossing_rate
