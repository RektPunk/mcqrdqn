import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from common.buffer import ReplayBuffer
from common.env import device
from common.loss import quantile_huber_loss


class L1PMLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        num_quantiles: int,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        # Parameter for quantile regression
        self.delta_bias = nn.Parameter(
            torch.randn(num_actions, 1, num_quantiles) * 0.05
        )
        self.delta_coef = nn.Parameter(
            torch.randn(num_actions, input_dim, num_quantiles) * 0.05
        )
        self.log_scale = nn.Parameter(torch.zeros(num_actions, 1))

        # Placeholder to store the L1 penalty during the forward pass
        self.register_buffer("_penalty", torch.tensor(0.0))

    def forward(self, x: torch.Tensor, tau: torch.Tensor | None) -> torch.Tensor:
        # Input shapes: (batch_size, input_dim)
        # return shape: (batch_size, num_actions, num_quantiles)

        # Concatenate intercepts and weights along the input/hidden dimension
        delta = torch.cat(
            [self.delta_bias, self.delta_coef], dim=1
        )  # (num_actions, input_dim + 1, num_quantiles)

        # Accumulate the quantile changes to obtain the actual weights and intercepts
        coef = torch.cumsum(delta, dim=2)  # (num_actions, input_dim + 1, num_quantiles)

        # Slice out the weight and intercept for constraint validation
        delta_coef = delta[:, 1:, 1:]  # (num_actions, input_dim, num_quantiles - 1)
        delta_bias = delta[:, 0:1, 1:]  # (num_actions, 1, num_quantiles - 1)

        # Sum the negative weight across all input features for each quantile
        delta_minus_sum = torch.sum(
            torch.clamp(-delta_coef, min=0.0), dim=1
        )  # (num_actions, num_quantiles - 1)

        # Clip intercept
        delta_bias_clipped = torch.clamp(
            delta_bias, min=6.0 * delta_minus_sum.unsqueeze(1)
        )  # (num_actions, 1, num_quantiles - 1)

        # Exponential mapping of scale: scale shape
        scale = torch.exp(self.log_scale).unsqueeze(0)  # (1, num_actions, 1)

        # Vectorized batched matrix multiplication over all actions
        # Inputs:
        # - x: (batch_size, input_dim) -> labeled 'bi'
        # - coef[:, 1:, :]: (num_actions, input_dim, num_quantiles) -> labeled 'aiq'
        # Output:
        # - term1: (batch_size, num_actions, num_quantiles) -> labeled 'baq'
        term1 = torch.einsum("bi,aiq->baq", x, coef[:, 1:, :])
        if self.training:
            # Store the total L1 penalty across all actions internally
            self._penalty = torch.mean(
                torch.abs(delta_bias - delta_bias_clipped), dim=2
            ).sum()
            grid = (term1 + coef[:, 0, :].unsqueeze(0)) * scale
        else:
            # Cumsum of the clipped variations to reconstruct the constrained intercepts
            term2 = torch.cumsum(
                torch.cat([coef[:, 0:1, 0:1], delta_bias_clipped], dim=2),
                dim=2,
            )  # (num_actions, 1, num_quantiles)
            term2 = term2.transpose(0, 1)  # (1, num_actions, num_quantiles)
            grid = (term1 + term2) * scale

        if tau is None:
            return grid

        scaled_tau = tau * (self.num_quantiles - 1)
        idx_low = torch.clamp(scaled_tau.long(), 0, self.num_quantiles - 2)
        idx_high = idx_low + 1

        weight_high = scaled_tau - idx_low.float()
        weight_low = 1.0 - weight_high

        idx_low_expanded = idx_low.unsqueeze(1).expand(-1, self.num_actions, -1)
        idx_high_expanded = idx_high.unsqueeze(1).expand(-1, self.num_actions, -1)

        q_low = grid.gather(2, idx_low_expanded)
        q_high = grid.gather(2, idx_high_expanded)

        return weight_low.unsqueeze(1) * q_low + weight_high.unsqueeze(1) * q_high

    @property
    def penalty(self) -> torch.Tensor:
        return self._penalty


class MCQNet(nn.Module):
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

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
        )
        self.output_head = L1PMLinear(hidden_dim, num_actions, num_quantiles)

    def forward(self, x: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.output_head(features, tau)

    def get_penalty(self) -> torch.Tensor:
        return self.output_head.penalty


class MCQRDQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_quantiles: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        l1_penalty_weight: float = 1.0,
        **kwargs,
    ):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.tau = tau
        self.l1_penalty_weight = l1_penalty_weight
        self.policy_net = MCQNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)
        self.target_net = MCQNet(
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

        with torch.no_grad():
            dist = self.policy_net(state_t)
            q_values = dist.mean(dim=2)
            action = q_values.argmax().item()

        return int(action)

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
        loss += self.l1_penalty_weight * self.policy_net.get_penalty()

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
