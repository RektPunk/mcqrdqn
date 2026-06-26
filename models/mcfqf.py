import random

import numpy as np
import torch
import torch.optim as optim
from numpy.typing import NDArray

from common.buffer import ReplayBuffer
from common.env import device
from common.loss import weighted_quantile_huber_loss
from models.fqf import FPNet
from models.mcqrdqn import MCQNet


class MCFQFAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_quantiles: int,
        lr: float = 1e-4,
        lr_fpn: float = 1e-5,
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
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)

        self.target_net = MCQNet(
            input_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)

        self.fpnet = FPNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)
        self.target_fpnet = FPNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_quantiles=num_quantiles,
            **kwargs,
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_fpnet.load_state_dict(self.fpnet.state_dict())
        self.target_net.eval()
        self.target_fpnet.eval()

        self.optimizer_val = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_fpn = optim.Adam(self.fpnet.parameters(), lr=lr_fpn)

    def select_action(self, state: NDArray[np.float32], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.fpnet.eval()
        self.policy_net.eval()

        with torch.no_grad():
            _, taus_hat, tau_deltas = self.fpnet(state_t)
            dist = self.policy_net(state_t, taus_hat)
            q_values = (dist * tau_deltas.unsqueeze(1)).sum(dim=2)
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
        self.fpnet.train()

        taus, taus_hat, tau_deltas = self.fpnet(states)
        curr_dist = (
            self.policy_net(states, taus_hat.detach()).gather(1, actions).squeeze(1)
        )
        with torch.no_grad():
            next_taus, next_taus_hat, next_tau_deltas = self.target_fpnet(next_states)
            next_policy_dist = self.policy_net(next_states, next_taus_hat)
            next_policy_q = (next_policy_dist * next_tau_deltas.unsqueeze(1)).sum(dim=2)
            best_actions = (
                next_policy_q.argmax(dim=1)
                .view(-1, 1, 1)
                .expand(-1, -1, self.num_quantiles)
            )
            next_target_dist = self.target_net(next_states, next_taus_hat)
            best_next_dist = next_target_dist.gather(1, best_actions).squeeze(1)
            target_dist = rewards + self.gamma * (1 - dones) * best_next_dist

        val_loss = weighted_quantile_huber_loss(
            curr_dist,
            target_dist,
            taus_hat.detach(),
            tau_deltas.detach(),
        )
        val_loss += self.l1_penalty_weight * self.policy_net.get_penalty()

        self.optimizer_val.zero_grad()
        val_loss.backward()
        with torch.no_grad():
            boundary_dist = self.policy_net(states, taus[:, 1:-1])
            actions_dist = actions[:, :, :-1]
            q_at_taus = boundary_dist.gather(1, actions_dist).squeeze(1)
            fpn_grads = 2 * q_at_taus - curr_dist[:, :-1] - curr_dist[:, 1:]
        fpn_loss = (fpn_grads * taus[:, 1:-1]).sum(dim=1).mean()
        self.optimizer_val.step()

        self.optimizer_fpn.zero_grad()
        fpn_loss.backward()
        self.optimizer_fpn.step()

        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters(),
                strict=True,
            ):
                target_param.copy_(
                    self.tau * policy_param + (1.0 - self.tau) * target_param
                )

            for target_param, policy_param in zip(
                self.target_fpnet.parameters(),
                self.fpnet.parameters(),
                strict=True,
            ):
                target_param.copy_(
                    self.tau * policy_param + (1.0 - self.tau) * target_param
                )
