import random
from collections import deque

import numpy as np
import torch

from common.env import device


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size),
            strict=True,
        )

        return (
            torch.as_tensor(np.array(state), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(action), dtype=torch.long, device=device),
            torch.as_tensor(np.array(reward), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(done), dtype=torch.uint8, device=device),
        )

    def __len__(self):
        return len(self.buffer)
