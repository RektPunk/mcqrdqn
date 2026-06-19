import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
from gymnasium.wrappers import FlattenObservation


def load_env(
    env_id: str,
    seed: int = 42,
    render_mode: str | None = None,
) -> tuple[gym.Env, int, int]:
    env = gym.make(env_id, render_mode=render_mode)
    if isinstance(seed, int):
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    obs_shape = env.observation_space.shape
    assert obs_shape is not None

    state_dim = obs_shape[0]
    assert isinstance(env.action_space, Discrete), "Action space must be Discrete"

    num_actions = int(env.action_space.n)

    return FlattenObservation(env), state_dim, num_actions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
