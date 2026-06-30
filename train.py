import argparse
from collections import deque

import numpy as np
import torch

from common.buffer import ReplayBuffer
from common.env import load_env
from common.logger import logger
from common.utils import get_epsilon, get_model_path, get_writer, load_config
from models import FQFAgent, MCFQFAgent, set_agent


def train(args: argparse.Namespace):
    env_id = args.env_id
    model_id = args.model_id
    seed = args.seed

    config = load_config(env_id, model_id)

    num_episodes = config["num_episodes"]
    batch_size = config["batch_size"]
    epsilon_decay = config["epsilon_decay"]
    buffer_capacity = config["buffer_capacity"]

    env, state_dim, num_actions = load_env(env_id, seed=seed)

    Agent = set_agent(model_id)
    agent = Agent(
        state_dim=state_dim,
        num_actions=num_actions,
        **config,
    )

    buffer = ReplayBuffer(buffer_capacity)
    writer = get_writer(model_id, env_id, seed)
    model_path = get_model_path(model_id, env_id, seed)

    best_avg_reward = float("-inf")
    reward_window = deque(maxlen=20)

    frame_idx = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward: float = 0.0
        epsilon: float = 0.0
        done = False

        while not done:
            epsilon = get_epsilon(frame_idx=frame_idx, epsilon_decay=epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, float(reward), next_state, done)
            state = next_state
            episode_reward += float(reward)
            frame_idx += 1
            agent.update(buffer, batch_size)

        writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        writer.add_scalar("Train/FrameIdx", frame_idx, episode)
        if episode % 10 == 0:
            logger.info(
                f"# {episode} | reward: {episode_reward:.2f} | epsilon: {epsilon:.2f}"
            )

        reward_window.append(episode_reward)
        current_avg_reward = np.mean(reward_window)
        if len(reward_window) >= 20 and current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            if isinstance(agent, (FQFAgent, MCFQFAgent)):
                checkpoint = {
                    "policy_net": agent.policy_net.state_dict(),
                    "fpnet": agent.fpnet.state_dict(),
                }
                torch.save(checkpoint, model_path)
            else:
                torch.save(agent.policy_net.state_dict(), model_path)
            logger.info(f"# {episode} | New best avg reward {best_avg_reward:.2f}!")

    writer.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="e.g., CartPole-v1",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mcqrdqn",
        help="dqn, qrdqn, mcqrdqn, iqn, fqf, mcfqf",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
