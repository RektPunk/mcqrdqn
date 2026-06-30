import argparse

import numpy as np
import torch

from common.env import load_env
from common.logger import logger
from common.utils import get_model_path, load_config
from models import FQFAgent, MCFQFAgent, set_agent


def play(args: argparse.Namespace):
    env_id = args.env_id
    model_id = args.model_id
    seed = args.seed
    num_episodes = args.num_episodes

    config = load_config(env_id, model_id)
    env, state_dim, num_actions = load_env(env_id, seed=seed, render_mode="human")

    Agent = set_agent(model_id)
    agent = Agent(
        state_dim=state_dim,
        num_actions=num_actions,
        **config,
    )

    model_path = get_model_path(model_id, env_id, seed)
    logger.info(f"Loading model from {model_path}...")

    try:
        checkpoint = torch.load(model_path)

        if isinstance(agent, (FQFAgent, MCFQFAgent)):
            agent.policy_net.load_state_dict(checkpoint["policy_net"])
            agent.fpnet.load_state_dict(checkpoint["fpnet"])
            agent.policy_net.eval()
            agent.fpnet.eval()
        else:
            agent.policy_net.load_state_dict(checkpoint)
            agent.policy_net.eval()
        logger.info("Model loaded successfully!")
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Please train the model first.")
        return
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.select_action(state, epsilon=0.0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += float(reward)

        total_rewards.append(episode_reward)
        logger.info(f"# {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f}")

    logger.info(f"Finished {num_episodes} test episodes.")
    logger.info(f"Stats: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")

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
        help="dqn, qrdqn, mcqrdqn, fqf, mcfqf",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=5)

    args = parser.parse_args()
    play(args)
