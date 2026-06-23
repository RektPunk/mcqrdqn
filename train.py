import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer
from common.env import load_env
from common.logger import logger
from common.utils import load_config
from models.dqn import DQNAgent
from models.mcqrdqn import MCQRDQNAgent
from models.qrdqn import QRDQNAgent


def train(args: argparse.Namespace):
    env_id = args.env_id
    model_id = args.model_id
    seed = args.seed

    config = load_config(env_id, model_id)

    num_episodes = config["num_episodes"]
    num_quantiles = config.get("num_quantiles", None)
    hidden_dim = config["hidden_dim"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    epsilon_decay = config["epsilon_decay"]
    target_reward = config["target_reward"]

    buffer_capacity = config["buffer_capacity"]

    success_threshold = 10

    env, state_dim, num_actions = load_env(env_id, seed=seed)

    match model_id:
        case "dqn":
            agent = DQNAgent(
                state_dim=state_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
                lr=lr,
            )
        case "qrdqn":
            assert isinstance(num_quantiles, int)
            agent = QRDQNAgent(
                state_dim=state_dim,
                num_actions=num_actions,
                num_quantiles=num_quantiles,
                hidden_dim=hidden_dim,
                lr=lr,
            )
        case "mcqrdqn":
            assert isinstance(num_quantiles, int)
            agent = MCQRDQNAgent(
                state_dim=state_dim,
                num_actions=num_actions,
                num_quantiles=num_quantiles,
                hidden_dim=hidden_dim,
                lr=lr,
            )
        case _:
            raise ValueError(f"Unknown model_id: {model_id}")

    buffer = ReplayBuffer(buffer_capacity)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = (
        Path("outputs")
        / "logs"
        / f"{model_id}_{env_id.replace('/', '_')}_seed{seed}_{timestamp}"
    )
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging initialized at: {log_dir}")

    epsilon_start = 1.0
    epsilon_final = 0.05

    def get_epsilon(frame_idx: int) -> float:
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1.0 * frame_idx / epsilon_decay
        )

    frame_idx = 0
    consecutive_success = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward: float = 0.0
        epsilon: float = 0.0
        done = False
        episode_crossings = []
        loss_vals = []

        while not done:
            epsilon = get_epsilon(frame_idx)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, float(reward), next_state, done)
            state = next_state
            episode_reward += float(reward)
            frame_idx += 1

            result = agent.update(buffer, batch_size)
            if result is not None:
                loss_val, cross_val = result
                loss_vals.append(loss_val)
                if cross_val is not None:
                    episode_crossings.append(cross_val)

        avg_crossing_rate = np.mean(episode_crossings) if episode_crossings else 0.0
        avg_loss = np.mean(loss_vals) if loss_vals else 0.0

        writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
        writer.add_scalar("Train/AvgLoss", avg_loss, episode)
        writer.add_scalar("Train/AvgCrossingRate", avg_crossing_rate, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        writer.add_scalar("Train/TotalFrames", frame_idx, episode)

        if episode % 10 == 0:
            logger.info(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}"
            )

        if target_reward and episode_reward >= target_reward:
            consecutive_success += 1
            if consecutive_success >= success_threshold:
                logger.info("Early stopping: Achieved target reward.")
                break
        else:
            consecutive_success = 0

    model_dir = Path("outputs") / "weights" / f"{env_id.replace('/', '_')}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}_{seed}.pth"

    torch.save(agent.policy_net.state_dict(), model_path)
    logger.info(f"Training finished. Model saved to {model_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True, help="e.g., CartPole-v1")
    parser.add_argument(
        "--model-id", type=str, default="mcqrdqn", help="dqn, qrdqn, mcqrdqn"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
