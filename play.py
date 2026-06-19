import argparse
from pathlib import Path

import torch

from common.env import device, load_env
from common.utils import load_config
from models.dqn import DQNet
from models.mcqrdqn import MCQRDQNet
from models.qrdqn import QRDQNet


def replay(args: argparse.Namespace):
    env_id = args.env_id
    model_id = args.model_id
    seed = args.seed
    num_test_episodes = args.num_test_episodes

    env, state_dim, num_actions = load_env(env_id, render_mode="human")

    config = load_config(env_id, model_id)
    num_quantiles = config["num_quantiles"] if "num_quantiles" in config else None
    hidden_dim = config["hidden_dim"]

    match model_id:
        case "dqn":
            model = DQNet(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                num_actions=num_actions,
            ).to(device)
        case "mcqrdqn":
            assert isinstance(num_quantiles, int)
            model = MCQRDQNet(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                num_actions=num_actions,
                num_quantiles=num_quantiles,
            ).to(device)
        case "qrdqn":
            assert isinstance(num_quantiles, int)
            model = QRDQNet(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                num_actions=num_actions,
                num_quantiles=num_quantiles,
            ).to(device)
        case _:
            raise ValueError(f"Unknown model_id: {model_id}")

    model_path = (
        Path("outputs")
        / "weights"
        / f"{model_id}_{env_id.replace('/', '_')}_{seed}.pth"
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path} on {device}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        return

    model.eval()

    for episode in range(num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action = model.action(state_t)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str)
    parser.add_argument("--model-id", type=str, default="mcqrdqn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test-episodes", type=int, default=5)

    args = parser.parse_args()
    replay(args)
