import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from common.logger import logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(
    env_id: str,
    model_id: str,
) -> dict[str, Any]:
    config_path = Path("configs") / f"{model_id}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path) as f:
        config = json.load(f)

    if env_id not in config:
        raise ValueError(f"Environment ID '{env_id}' not found in '{config_path}'.")
    return config[env_id]


def get_writer(model_id: str, env_id: str, seed: int) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = (
        Path("outputs")
        / "logs"
        / f"{model_id}_{env_id.replace('/', '_')}_seed{seed}_{timestamp}"
    )
    logger.info(f"TensorBoard logging initialized at: {log_dir}")
    return SummaryWriter(log_dir=str(log_dir))


def get_model_path(model_id: str, env_id: str, seed: int) -> Path:
    model_dir = Path("outputs") / "weights" / f"{env_id.replace('/', '_')}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}_{seed}.pth"
    return model_path


def get_epsilon(
    frame_idx: int,
    epsilon_decay: float,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
) -> float:
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )
