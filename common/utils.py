import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


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
) -> dict:
    config_path = Path("configs") / f"{model_id}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path) as f:
        config = json.load(f)

    if env_id not in config:
        raise ValueError(f"Environment ID '{env_id}' not found in '{config_path}'.")
    return config[env_id]


def init_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(name)s:%(levelname)s] %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


logger = init_logger()
