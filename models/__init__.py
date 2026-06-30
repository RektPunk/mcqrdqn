from .dqn import DQNAgent
from .mcqrdqn import MCQRDQNAgent
from .qrdqn import QRDQNAgent


def set_agent(model_id: str):
    match model_id:
        case "dqn":
            return DQNAgent
        case "mcqrdqn":
            return MCQRDQNAgent
        case "qrdqn":
            return QRDQNAgent
        case _:
            raise ValueError(f"Unknown model_id: {model_id}")
