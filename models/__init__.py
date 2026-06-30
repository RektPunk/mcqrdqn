from .dqn import DQNAgent
from .fqf import FQFAgent
from .iqn import IQNAgent
from .mcfqf import MCFQFAgent
from .mcqrdqn import MCQRDQNAgent
from .qrdqn import QRDQNAgent


def set_agent(model_id: str):
    match model_id:
        case "dqn":
            return DQNAgent
        case "fqf":
            return FQFAgent
        case "iqn":
            return IQNAgent
        case "mcqrdqn":
            return MCQRDQNAgent
        case "mcfqf":
            return MCFQFAgent
        case "qrdqn":
            return QRDQNAgent
        case _:
            raise ValueError(f"Unknown model_id: {model_id}")
