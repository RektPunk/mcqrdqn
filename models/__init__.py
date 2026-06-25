# from .dqn import DQNAgent, DQNet
from .fqf import FPNet, FQFAgent, FQFNet
from .iqn import IQNAgent, IQNet
from .mcqrdqn import MCQRDQNAgent, MCQRDQNet
from .qrdqn import QRDQNAgent, QRDQNet


def set_agent(model_id: str):
    match model_id:
        # case "dqn":
        #     return DQNAgent
        case "fqf":
            return FQFAgent
        case "iqn":
            return IQNAgent
        case "mcqrdqn":
            return MCQRDQNAgent
        case "qrdqn":
            return QRDQNAgent
        case _:
            raise ValueError(f"Unknown model_id: {model_id}")


def set_model(model_id: str):
    match model_id:
        # case "dqn":
        #     return DQNet
        case "fqf":
            return FQFNet, FPNet
        case "iqn":
            return IQNet
        case "mcqrdqn":
            return MCQRDQNet
        case "qrdqn":
            return QRDQNet
        case _:
            raise ValueError(f"Unknown model_id: {model_id}")
