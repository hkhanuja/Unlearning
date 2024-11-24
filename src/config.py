from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    model_path: str = "chavinlo/alpaca-native"
    num_assistant_layers: int = 8
    load_in_8bit: bool = False
    device: str = "cuda"
    
@dataclass
class LoRAConfig:
    r: int = 32
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

@dataclass
class TrainingConfig:
    learning_rate: float = 5e-4
    num_epochs: int = 15
    batch_size: int = 4
    retain_weight: float = 6.5
    max_grad_norm: float = 1.0
    max_length: int = 512
    device: str = "cuda"
    warmup_steps: int = 100  # Add warmup
    weight_decay: float = 0.01  # Add some regularization