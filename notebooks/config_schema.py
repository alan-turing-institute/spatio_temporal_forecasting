"""Structured configuration schemas for PhysicsNEMO training with Hydra."""

from dataclasses import dataclass
from typing import Tuple
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """FNO model configuration."""

    name: str
    n_vars: int
    n_modes: Tuple[int, int]
    hidden_channels: int
    n_layers: int
    use_skip_connections: bool
    t_in: int
    t_out: int


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int
    learning_rate: float
    lr_decay: float


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str
    n_steps_input: int
    n_steps_output: int
    batch_size: int
    n_channels: int


@dataclass
class Config:
    """Main configuration."""

    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    device: str
    output_path: str


def register_configs() -> None:
    """Register structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    cs.store(group="model", name="fno_schema", node=ModelConfig)
    cs.store(group="training", name="default_schema", node=TrainingConfig)
