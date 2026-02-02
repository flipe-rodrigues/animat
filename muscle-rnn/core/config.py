"""Configuration classes for the muscle-RNN project."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple, List
import json
from pathlib import Path

from .constants import (
    DEFAULT_RNN_HIDDEN_SIZE,
    DEFAULT_MLP_HIDDEN_SIZES,
    DEFAULT_TARGET_GRID_SIZE,
    DEFAULT_TARGET_SIGMA,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_NUM_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_EVERY,
)


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""

    # Core dimensions
    num_muscles: int
    num_sensors: int
    num_target_units: int = DEFAULT_TARGET_GRID_SIZE ** 2

    # Target encoding
    target_grid_size: int = DEFAULT_TARGET_GRID_SIZE
    target_sigma: float = DEFAULT_TARGET_SIGMA
    workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # RNN architecture
    rnn_hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE
    rnn_type: str = "rnn"
    num_rnn_layers: int = 1

    # MLP architecture
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_MLP_HIDDEN_SIZES))

    # Bias settings
    proprioceptive_bias: bool = True
    target_encoding_bias: bool = True
    input_projection_bias: bool = True
    output_bias: bool = True
    reflex_bias: bool = False

    def __post_init__(self):
        self.num_target_units = self.target_grid_size ** 2
        if self.num_muscles <= 0:
            raise ValueError(f"num_muscles must be positive, got {self.num_muscles}")
        if self.num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {self.num_sensors}")

    @property
    def input_size(self) -> int:
        """Total observation dimension (sensors + raw target XYZ)."""
        return self.num_sensors + 3

    @property
    def action_size(self) -> int:
        return self.num_muscles

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingConfig:
    """Base configuration for training experiments."""

    xml_path: str
    output_dir: str
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    save_checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    seed: Optional[int] = None

    def __post_init__(self):
        if not Path(self.xml_path).exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
