"""
Unified configuration classes for the muscle-RNN project.
"""

from dataclasses import dataclass, field
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
    """
    Configuration for neural network architecture.
    
    All dimension attributes use consistent naming: num_* prefix.
    """
    
    # Core dimensions (required)
    num_muscles: int
    num_sensors: int  # Total sensor inputs (length + velocity + force per muscle)
    num_target_units: int  # Grid size^2 for target encoding
    
    # Target encoding parameters
    target_grid_size: int = DEFAULT_TARGET_GRID_SIZE
    target_sigma: float = DEFAULT_TARGET_SIGMA
    workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    
    # RNN architecture
    rnn_hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE
    rnn_type: str = "rnn"  # 'rnn', 'gru', 'lstm'
    num_rnn_layers: int = 1
    
    # MLP architecture (for teacher networks)
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_MLP_HIDDEN_SIZES))
    
    # Bias settings for each module
    proprioceptive_bias: bool = True
    target_encoding_bias: bool = True
    input_projection_bias: bool = True
    output_bias: bool = True
    reflex_bias: bool = False
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure num_target_units matches grid size
        if self.num_target_units != self.target_grid_size ** 2:
            self.num_target_units = self.target_grid_size ** 2
        
        # Validate dimensions
        if self.num_muscles <= 0:
            raise ValueError(f"num_muscles must be positive, got {self.num_muscles}")
        if self.num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {self.num_sensors}")
    
    @property
    def input_size(self) -> int:
        """Total observation dimension (sensors + raw target XYZ)."""
        return self.num_sensors + 3  # 3 for XYZ position
    
    @property
    def action_size(self) -> int:
        """Action dimension (alpha motor neurons only)."""
        return self.num_muscles
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'num_muscles': self.num_muscles,
            'num_sensors': self.num_sensors,
            'num_target_units': self.num_target_units,
            'target_grid_size': self.target_grid_size,
            'target_sigma': self.target_sigma,
            'workspace_bounds': self.workspace_bounds,
            'rnn_hidden_size': self.rnn_hidden_size,
            'rnn_type': self.rnn_type,
            'num_rnn_layers': self.num_rnn_layers,
            'mlp_hidden_sizes': self.mlp_hidden_sizes,
            'proprioceptive_bias': self.proprioceptive_bias,
            'target_encoding_bias': self.target_encoding_bias,
            'input_projection_bias': self.input_projection_bias,
            'output_bias': self.output_bias,
            'reflex_bias': self.reflex_bias,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingConfig:
    """
    Base configuration for training experiments.
    
    Shared by all training methods (CMA-ES, distillation, etc.).
    """
    
    # Paths
    xml_path: str
    output_dir: str
    
    # Training parameters
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    
    # Checkpointing
    save_checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not Path(self.xml_path).exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'xml_path': self.xml_path,
            'output_dir': self.output_dir,
            'max_episode_steps': self.max_episode_steps,
            'num_eval_episodes': self.num_eval_episodes,
            'save_checkpoint_every': self.save_checkpoint_every,
            'seed': self.seed,
        }
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
