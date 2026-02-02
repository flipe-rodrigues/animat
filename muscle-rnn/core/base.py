"""
Base classes and interfaces for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from core.constants import DEFAULT_CALIBRATION_EPISODES


class BaseController(nn.Module, ABC):
    """
    Abstract base class for all controllers.
    
    Defines the interface that all controllers must implement.
    """
    
    @abstractmethod
    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Initialize hidden state for new episodes."""
        pass
    
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        prev_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for single timestep.
        
        Args:
            obs: Observation [batch, obs_dim]
            hidden: Optional hidden state
            prev_alpha: Previous alpha MN output
            
        Returns:
            action: Alpha MN activations [batch, num_muscles]
            new_hidden: Updated hidden state (None for MLPs)
            info: Dictionary of intermediate outputs
        """
        pass
    
    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as a flat numpy array (for CMA-ES)."""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat numpy array (for CMA-ES)."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat_params[idx : idx + size].reshape(p.shape),
                dtype=p.dtype,
                device=p.device,
            )
            idx += size
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseTrainer(ABC):
    """
    Abstract base class for training algorithms.
    
    Defines the interface for training procedures.
    """
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Run the training procedure.
        
        Returns:
            Dictionary with training results and metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = DEFAULT_CALIBRATION_EPISODES) -> Dict[str, Any]:
        """Evaluate current controller."""
        pass
