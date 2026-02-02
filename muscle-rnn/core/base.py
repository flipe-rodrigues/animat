"""Base classes and interfaces for extensibility."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class BaseController(nn.Module, ABC):
    """Abstract base class for all controllers."""

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

        Returns:
            action: Alpha MN activations [batch, num_muscles]
            new_hidden: Updated hidden state (None for MLPs)
            info: Dictionary of intermediate outputs
        """
        pass

    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as flat numpy array (for CMA-ES)."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from flat numpy array (for CMA-ES)."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(flat_params[idx : idx + size].reshape(p.shape)))
            idx += size

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
