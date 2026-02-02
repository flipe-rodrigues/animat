"""
MLP Core Module - Feedforward network (no recurrence)

Used for teacher networks in distillation learning.
Has larger hidden layers to compensate for lack of memory.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPCore(nn.Module):
    """
    Feedforward MLP core (no recurrence).
    
    Uses multiple hidden layers to compensate for lack of temporal memory.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        self.hidden_sizes = hidden_sizes

        # Build MLP layers
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final projection to output size
        layers.append(nn.Linear(prev_dim, output_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_size]

        Returns:
            output: MLP output [batch, output_size]
        """
        return self.mlp(x)

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """No-op for interface compatibility with RNN."""
        pass
