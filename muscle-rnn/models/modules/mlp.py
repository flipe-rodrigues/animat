"""MLP Core Module - Feedforward network (no recurrence)."""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPCore(nn.Module):
    """Feedforward MLP core (no recurrence). Used for teacher networks in distillation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """No-op for interface compatibility with RNN."""
        pass
