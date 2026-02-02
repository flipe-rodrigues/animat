"""Motor Module - Alpha and Gamma motor neuron outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MotorModule(nn.Module):
    """
    Output module producing motor commands.

    - Alpha MN: direct muscle activations (0-1 via sigmoid)
    - Gamma static: modulates Type II sensitivity (0-2 range)
    - Gamma dynamic: modulates Type Ia sensitivity (0-2 range)
    """

    def __init__(self, input_size: int, num_muscles: int, output_bias: bool = True):
        super().__init__()
        self.num_muscles = num_muscles

        self.alpha_head = nn.Linear(input_size, num_muscles, bias=output_bias)
        self.gamma_static_head = nn.Linear(input_size, num_muscles, bias=output_bias)
        self.gamma_dynamic_head = nn.Linear(input_size, num_muscles, bias=output_bias)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate motor outputs from hidden state.

        Returns:
            alpha_cortical: Cortical contribution to alpha MN [batch, num_muscles]
            gamma_static: Gamma static outputs (0-2) [batch, num_muscles]
            gamma_dynamic: Gamma dynamic outputs (0-2) [batch, num_muscles]
        """
        alpha_cortical = F.relu(self.alpha_head(hidden_state))
        gamma_static = F.relu(self.gamma_static_head(hidden_state)) * 2.0
        gamma_dynamic = F.relu(self.gamma_dynamic_head(hidden_state)) * 2.0

        return alpha_cortical, gamma_static, gamma_dynamic
