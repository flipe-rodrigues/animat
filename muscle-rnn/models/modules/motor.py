"""
Motor Module - Alpha and Gamma motor neuron outputs

Produces motor commands from controller hidden state:
- Alpha motor neurons: direct muscle activations (0-1 range)
- Gamma static: modulates Type II (length) sensory responses (0-2 range)
- Gamma dynamic: modulates Type Ia (velocity) sensory responses (0-2 range)

Note: The stretch reflex (Ia/II -> Alpha) is now handled at the controller level,
not in this module. This keeps the motor module decoupled from sensory processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MotorModule(nn.Module):
    """
    Output module producing motor commands.

    Single ReLU layer for each output type (no hidden layers).
    Reflex pathways are handled externally by the controller.
    """

    def __init__(
        self,
        input_size: int,
        num_muscles: int,
        output_bias: bool = True,
    ):
        super().__init__()
        self.num_muscles = num_muscles
        self.input_size = input_size

        # Separate single-layer output heads (ReLU applied in forward)
        self.alpha_head = nn.Linear(input_size, num_muscles, bias=output_bias)
        self.gamma_static_head = nn.Linear(input_size, num_muscles, bias=output_bias)
        self.gamma_dynamic_head = nn.Linear(input_size, num_muscles, bias=output_bias)

    def forward(
        self, 
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate motor outputs from hidden state.

        Args:
            hidden_state: RNN/MLP hidden state [batch, input_size]

        Returns:
            alpha_cortical: Cortical contribution to alpha MN [batch, num_muscles]
            gamma_static: Gamma static outputs [batch, num_muscles] (0-2 range)
            gamma_dynamic: Gamma dynamic outputs [batch, num_muscles] (0-2 range)
        """
        # Generate outputs with ReLU (single layer each)
        alpha_cortical = F.relu(self.alpha_head(hidden_state))
        gamma_static = F.relu(self.gamma_static_head(hidden_state)) * 2.0  # Scale to 0-2 range
        gamma_dynamic = F.relu(self.gamma_dynamic_head(hidden_state)) * 2.0

        return alpha_cortical, gamma_static, gamma_dynamic
