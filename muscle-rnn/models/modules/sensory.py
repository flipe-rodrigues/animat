"""
Sensory Module - Proprioceptive sensory neurons

Implements biologically-inspired sensory neurons:
- Type Ia: velocity-sensitive (dynamic spindle afferents)
- Type II: length-sensitive (static spindle afferents)
- Type Ib: force-sensitive (Golgi tendon organs)

Gamma motor neurons modulate spindle sensitivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SensoryModule(nn.Module):
    """
    Proprioceptive sensory neuron module using vectorized operations.

    Processes all muscles simultaneously for efficiency.
    """

    def __init__(self, num_muscles: int, use_bias: bool = True):
        super().__init__()
        self.num_muscles = num_muscles

        # Single layer per sensory type - processes all muscles at once
        self.type_Ia = nn.Linear(num_muscles, num_muscles, bias=use_bias)
        self.type_II = nn.Linear(num_muscles, num_muscles, bias=use_bias)
        self.type_Ib = nn.Linear(num_muscles, num_muscles, bias=use_bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with diagonal-dominant structure (each muscle mainly affects itself)."""
        for layer in [self.type_Ia, self.type_II, self.type_Ib]:
            # Diagonal structure with small off-diagonal noise
            nn.init.eye_(layer.weight)
            layer.weight.data *= 0.8
            layer.weight.data += torch.randn_like(layer.weight.data) * 0.1
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        lengths: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        gamma_static: Optional[torch.Tensor] = None,
        gamma_dynamic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process proprioceptive inputs through sensory neurons.

        Args:
            lengths: [batch, num_muscles]
            velocities: [batch, num_muscles]
            forces: [batch, num_muscles]
            gamma_static: [batch, num_muscles] modulation for Type II
            gamma_dynamic: [batch, num_muscles] modulation for Type Ia

        Returns:
            concatenated: [batch, num_muscles * 3]
            sensory_outputs: dict with individual outputs
        """
        if gamma_static is None:
            gamma_static = torch.ones_like(lengths)
        if gamma_dynamic is None:
            gamma_dynamic = torch.ones_like(velocities)

        # Apply gamma modulation
        modulated_length = lengths + gamma_static
        modulated_velocity = velocities * gamma_dynamic

        # Process through sensory layers (vectorized)
        type_Ia_out = F.relu(self.type_Ia(modulated_velocity))
        type_II_out = F.relu(self.type_II(modulated_length))
        type_Ib_out = F.relu(self.type_Ib(forces))

        concatenated = torch.cat([type_Ia_out, type_II_out, type_Ib_out], dim=-1)

        return concatenated, {
            "type_Ia": type_Ia_out,
            "type_II": type_II_out,
            "type_Ib": type_Ib_out,
        }
