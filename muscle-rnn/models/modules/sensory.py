"""
Sensory Module - Proprioceptive sensory neurons

Implements biologically-inspired sensory neurons:
- Type Ia: respond to muscle velocity (dynamic spindle afferents)
- Type II: respond to muscle length (static spindle afferents)  
- Type Ib: respond to muscle force (Golgi tendon organs)

Gamma motor neurons modulate spindle sensitivity:
- Gamma static: modulates Type II response to length
- Gamma dynamic: modulates Type Ia response to velocity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SensoryModule(nn.Module):
    """
    Proprioceptive sensory neuron module.

    Each sensory type is a single ReLU unit per muscle (no hidden layers).
    Gamma motor neurons modulate spindle sensitivity before processing.
    """

    def __init__(
        self,
        num_muscles: int,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_muscles = num_muscles
        self.use_bias = use_bias

        # Sensory neuron layers (one per muscle, per type) - single layer ReLU
        # Type Ia: velocity -> ReLU (1 input, 1 output per muscle)
        self.type_Ia = nn.ModuleList(
            [nn.Linear(1, 1, bias=use_bias) for _ in range(num_muscles)]
        )

        # Type II: length -> ReLU (1 input, 1 output per muscle)
        self.type_II = nn.ModuleList(
            [nn.Linear(1, 1, bias=use_bias) for _ in range(num_muscles)]
        )

        # Type Ib: force -> ReLU (Golgi tendon organ, 1 input, 1 output per muscle)
        self.type_Ib = nn.ModuleList(
            [nn.Linear(1, 1, bias=use_bias) for _ in range(num_muscles)]
        )

        # Initialize with biologically plausible weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with biologically plausible values."""
        for i in range(self.num_muscles):
            # Type Ia: positive response to velocity
            nn.init.uniform_(self.type_Ia[i].weight, 0.5, 1.5)
            if self.type_Ia[i].bias is not None:
                nn.init.zeros_(self.type_Ia[i].bias)

            # Type II: positive response to length
            nn.init.uniform_(self.type_II[i].weight, 0.5, 1.5)
            if self.type_II[i].bias is not None:
                nn.init.zeros_(self.type_II[i].bias)

            # Type Ib: positive response to force
            nn.init.uniform_(self.type_Ib[i].weight, 0.5, 1.5)
            if self.type_Ib[i].bias is not None:
                nn.init.zeros_(self.type_Ib[i].bias)

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
            lengths: Muscle lengths [batch, num_muscles]
            velocities: Muscle velocities [batch, num_muscles]
            forces: Muscle forces [batch, num_muscles]
            gamma_static: Gamma static modulation [batch, num_muscles]
            gamma_dynamic: Gamma dynamic modulation [batch, num_muscles]

        Returns:
            concatenated: All sensory outputs concatenated [batch, num_muscles * 3]
            sensory_outputs: Dictionary of individual sensory outputs
        """
        batch_size = lengths.shape[0]

        if gamma_static is None:
            gamma_static = torch.ones_like(lengths)
        if gamma_dynamic is None:
            gamma_dynamic = torch.ones_like(velocities)

        type_Ia_outputs = []
        type_II_outputs = []
        type_Ib_outputs = []

        for i in range(self.num_muscles):
            # Get inputs for this muscle
            length_i = lengths[:, i : i + 1]
            velocity_i = velocities[:, i : i + 1]
            force_i = forces[:, i : i + 1]

            # Apply gamma modulation before sensory processing
            modulated_length = length_i + gamma_static[:, i : i + 1]
            modulated_velocity = velocity_i * gamma_dynamic[:, i : i + 1]

            # Type Ia: responds to velocity only (modulated by gamma dynamic)
            Ia_output = F.relu(self.type_Ia[i](modulated_velocity))
            type_Ia_outputs.append(Ia_output)

            # Type II: responds to length only (modulated by gamma static)
            II_output = F.relu(self.type_II[i](modulated_length))
            type_II_outputs.append(II_output)

            # Type Ib: responds to force (not modulated by gamma)
            Ib_output = F.relu(self.type_Ib[i](force_i))
            type_Ib_outputs.append(Ib_output)

        # Concatenate all sensory outputs
        type_Ia_all = torch.cat(type_Ia_outputs, dim=1)  # [batch, num_muscles]
        type_II_all = torch.cat(type_II_outputs, dim=1)  # [batch, num_muscles]
        type_Ib_all = torch.cat(type_Ib_outputs, dim=1)  # [batch, num_muscles]

        # Concatenate for output (no hidden layer)
        concatenated = torch.cat([type_Ia_all, type_II_all, type_Ib_all], dim=1)

        sensory_outputs = {
            "type_Ia": type_Ia_all,
            "type_II": type_II_all,
            "type_Ib": type_Ib_all,
        }

        return concatenated, sensory_outputs
