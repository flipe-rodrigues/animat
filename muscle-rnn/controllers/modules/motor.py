"""Motor output modules."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import Activation


class MotorModule(nn.Module):
    """
    Generates motor commands from descending signals and sensory feedback.

    Implements:
    - Alpha motor neuron activation (drives muscles)
    - Gamma motor neuron activation (modulates spindle sensitivity)
    - Spinal reflex pathways (Ia and II to alpha)

    Maintains internal state for alpha and gamma activations.
    """

    def __init__(
        self,
        num_muscles: int,
        core_output_size: int,
        activation: Activation = Activation.SIGMOID,
        use_bias: bool = False,
    ):
        super().__init__()
        self.num_muscles = num_muscles
        self.activation_fn = activation.to_function()

        # Spinal reflex pathways
        self.Ia_to_alpha = nn.Linear(num_muscles, num_muscles, bias=False)
        self.II_to_alpha = nn.Linear(num_muscles, num_muscles, bias=False)

        # Descending commands from core
        self.core_to_alpha = nn.Linear(core_output_size, num_muscles, bias=use_bias)
        self.core_to_gamma_static = nn.Linear(core_output_size, num_muscles, bias=use_bias)
        self.core_to_gamma_dynamic = nn.Linear(core_output_size, num_muscles, bias=use_bias)

        # Internal state
        self._alpha: Optional[torch.Tensor] = None
        self._gamma_static: Optional[torch.Tensor] = None
        self._gamma_dynamic: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset all motor state."""
        device = device or torch.device("cpu")
        self._alpha = torch.zeros(batch_size, self.num_muscles, device=device)
        self._gamma_static = torch.zeros(batch_size, self.num_muscles, device=device)
        self._gamma_dynamic = torch.zeros(batch_size, self.num_muscles, device=device)

    @property
    def alpha(self) -> Optional[torch.Tensor]:
        """Last computed alpha activation (detached from graph)."""
        return self._alpha

    @property
    def gamma_static(self) -> Optional[torch.Tensor]:
        """Current static gamma drive."""
        return self._gamma_static

    @property
    def gamma_dynamic(self) -> Optional[torch.Tensor]:
        """Current dynamic gamma drive."""
        return self._gamma_dynamic

    def forward(
        self,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        core: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute motor outputs.

        Returns (alpha, gamma_static, gamma_dynamic).
        """
        alpha = self.activation_fn(
            self.Ia_to_alpha(spindle_Ia)
            + self.II_to_alpha(spindle_II)
            + self.core_to_alpha(core)
        )
        gamma_static = self.activation_fn(self.core_to_gamma_static(core))
        gamma_dynamic = self.activation_fn(self.core_to_gamma_dynamic(core))

        # Update state
        self._alpha = alpha.detach()
        self._gamma_static = gamma_static.detach()
        self._gamma_dynamic = gamma_dynamic.detach()

        return alpha, gamma_static, gamma_dynamic
