"""Sensory processing modules."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import Activation


class SensoryModule(nn.Module):
    """
    Simulates proprioceptive sensory processing.

    Models:
    - Muscle spindle Ia afferents (velocity-sensitive, modulated by gamma_dynamic)
    - Muscle spindle II afferents (length-sensitive, modulated by gamma_static)
    - Golgi tendon organ Ib afferents (force-sensitive)

    Maintains internal state for afferent activations.
    """

    def __init__(self, num_muscles: int, activation: Activation = Activation.RELU):
        super().__init__()
        self.num_muscles = num_muscles
        self.activation_fn = activation.to_function()

        # Internal state
        self._spindle_Ia: Optional[torch.Tensor] = None
        self._spindle_II: Optional[torch.Tensor] = None
        self._golgi_Ib: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Reset sensory state."""
        self._spindle_Ia = None
        self._spindle_II = None
        self._golgi_Ib = None

    @property
    def spindle_Ia(self) -> Optional[torch.Tensor]:
        """Last computed Ia afferent activation."""
        return self._spindle_Ia

    @property
    def spindle_II(self) -> Optional[torch.Tensor]:
        """Last computed II afferent activation."""
        return self._spindle_II

    @property
    def golgi_Ib(self) -> Optional[torch.Tensor]:
        """Last computed Ib afferent activation."""
        return self._golgi_Ib

    def forward(
        self,
        lengths: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        gamma_static: torch.Tensor,
        gamma_dynamic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (spindle_Ia, spindle_II, golgi_Ib)."""
        self._spindle_Ia = self.activation_fn(velocities * gamma_dynamic)
        self._spindle_II = self.activation_fn(lengths + gamma_static)
        self._golgi_Ib = self.activation_fn(forces)
        return self._spindle_Ia, self._spindle_II, self._golgi_Ib
