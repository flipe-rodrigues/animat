"""Core processing network modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

from ..config import Activation


class BaseCore(nn.Module, ABC):
    """Abstract base for core processing networks."""

    def __init__(self, input_size: int, target_size: int, num_muscles: int):
        super().__init__()
        self._input_size = input_size

        # Input projections (shared by all cores)
        self.target_to_core = nn.Linear(target_size, input_size, bias=False)
        self.Ia_to_core = nn.Linear(num_muscles, input_size, bias=False)
        self.II_to_core = nn.Linear(num_muscles, input_size, bias=False)
        self.Ib_to_core = nn.Linear(num_muscles, input_size, bias=False)
        self.alpha_to_core = nn.Linear(num_muscles, input_size, bias=False)

    def _project_inputs(
        self,
        target: torch.Tensor,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        golgi_Ib: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Project all inputs into core space."""
        return (
            self.target_to_core(target)
            + self.Ia_to_core(spindle_Ia)
            + self.II_to_core(spindle_II)
            + self.Ib_to_core(golgi_Ib)
            + self.alpha_to_core(alpha)
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def state(self) -> Optional[torch.Tensor]:
        """Internal state for recurrent networks, None for feedforward."""
        return None

    @abstractmethod
    def reset_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> None:
        """Reset internal state. No-op for feedforward networks."""
        pass

    @abstractmethod
    def forward(
        self,
        target: torch.Tensor,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        golgi_Ib: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        pass


class RNNCore(BaseCore):
    """Recurrent core with Elman-style hidden state."""

    def __init__(
        self,
        size: int,
        target_size: int,
        num_muscles: int,
        activation: Activation = Activation.RELU,
        use_bias: bool = False,
    ):
        super().__init__(
            input_size=size, target_size=target_size, num_muscles=num_muscles
        )
        self.activation_fn = activation.to_function()
        self.core_to_core = nn.Linear(size, size, bias=use_bias)
        self._state: Optional[torch.Tensor] = None

    @property
    def state(self) -> Optional[torch.Tensor]:
        return self._state

    def reset_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> None:
        device = device or torch.device("cpu")
        self._state = torch.zeros(batch_size, self.input_size, device=device)

    def forward(
        self,
        target: torch.Tensor,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        golgi_Ib: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if self._state is None:
            raise RuntimeError("Call reset_state() before forward()")

        inputs = self._project_inputs(target, spindle_Ia, spindle_II, golgi_Ib, alpha)
        self._state = self.activation_fn(inputs + self.core_to_core(self._state)).detach()
        return self._state


class MLPCore(BaseCore):
    """Feedforward multi-layer perceptron core."""

    def __init__(
        self,
        layer_sizes: List[int],
        target_size: int,
        num_muscles: int,
        activation: Activation = Activation.RELU,
        use_bias: bool = True,
    ):
        if not layer_sizes:
            raise ValueError("layer_sizes must have at least one element")

        # First layer size is used for input projection in BaseCore
        super().__init__(
            input_size=layer_sizes[0], target_size=target_size, num_muscles=num_muscles
        )

        self.core_to_core = self._build_layers(layer_sizes, activation, use_bias)
        self._output_size = layer_sizes[-1]

    @property
    def output_size(self) -> int:
        return self._output_size

    def _build_layers(
        self,
        layer_sizes: List[int],
        activation: Activation,
        use_bias: bool,
    ) -> nn.Module:
        if len(layer_sizes) == 1:
            return nn.Identity()

        layers: List[nn.Module] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size, bias=use_bias))
            layers.append(activation.to_module())

        return nn.Sequential(*layers)

    def reset_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> None:
        pass  # Stateless

    def forward(
        self,
        target: torch.Tensor,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        golgi_Ib: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._project_inputs(target, spindle_Ia, spindle_II, golgi_Ib, alpha)
        return self.core_to_core(inputs)
