"""Shared configuration and types for neural controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Union

import torch
import torch.nn as nn


class Activation(Enum):
    """Supported activation functions."""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"

    def to_module(self) -> nn.Module:
        return {
            Activation.RELU: nn.ReLU,
            Activation.TANH: nn.Tanh,
            Activation.SIGMOID: nn.Sigmoid,
        }[self]()

    def to_function(self) -> callable:
        return {
            Activation.RELU: torch.relu,
            Activation.TANH: torch.tanh,
            Activation.SIGMOID: torch.sigmoid,
        }[self]


@dataclass(frozen=True)
class WorkspaceBounds:
    """2D workspace boundaries for target encoding."""

    x: Tuple[float, float] = (-0.3, 0.3)
    y: Tuple[float, float] = (0.1, 0.5)


@dataclass
class ControllerConfig:
    """Configuration for neural motor controllers."""

    num_muscles: int
    core_units: Union[int, List[int]] = 32
    target_grid_size: int = 4
    target_sigma: float = 0.5
    workspace_bounds: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    core_activation: Activation = Activation.RELU
    motor_activation: Activation = Activation.SIGMOID
    use_core_bias: bool = True
    use_motor_bias: bool = True

    @property
    def is_recurrent(self) -> bool:
        return isinstance(self.core_units, int)

    @property
    def core_output_size(self) -> int:
        return self.core_units if self.is_recurrent else self.core_units[-1]

    @property
    def target_size(self) -> int:
        return self.target_grid_size**2
