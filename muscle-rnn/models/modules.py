from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


"""
.########.....###.....######..########
.##.....##...##.##...##....##.##......
.##.....##..##...##..##.......##......
.########..##.....##..######..######..
.##.....##.#########.......##.##......
.##.....##.##.....##.##....##.##......
.########..##.....##..######..########
"""


class BaseModule(nn.Module, ABC):
    def __init__(
        self,
        num_units: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ):
        super().__init__()
        self._num_units = num_units
        self.activation_fn = activation_fn

    @property
    def num_units(self) -> int:
        return self._num_units


"""
.########....###....########...######...########.########
....##......##.##...##.....##.##....##..##..........##...
....##.....##...##..##.....##.##........##..........##...
....##....##.....##.########..##...####.######......##...
....##....#########.##...##...##....##..##..........##...
....##....##.....##.##....##..##....##..##..........##...
....##....##.....##.##.....##..######...########....##...
"""


class TargetEncoder(BaseModule):
    def __init__(
        self,
        grid_size: int = 4,
        sigma: float = 0.5,
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = {
            "x": (-0.3, 0.3),
            "y": (0.1, 0.5),
        },
    ):
        super().__init__(num_units=grid_size**2)
        self._sigma = sigma
        self._grid_size = grid_size
        self._create_grid(grid_size, workspace_bounds)

    def _create_grid(
        self, grid_size: int, workspace_bounds: Dict[str, Tuple[float, float]]
    ) -> None:
        x_min, x_max = workspace_bounds["x"]
        y_min, y_max = workspace_bounds["y"]
        x_centers = torch.linspace(x_min, x_max, grid_size)
        y_centers = torch.linspace(y_min, y_max, grid_size)
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="xy")
        self.register_buffer(
            "grid_centers", torch.stack([xx.flatten(), yy.flatten()], dim=1)
        )

    def forward(self, target_xyz: torch.Tensor) -> torch.Tensor:

        target_xy = target_xyz[:, :2]
        diff = target_xy.unsqueeze(1) - self.grid_centers.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        activation = torch.exp(-dist_sq / (2 * self._sigma**2))

        return activation / (activation.sum(dim=1, keepdim=True) + 1e-8)


"""
..######..########.##....##..######...#######..########..##....##
.##....##.##.......###...##.##....##.##.....##.##.....##..##..##.
.##.......##.......####..##.##.......##.....##.##.....##...####..
..######..######...##.##.##..######..##.....##.########.....##...
.......##.##.......##..####.......##.##.....##.##...##......##...
.##....##.##.......##...###.##....##.##.....##.##....##.....##...
..######..########.##....##..######...#######..##.....##....##...
"""


class SensoryModule(BaseModule):

    def __init__(
        self,
        num_muscles: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ):
        super().__init__(num_units=num_muscles * 3, activation_fn=activation_fn)

    def forward(
        self,
        lengths: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        gamma_static: torch.Tensor,
        gamma_dynamic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        spindle_Ia = self.activation_fn(velocities * gamma_dynamic)
        spindle_II = self.activation_fn(lengths + gamma_static)
        golgi_Ib = self.activation_fn(forces)

        return spindle_Ia, spindle_II, golgi_Ib


"""
.##.....##..#######..########..#######..########.
.###...###.##.....##....##....##.....##.##.....##
.####.####.##.....##....##....##.....##.##.....##
.##.###.##.##.....##....##....##.....##.########.
.##.....##.##.....##....##....##.....##.##...##..
.##.....##.##.....##....##....##.....##.##....##.
.##.....##..#######.....##.....#######..##.....##
"""


class MotorModule(BaseModule):
    def __init__(
        self,
        num_muscles: int,
        core_output_size: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        bias: bool = False,
    ):
        super().__init__(num_units=num_muscles * 3, activation_fn=activation_fn)

        self.Ia_to_alpha = nn.Linear(num_muscles, num_muscles, bias=False)
        self.II_to_alpha = nn.Linear(num_muscles, num_muscles, bias=False)
        self.core_to_alpha = nn.Linear(core_output_size, num_muscles, bias=bias)
        self.core_to_gamma_static = nn.Linear(core_output_size, num_muscles, bias=bias)
        self.core_to_gamma_dynamic = nn.Linear(core_output_size, num_muscles, bias=bias)
        
        self._alpha = None

    @property
    def alpha_state(self) -> Optional[torch.Tensor]:
        return self._alpha

    def forward(
        self,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        core: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        alpha = self.activation_fn(
            self.Ia_to_alpha(spindle_Ia)
            + self.II_to_alpha(spindle_II)
            + self.core_to_alpha(core)
        )
        gamma_static = self.activation_fn(self.core_to_gamma_static(core))
        gamma_dynamic = self.activation_fn(self.core_to_gamma_dynamic(core))

        self._alpha = alpha.detach()
        return alpha, gamma_static, gamma_dynamic


"""
..######...#######..########..########
.##....##.##.....##.##.....##.##......
.##.......##.....##.##.....##.##......
.##.......##.....##.########..######..
.##.......##.....##.##...##...##......
.##....##.##.....##.##....##..##......
..######...#######..##.....##.########
"""


class BaseCore(BaseModule, ABC):

    @abstractmethod
    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        pass

    @property
    def core_state(self) -> Optional[torch.Tensor]:
        return None

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


"""
.########..##....##.##....##
.##.....##.###...##.###...##
.##.....##.####..##.####..##
.########..##.##.##.##.##.##
.##...##...##..####.##..####
.##....##..##...###.##...###
.##.....##.##....##.##....##
"""


class RNNCore(BaseCore):
    def __init__(
        self,
        core_size: int,
        target_size: int,
        num_muscles: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        bias: bool = False,
    ):
        super().__init__(num_units=core_size, activation_fn=activation_fn)

        self.target_to_core = nn.Linear(target_size, core_size, bias=False)
        self.Ia_to_core = nn.Linear(num_muscles, core_size, bias=False)
        self.II_to_core = nn.Linear(num_muscles, core_size, bias=False)
        self.Ib_to_core = nn.Linear(num_muscles, core_size, bias=False)
        self.alpha_to_core = nn.Linear(num_muscles, core_size, bias=False)
        self.core_to_core = nn.Linear(core_size, core_size, bias=bias)

        self._core = None

    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        device = device or torch.device("cpu")
        self._core = torch.zeros(batch_size, self.num_units, device=device)

    @property
    def core_state(self) -> Optional[torch.Tensor]:
        return self._core

    def forward(
        self,
        target: torch.Tensor,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        golgi_Ib: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:

        core = self.activation_fn(
            self.target_to_core(target)
            + self.Ia_to_core(spindle_Ia)
            + self.II_to_core(spindle_II)
            + self.Ib_to_core(golgi_Ib)
            + self.alpha_to_core(alpha)
            + self.core_to_core(self._core)
        )

        self._core = core.detach()
        return core


"""
.##.....##.##.......########.
.###...###.##.......##.....##
.####.####.##.......##.....##
.##.###.##.##.......########.
.##.....##.##.......##.......
.##.....##.##.......##.......
.##.....##.########.##.......
"""


class MLPCore(BaseCore):
    def __init__(
        self,
        layer_sizes: List[int],
        target_size: int,
        num_muscles: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        bias: bool = True,
    ):
        super().__init__(num_units=sum(layer_sizes), activation_fn=activation_fn)
        input_layer_size = layer_sizes[0]

        self.target_to_core = nn.Linear(target_size, input_layer_size, bias=False)
        self.Ia_to_core = nn.Linear(num_muscles, input_layer_size, bias=False)
        self.II_to_core = nn.Linear(num_muscles, input_layer_size, bias=False)
        self.Ib_to_core = nn.Linear(num_muscles, input_layer_size, bias=False)
        self.alpha_to_core = nn.Linear(num_muscles, input_layer_size, bias=False)

        layers = []
        prev_size = input_layer_size
        for layer_size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, layer_size, bias=bias))
            if activation_fn == torch.relu:
                layers.append(nn.ReLU())
            elif activation_fn == torch.tanh:
                layers.append(nn.Tanh())
            elif activation_fn == torch.sigmoid:
                layers.append(nn.Sigmoid())
            prev_size = layer_size
        self.core_to_core = nn.Sequential(*layers) if layers else nn.Identity()

    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        pass

    def forward(
        self,
        target: torch.Tensor,
        spindle_Ia: torch.Tensor,
        spindle_II: torch.Tensor,
        golgi_Ib: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:

        core = self.core_to_core(
            self.target_to_core(target)
            + self.Ia_to_core(spindle_Ia)
            + self.II_to_core(spindle_II)
            + self.Ib_to_core(golgi_Ib)
            + self.alpha_to_core(alpha)
        )

        return core
