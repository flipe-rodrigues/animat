from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


"""
..######...#######..##....##.########.####..######..
.##....##.##.....##.###...##.##........##..##....##.
.##.......##.....##.####..##.##........##..##.......
.##.......##.....##.##.##.##.######....##..##...####
.##.......##.....##.##..####.##........##..##....##.
.##....##.##.....##.##...###.##........##..##....##.
..######...#######..##....##.##.......####..######..
"""


@dataclass
class ControllerConfig:

    num_muscles: int
    num_core_units: Union[int, List[int]] = 32
    target_grid_size: int = 4
    target_sigma: float = 0.5
    workspace_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"x": (-0.3, 0.3), "y": (0.1, 0.5)}
    )
    core_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu
    motor_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid
    core_bias: bool = True
    motor_bias: bool = True

    @property
    def core_input_size(self) -> int:
        if isinstance(self.num_core_units, list):
            return self.num_core_units[0]
        return self.num_core_units

    @property
    def core_output_size(self) -> int:
        if isinstance(self.num_core_units, list):
            return self.num_core_units[-1]
        return self.num_core_units

    @property
    def num_Ia_units(self) -> int:
        return self.num_muscles

    @property
    def num_II_units(self) -> int:
        return self.num_muscles

    @property
    def num_Ib_units(self) -> int:
        return self.num_muscles

    @property
    def num_target_units(self) -> int:
        return self.target_grid_size**2

    @property
    def num_alpha_units(self) -> int:
        return self.num_muscles

    @property
    def num_gamma_units(self) -> int:
        return self.num_muscles

    @property
    def activation_fn(self) -> Callable:
        return self.core_activation_fn


from .modules import (
    BaseModule,
    TargetEncoder,
    SensoryModule,
    MotorModule,
    BaseCore,
    RNNCore,
    MLPCore,
)


"""
.########.....###.....######..########
.##.....##...##.##...##....##.##......
.##.....##..##...##..##.......##......
.########..##.....##..######..######..
.##.....##.#########.......##.##......
.##.....##.##.....##.##....##.##......
.########..##.....##..######..########
"""


class BaseController(nn.Module, ABC):

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        self.target_encoder = TargetEncoder(
            grid_size=config.target_grid_size,
            sigma=config.target_sigma,
            workspace_bounds=config.workspace_bounds,
        )
        self.sensory = SensoryModule(
            num_muscles=config.num_muscles,
            activation_fn=config.core_activation_fn,
        )
        self.motor = MotorModule(
            num_muscles=config.num_muscles,
            activation_fn=config.motor_activation_fn,
            core_output_size=config.core_output_size,
            bias=config.motor_bias,
        )
        self.core: BaseCore = self._create_core(config)
        self._reset_state()

    @abstractmethod
    def _create_core(self, config: ControllerConfig) -> BaseCore:
        pass

    def _reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        device = device or torch.device("cpu")
        self._gamma_static = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_dynamic = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self.core.reset_state(batch_size, device)

    def _parse_observation(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.config.num_muscles
        lengths = obs[:, :n]
        velocities = obs[:, n : 2 * n]
        forces = obs[:, 2 * n : 3 * n]
        target_xyz = obs[:, 3 * n : 3 * n + 3]
        target = self.target_encoder(target_xyz)
        return lengths, velocities, forces, target

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        batch_size, device = obs.shape[0], obs.device

        # Ensure state matches batch size
        if self._gamma_static is None or self._gamma_static.shape[0] != batch_size:
            self._reset_state(batch_size, device)

        # Parse observations
        lengths, velocities, forces, target = self._parse_observation(obs)

        # Sensory processing
        spindle_Ia, spindle_II, golgi_Ib = self.sensory(
            lengths, velocities, forces, self._gamma_static, self._gamma_dynamic
        )

        # Get previous alpha for core input
        prev_alpha = self.motor.alpha_state
        if prev_alpha is None:
            prev_alpha = torch.zeros(batch_size, self.config.num_muscles, device=device)

        # Core processing
        core = self.core(target, spindle_Ia, spindle_II, golgi_Ib, prev_alpha)

        # Motor output
        alpha, gamma_static, gamma_dynamic = self.motor(spindle_Ia, spindle_II, core)

        # Update state
        self._gamma_static = gamma_static.detach()
        self._gamma_dynamic = gamma_dynamic.detach()

        # Collect info for visualization/analysis
        info = {
            "alpha": alpha,
            "gamma_static": gamma_static,
            "gamma_dynamic": gamma_dynamic,
            "sensory_outputs": (spindle_Ia, spindle_II, golgi_Ib),
            "core_state": self.core.core_state,
        }

        return alpha, info

    # -------------------------------------------------------------------------
    # CMA-ES Interface
    # -------------------------------------------------------------------------

    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as a flat numpy array for CMA-ES."""
        return np.concatenate(
            [p.data.cpu().numpy().flatten() for p in self.parameters()]
        )

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat numpy array (CMA-ES solution)."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(
                torch.from_numpy(flat_params[idx : idx + size].reshape(p.shape)).to(
                    p.device
                )
            )
            idx += size

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------------
    # Stable Baselines 3 Interface (for distillation)
    # -------------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:

        # Handle episode starts
        if episode_start is not None:
            if np.any(episode_start):
                self._reset_state()

        # Convert to tensor
        obs_tensor = torch.from_numpy(observation).float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            action, _ = self.forward(obs_tensor)

        return action.cpu().numpy().squeeze(0), None


"""
.########..##....##.##....##
.##.....##.###...##.###...##
.##.....##.####..##.####..##
.########..##.##.##.##.##.##
.##...##...##..####.##..####
.##....##..##...###.##...###
.##.....##.##....##.##....##
"""


class RNNController(BaseController):
    def __init__(self, config: ControllerConfig):
        if isinstance(config.num_core_units, list):
            raise ValueError(
                "RNNController requires num_core_units to be an int, not a list"
            )
        super().__init__(config)

    def _create_core(self, config: ControllerConfig) -> BaseCore:
        return RNNCore(
            core_size=config.num_core_units,
            target_size=config.num_target_units,
            num_muscles=config.num_muscles,
            activation_fn=config.core_activation_fn,
            bias=config.core_bias,
        )


"""
.##.....##.##.......########.
.###...###.##.......##.....##
.####.####.##.......##.....##
.##.###.##.##.......########.
.##.....##.##.......##.......
.##.....##.##.......##.......
.##.....##.########.##.......
"""


class MLPController(BaseController):
    def __init__(self, config: ControllerConfig):
        if not isinstance(config.num_core_units, list):
            raise ValueError(
                "MLPController requires num_core_units to be a list, not an int"
            )
        super().__init__(config)

    def _create_core(self, config: ControllerConfig) -> BaseCore:
        return MLPCore(
            layer_sizes=config.num_core_units,
            target_size=config.num_target_units,
            num_muscles=config.num_muscles,
            activation_fn=config.core_activation_fn,
            bias=config.core_bias,
        )
