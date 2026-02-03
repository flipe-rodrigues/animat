"""
Neural Controllers for Motor Control

Provides neural network controllers with support for:
- CMA-ES evolutionary optimization (flat parameter interface)
- Distillation learning via Stable Baselines 3 (policy interface)
- Recurrent and feedforward architectures
"""

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
    """Configuration for neural controllers.

    Args:
        num_muscles: Number of muscle actuators
        num_core_units: Core network size. Use int for RNN (e.g., 32) or
                       List[int] for MLP hidden layers (e.g., [128, 128])
        target_grid_size: Spatial encoding grid size
        target_sigma: Gaussian encoding width
        workspace_bounds: Workspace limits for target encoding
        nonlinearity: Activation function ('relu' or 'tanh')
        use_bias: Whether to use bias in linear layers
    """

    num_muscles: int
    num_core_units: Union[int, List[int]] = 32
    target_grid_size: int = 4
    target_sigma: float = 0.5
    workspace_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"x": (-0.3, 0.3), "y": (0.1, 0.5)}
    )
    nonlinearity: str = "relu"
    use_bias: bool = True

    @property
    def core_input_size(self) -> int:
        """Size of input to core network (first layer)."""
        if isinstance(self.num_core_units, list):
            return self.num_core_units[0]
        return self.num_core_units

    @property
    def core_output_size(self) -> int:
        """Size of output from core network (last layer)."""
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
        return torch.relu if self.nonlinearity == "relu" else torch.tanh


"""
.########....###....########...######...########.########
....##......##.##...##.....##.##....##..##..........##...
....##.....##...##..##.....##.##........##..........##...
....##....##.....##.########..##...####.######......##...
....##....#########.##...##...##....##..##..........##...
....##....##.....##.##....##..##....##..##..........##...
....##....##.....##.##.....##..######...########....##...
"""


class TargetEncoder(nn.Module):
    """Encodes target positions as Gaussian activations on a spatial grid."""

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.sigma = config.target_sigma
        self._create_grid(config)

    def _create_grid(self, config: ControllerConfig) -> None:
        x_min, x_max = config.workspace_bounds["x"]
        y_min, y_max = config.workspace_bounds["y"]
        x_centers = torch.linspace(x_min, x_max, config.target_grid_size)
        y_centers = torch.linspace(y_min, y_max, config.target_grid_size)
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="xy")
        self.register_buffer(
            "grid_centers", torch.stack([xx.flatten(), yy.flatten()], dim=1)
        )

    def forward(self, target_xyz: torch.Tensor) -> torch.Tensor:
        target_xy = target_xyz[:, :2]
        diff = target_xy.unsqueeze(1) - self.grid_centers.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        activation = torch.exp(-dist_sq / (2 * self.sigma**2))
        return activation / (activation.sum(dim=1, keepdim=True) + 1e-8)


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
    """
    Abstract base controller with proprioceptive pathways.

    Supports both CMA-ES (via flat parameter interface) and distillation
    (via SB3-compatible predict method).
    """

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        self.target_encoder = TargetEncoder(config)
        self._init_weights(config)
        self._reset_state()

    def _init_weights(self, config: ControllerConfig) -> None:
        self.Ia_to_core = nn.Linear(config.num_Ia_units, config.core_input_size, False)
        self.II_to_core = nn.Linear(config.num_II_units, config.core_input_size, False)
        self.Ib_to_core = nn.Linear(config.num_Ib_units, config.core_input_size, False)
        self.Ia_to_alpha = nn.Linear(config.num_muscles, config.num_alpha_units, False)
        self.II_to_alpha = nn.Linear(config.num_muscles, config.num_alpha_units, False)
        self.target_to_core = nn.Linear(
            config.num_target_units, config.core_input_size, False
        )
        self.alpha_to_core = nn.Linear(
            config.num_alpha_units, config.core_input_size, False
        )
        self.core_to_alpha = nn.Linear(
            config.core_output_size, config.num_alpha_units, False
        )
        self.core_to_gamma_static = nn.Linear(
            config.core_output_size, config.num_gamma_units, False
        )
        self.core_to_gamma_dynamic = nn.Linear(
            config.core_output_size, config.num_gamma_units, False
        )

    def _reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        """Reset internal recurrent state."""
        device = device or torch.device("cpu")
        self._gamma_static = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_dynamic = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._prev_alpha = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._core = None

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

    @abstractmethod
    def _core_forward(
        self,
        sensory_input: torch.Tensor,
        target_input: torch.Tensor,
        alpha_input: torch.Tensor,
    ) -> torch.Tensor:
        """Process inputs through core network. Override in subclasses."""
        pass

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass computing muscle activations.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            alpha: Muscle activations [batch, num_muscles]
            info: Dict with intermediate values for analysis
        """
        batch_size, device = obs.shape[0], obs.device

        # Ensure state matches batch size
        if self._gamma_static is None or self._gamma_static.shape[0] != batch_size:
            self._reset_state(batch_size, device)

        # Parse observations
        lengths, velocities, forces, target = self._parse_observation(obs)

        # Sensory transform (spindle and Golgi tendon organ models)
        spindle_Ia = velocities * self._gamma_dynamic
        spindle_II = lengths + self._gamma_static
        golgi_Ib = forces

        # Aggregate sensory inputs to core
        sensory_to_core = (
            self.Ia_to_core(spindle_Ia)
            + self.II_to_core(spindle_II)
            + self.Ib_to_core(golgi_Ib)
        )
        target_to_core = self.target_to_core(target)
        alpha_to_core = self.alpha_to_core(self._prev_alpha)

        # Core processing
        core = self._core_forward(sensory_to_core, target_to_core, alpha_to_core)

        # Outputs from core
        core_to_alpha = self.core_to_alpha(core)
        core_to_gamma_static = self.core_to_gamma_static(core)
        core_to_gamma_dynamic = self.core_to_gamma_dynamic(core)

        # Monosynaptic stretch reflex
        reflex_to_alpha = self.Ia_to_alpha(spindle_Ia) + self.II_to_alpha(spindle_II)

        # Final activations
        alpha = torch.sigmoid(core_to_alpha + reflex_to_alpha)
        gamma_static_new = torch.sigmoid(core_to_gamma_static)
        gamma_dynamic_new = torch.sigmoid(core_to_gamma_dynamic)

        # Update state
        self._gamma_static = gamma_static_new.detach()
        self._gamma_dynamic = gamma_dynamic_new.detach()
        self._prev_alpha = alpha.detach()

        info = {
            "alpha": alpha,
            "gamma_static": gamma_static_new,
            "gamma_dynamic": gamma_dynamic_new,
            "spindle_Ia": spindle_Ia,
            "spindle_II": spindle_II,
            "golgi_Ib": golgi_Ib,
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
        """
        SB3-compatible predict interface for distillation.

        Args:
            observation: Numpy observation(s)
            state: Ignored (for API compatibility)
            episode_start: If True, reset internal state
            deterministic: Ignored (controller is deterministic)

        Returns:
            actions: Numpy array of actions
            state: None (state handled internally)
        """
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
    """Recurrent controller with internal dynamics.

    Expects config.num_core_units to be an int.
    """

    def __init__(self, config: ControllerConfig):
        if isinstance(config.num_core_units, list):
            raise ValueError(
                "RNNController requires num_core_units to be an int, not a list"
            )
        super().__init__(config)
        self.core_to_core_rnn = nn.Linear(
            config.num_core_units, config.num_core_units, bias=config.use_bias
        )

    def _reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        super()._reset_state(batch_size, device)
        device = device or torch.device("cpu")
        self._core = torch.zeros(batch_size, self.config.num_core_units, device=device)

    def _core_forward(
        self,
        sensory_input: torch.Tensor,
        target_input: torch.Tensor,
        alpha_input: torch.Tensor,
    ) -> torch.Tensor:
        if self._core is None:
            self._core = torch.zeros_like(sensory_input)
        new_core = self.config.activation_fn(
            sensory_input + target_input + alpha_input + self.core_to_core_rnn(self._core)
        )
        self._core = new_core.detach()
        return new_core


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
    """Feedforward controller with MLP core.

    Expects config.num_core_units to be a List[int] specifying hidden layer sizes.
    """

    def __init__(self, config: ControllerConfig):
        if not isinstance(config.num_core_units, list):
            raise ValueError(
                "MLPController requires num_core_units to be a list, not an int"
            )
        super().__init__(config)

    def _init_weights(self, config: ControllerConfig) -> None:
        super()._init_weights(config)

        # Build MLP from config.num_core_units list
        layers = []
        hidden_sizes = config.num_core_units  # This is a list
        input_size = config.core_input_size

        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(input_size, hidden_size, bias=config.use_bias))
            if config.nonlinearity == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            input_size = hidden_size

        self.core_to_core_mlp = nn.Sequential(*layers) if layers else nn.Identity()

    def _core_forward(
        self,
        sensory_input: torch.Tensor,
        target_input: torch.Tensor,
        alpha_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.core_to_core_mlp(sensory_input + target_input + alpha_input)
