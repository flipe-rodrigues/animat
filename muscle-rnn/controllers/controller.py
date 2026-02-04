"""
Neural motor controller with biologically-inspired sensory processing.

This module implements a hierarchical motor control architecture with:
- Target encoding via spatial receptive fields
- Proprioceptive sensory processing (muscle spindles, Golgi tendon organs)
- Configurable core networks (RNN or MLP)
- Motor output with alpha and gamma motor neuron pathways
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ControllerConfig
from .modules import MotorModule, RNNCore, MLPCore, SensoryModule, TargetEncoder
from .modules.core import BaseCore


class Controller(nn.Module):
    """
    Neural motor controller integrating sensory processing and motor output.

    Supports both RNN and MLP cores via configuration. Provides interfaces
    for CMA-ES optimization and Stable Baselines 3 compatibility.
    """

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config

        # Build modules
        self.target_encoder = TargetEncoder(
            grid_size=config.target_grid_size,
            sigma=config.target_sigma,
            bounds=config.workspace_bounds,
        )
        self.sensory = SensoryModule(
            num_muscles=config.num_muscles,
            activation=config.core_activation,
        )
        self.motor = MotorModule(
            num_muscles=config.num_muscles,
            core_output_size=config.core_output_size,
            activation=config.motor_activation,
            use_bias=config.use_motor_bias,
        )
        self.core = self._create_core(config)

    def _create_core(self, config: ControllerConfig) -> BaseCore:
        common = dict(
            target_size=config.target_size,
            num_muscles=config.num_muscles,
            activation=config.core_activation,
            use_bias=config.use_core_bias,
        )
        if config.is_recurrent:
            return RNNCore(size=config.core_units, **common)
        return MLPCore(layer_sizes=config.core_units, **common)

    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset all internal state for new episodes."""
        self.core.reset_state(batch_size, device)
        self.motor.reset_state(batch_size, device)
        self.sensory.reset_state()

    def _parse_observation(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract muscle states and target from observation vector."""
        n = self.config.num_muscles
        return (
            obs[:, :n],  # lengths
            obs[:, n : 2 * n],  # velocities
            obs[:, 2 * n : 3 * n],  # forces
            self.target_encoder(obs[:, 3 * n : 3 * n + 3]),  # target
        )

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        """Initialize state if needed or batch size changed."""
        if self.motor.gamma_static is None or self.motor.gamma_static.shape[0] != batch_size:
            self.reset_state(batch_size, device)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the controller.

        Args:
            obs: Observation tensor [batch, obs_dim] containing muscle lengths,
                 velocities, forces, and 3D target position.

        Returns:
            Tuple of (alpha, info_dict) where info_dict contains diagnostic tensors.
        """
        self._ensure_state(obs.shape[0], obs.device)

        # Parse observation
        lengths, velocities, forces, target = self._parse_observation(obs)

        # Sensory processing (using gamma state from motor module)
        spindle_Ia, spindle_II, golgi_Ib = self.sensory(
            lengths, velocities, forces, self.motor.gamma_static, self.motor.gamma_dynamic
        )

        # Core processing
        core_output = self.core(target, spindle_Ia, spindle_II, golgi_Ib, self.motor.alpha)

        # Motor output (updates internal gamma state)
        alpha, gamma_static, gamma_dynamic = self.motor(spindle_Ia, spindle_II, core_output)

        info = {
            "alpha": alpha,
            "gamma_static": gamma_static,
            "gamma_dynamic": gamma_dynamic,
            "spindle_Ia": self.sensory.spindle_Ia,
            "spindle_II": self.sensory.spindle_II,
            "golgi_Ib": self.sensory.golgi_Ib,
            "core_state": self.core.state,
        }

        return alpha, info

    # -------------------------------------------------------------------------
    # CMA-ES Interface
    # -------------------------------------------------------------------------

    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as a flat numpy array."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat numpy array."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(
                torch.from_numpy(flat_params[idx : idx + size].reshape(p.shape)).to(p.device)
            )
            idx += size

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------------
    # Stable Baselines 3 Interface
    # -------------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Stable Baselines 3 compatible prediction interface.

        Args:
            observation: Numpy observation array.
            state: Unused (maintained for API compatibility).
            episode_start: Boolean array indicating new episodes.
            deterministic: Unused (controller is deterministic).

        Returns:
            Tuple of (action array, None).
        """
        if episode_start is not None and np.any(episode_start):
            self.reset_state()

        obs_tensor = torch.from_numpy(observation).float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action, _ = self.forward(obs_tensor)

        return action.cpu().numpy().squeeze(0), None


# =============================================================================
# Factory Functions
# =============================================================================


def create_rnn_controller(num_muscles: int, core_size: int = 32, **kwargs) -> Controller:
    """Create an RNN-based controller."""
    return Controller(ControllerConfig(num_muscles=num_muscles, core_units=core_size, **kwargs))


def create_mlp_controller(num_muscles: int, layer_sizes: List[int] = [64, 32], **kwargs) -> Controller:
    """Create an MLP-based controller."""
    return Controller(ControllerConfig(num_muscles=num_muscles, core_units=layer_sizes, **kwargs))
