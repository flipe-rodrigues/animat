"""
Controllers for Muscle-Driven Arm Control

Main controllers that integrate all modules:
- RNNController: Uses RNN core for temporal integration
- MLPController: Uses MLP core (for teacher in distillation)

Both controllers accept either:
- Structured ObservationTensor objects (preferred)
- Flat tensor arrays (for legacy compatibility)

The controllers implement monosynaptic stretch reflex arcs:
- Type Ia (velocity) -> Alpha motor neurons
- Type II (length) -> Alpha motor neurons
This coupling is a controller-level property, not part of the motor module.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List, Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ModelConfig
from core.base import BaseController
from core.types import ObservationTensor, ProprioceptionTensor
from .modules import (
    SensoryModule,
    MotorModule,
    TargetEncoder,
    RNNCore,
    MLPCore,
)


class ControllerBase(BaseController):
    """
    Base implementation for controllers with common functionality.
    
    Accepts both structured (ObservationTensor) and flat tensor inputs.
    Implements monosynaptic stretch reflex arcs (Ia/II -> Alpha).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Sensory module
        self.sensory = SensoryModule(
            num_muscles=config.num_muscles,
            use_bias=config.proprioceptive_bias,
        )

        # Target encoder: converts raw XYZ to Gaussian grid
        self.target_encoder = TargetEncoder(
            grid_size=config.target_grid_size,
            sigma=config.target_sigma,
            workspace_bounds=config.workspace_bounds,
        )

        # Motor module (produces cortical outputs only)
        self.motor = MotorModule(
            input_size=config.rnn_hidden_size,
            num_muscles=config.num_muscles,
            output_bias=config.output_bias,
        )

        # Monosynaptic stretch reflex arcs (Ia/II -> Alpha MN)
        # These are controller-level connections, not part of motor module
        self.Ia_to_alpha = nn.Linear(config.num_muscles, config.num_muscles, bias=config.reflex_bias)
        self.II_to_alpha = nn.Linear(config.num_muscles, config.num_muscles, bias=config.reflex_bias)
        self._init_reflex_weights()

        # State tracking
        self._gamma_static = None
        self._gamma_dynamic = None
        self._prev_alpha = None

    def _init_reflex_weights(self):
        """Initialize reflex connections with diagonal structure."""
        # Initialize with near-diagonal structure 
        # (each muscle's Ia/II projects mainly to same alpha MN)
        n = self.config.num_muscles
        Ia_weight = torch.eye(n) * 0.3 + torch.randn(n, n) * 0.05
        II_weight = torch.eye(n) * 0.2 + torch.randn(n, n) * 0.05

        self.Ia_to_alpha.weight.data = Ia_weight
        self.II_to_alpha.weight.data = II_weight

    def compute_alpha_with_reflex(
        self,
        alpha_cortical: torch.Tensor,
        type_Ia: torch.Tensor,
        type_II: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine cortical alpha with reflex contributions.
        
        Args:
            alpha_cortical: Cortical contribution [batch, num_muscles]
            type_Ia: Type Ia sensory activations [batch, num_muscles]
            type_II: Type II sensory activations [batch, num_muscles]
            
        Returns:
            alpha: Combined alpha MN output (0-1 range) [batch, num_muscles]
        """
        alpha_reflex_Ia = self.Ia_to_alpha(type_Ia)
        alpha_reflex_II = self.II_to_alpha(type_II)
        
        # Combine cortical and reflex contributions, then sigmoid to 0-1
        alpha = torch.sigmoid(alpha_cortical + alpha_reflex_Ia + alpha_reflex_II)
        return alpha

    def parse_input(
        self, obs: Union[torch.Tensor, ObservationTensor]
    ) -> Tuple[ProprioceptionTensor, torch.Tensor]:
        """
        Parse input to structured proprioception and encoded target.
        
        Args:
            obs: Either flat tensor [batch, obs_dim] or ObservationTensor
            
        Returns:
            proprio: ProprioceptionTensor with lengths, velocities, forces
            target_encoded: Gaussian grid encoding [batch, num_target_units]
        """
        if isinstance(obs, ObservationTensor):
            # Already structured - use directly
            proprio = obs.proprio
            target_encoded = self.target_encoder.encode(obs.target)
        else:
            # Flat tensor - parse by position (legacy support)
            num_muscles = self.config.num_muscles
            proprio = ProprioceptionTensor(
                lengths=obs[:, :num_muscles],
                velocities=obs[:, num_muscles:2*num_muscles],
                forces=obs[:, 2*num_muscles:3*num_muscles],
            )
            target_xyz = obs[:, 3*num_muscles:3*num_muscles + 3]
            target_encoded = self.target_encoder.encode(target_xyz)
        
        return proprio, target_encoded


class RNNController(ControllerBase):
    """
    RNN-based controller for muscle-driven arm control.

    Architecture:
    1. Sensory module processes proprioceptive inputs (Ia, II, Ib)
    2. Target encoding module processes target grid
    3. RNN core integrates over time (the only recurrent layer)
    4. Motor module generates alpha MN and gamma outputs
    5. Alpha MNs project back as feedback

    Note: No explicit time or phase encoding - the RNN learns temporal
    structure through its recurrent dynamics.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Compute RNN input dimension
        # Includes: sensory outputs, target encoding, and alpha feedback
        rnn_input_dim = (
            config.num_sensors  # Ia, II, Ib outputs
            + config.num_target_units  # Target encoding output
            + config.num_muscles  # Alpha MN feedback
        )

        # RNN core (the only recurrent module)
        self.core = RNNCore(
            input_size=rnn_input_dim,
            hidden_size=config.rnn_hidden_size,
            rnn_type=config.rnn_type,
            num_layers=config.num_rnn_layers,
            input_projection_bias=config.input_projection_bias,
        )

        # Hidden state tracking
        self._hidden = None

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Initialize hidden state for new episodes."""
        self._hidden = self.core.init_hidden(batch_size, device)
        self._prev_alpha = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_static = torch.ones(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_dynamic = torch.ones(
            batch_size, self.config.num_muscles, device=device
        )

    def forward(
        self,
        obs: Union[torch.Tensor, ObservationTensor],
        hidden: Optional[torch.Tensor] = None,
        prev_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for single timestep.

        Args:
            obs: Observation - either flat tensor or ObservationTensor
            hidden: Optional hidden state
            prev_alpha: Previous alpha MN output for feedback

        Returns:
            action: Alpha MN activations [batch, num_muscles]
            new_hidden: Updated hidden state
            info: Dictionary of intermediate outputs
        """
        # Determine batch size and device from input
        if isinstance(obs, ObservationTensor):
            batch_size = obs.batch_size
            device = obs.device
        else:
            batch_size = obs.shape[0]
            device = obs.device

        # Initialize states if needed
        if hidden is None:
            if self._hidden is None:
                self.init_hidden(batch_size, device)
            hidden = self._hidden

        if prev_alpha is None:
            if self._prev_alpha is None:
                self._prev_alpha = torch.zeros(batch_size, self.config.num_muscles, device=device)
            prev_alpha = self._prev_alpha

        gamma_static = self._gamma_static if self._gamma_static is not None else torch.ones(batch_size, self.config.num_muscles, device=device)
        gamma_dynamic = self._gamma_dynamic if self._gamma_dynamic is not None else torch.ones(batch_size, self.config.num_muscles, device=device)

        # Parse input to structured form
        proprio, target_encoded = self.parse_input(obs)

        # Process sensory input with gamma modulation
        sensory_out, sensory_outputs = self.sensory(
            proprio.lengths,
            proprio.velocities,
            proprio.forces,
            gamma_static,
            gamma_dynamic,
        )

        # Combine inputs for RNN
        rnn_input = torch.cat([sensory_out, target_encoded, prev_alpha], dim=1)
        rnn_output, new_hidden = self.core(rnn_input, hidden)

        # Generate motor outputs (cortical contribution)
        alpha_cortical, gamma_static_new, gamma_dynamic_new = self.motor(rnn_output)

        # Apply stretch reflex to get final alpha output
        alpha = self.compute_alpha_with_reflex(
            alpha_cortical,
            sensory_outputs["type_Ia"],
            sensory_outputs["type_II"],
        )

        # Update stored state
        self._hidden = new_hidden
        self._prev_alpha = alpha.detach()
        self._gamma_static = gamma_static_new.detach()
        self._gamma_dynamic = gamma_dynamic_new.detach()

        info = {
            "alpha": alpha,
            "alpha_cortical": alpha_cortical,
            "gamma_static": gamma_static_new,
            "gamma_dynamic": gamma_dynamic_new,
            "sensory_outputs": sensory_outputs,
            "rnn_hidden": rnn_output,
        }

        return alpha, new_hidden, info

    def forward_sequence(
        self, obs_sequence: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for sequence of observations.

        Args:
            obs_sequence: Observations [batch, seq_len, obs_dim]
            hidden: Optional initial hidden state

        Returns:
            actions: Actions [batch, seq_len, action_dim]
            final_hidden: Final hidden state
            infos: List of info dicts per timestep
        """
        batch_size, seq_len, _ = obs_sequence.shape
        device = obs_sequence.device

        if hidden is None:
            self.init_hidden(batch_size, device)
            hidden = self._hidden

        actions = []
        infos = []
        prev_alpha = torch.zeros(batch_size, self.config.num_muscles, device=device)

        for t in range(seq_len):
            obs_t = obs_sequence[:, t, :]
            action, hidden, info = self.forward(obs_t, hidden, prev_alpha)
            actions.append(action)
            infos.append(info)
            prev_alpha = info["alpha"].detach()

        actions = torch.stack(actions, dim=1)

        return actions, hidden, infos


class MLPController(ControllerBase):
    """
    MLP-based controller (no recurrence).

    Same interface as RNNController but uses feedforward MLP core.
    Used as teacher network in distillation learning.
    """

    def __init__(self, config: ModelConfig, hidden_sizes: Optional[List[int]] = None):
        super().__init__(config)

        if hidden_sizes is None:
            hidden_sizes = config.mlp_hidden_sizes

        # Compute MLP input dimension
        mlp_input_dim = (
            config.num_sensors  # Sensory outputs
            + config.num_target_units  # Target encoding output
        )

        # MLP core (no recurrence)
        self.core = MLPCore(
            input_size=mlp_input_dim,
            output_size=config.rnn_hidden_size,  # Use same output size as RNN
            hidden_sizes=hidden_sizes,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Initialize state for new episodes (no hidden state for MLP)."""
        self._prev_alpha = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_static = torch.ones(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_dynamic = torch.ones(
            batch_size, self.config.num_muscles, device=device
        )

    def forward(
        self,
        obs: Union[torch.Tensor, ObservationTensor],
        hidden: Optional[torch.Tensor] = None,
        prev_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            obs: Observation - either flat tensor or ObservationTensor
            hidden: Ignored (for interface compatibility)
            prev_alpha: Ignored (MLP doesn't use feedback)

        Returns:
            action: Alpha MN activations [batch, num_muscles]
            None: (no hidden state)
            info: Dictionary of intermediate outputs
        """
        # Determine batch size and device from input
        if isinstance(obs, ObservationTensor):
            batch_size = obs.batch_size
            device = obs.device
        else:
            batch_size = obs.shape[0]
            device = obs.device

        # Use default gamma values (MLP doesn't track state)
        gamma_static = torch.ones(batch_size, self.config.num_muscles, device=device)
        gamma_dynamic = torch.ones(batch_size, self.config.num_muscles, device=device)

        # Parse input to structured form
        proprio, target_encoded = self.parse_input(obs)

        # Process sensory input
        sensory_out, sensory_outputs = self.sensory(
            proprio.lengths,
            proprio.velocities,
            proprio.forces,
            gamma_static,
            gamma_dynamic,
        )

        # Combine inputs for MLP
        mlp_input = torch.cat([sensory_out, target_encoded], dim=1)
        mlp_output = self.core(mlp_input)

        # Generate motor outputs (cortical contribution)
        alpha_cortical, gamma_static_new, gamma_dynamic_new = self.motor(mlp_output)

        # Apply stretch reflex to get final alpha output
        alpha = self.compute_alpha_with_reflex(
            alpha_cortical,
            sensory_outputs["type_Ia"],
            sensory_outputs["type_II"],
        )

        info = {
            "alpha": alpha,
            "alpha_cortical": alpha_cortical,
            "gamma_static": gamma_static_new,
            "gamma_dynamic": gamma_dynamic_new,
            "sensory_outputs": sensory_outputs,
        }

        return alpha, None, info


def create_controller(
    config: ModelConfig, controller_type: str = "rnn", **kwargs
) -> BaseController:
    """
    Factory function to create controllers.

    Args:
        config: Model configuration
        controller_type: 'rnn' or 'mlp'
        **kwargs: Additional arguments for specific controller types

    Returns:
        Controller instance
    """
    controller_type = controller_type.lower()
    
    if controller_type == "rnn":
        return RNNController(config)
    elif controller_type == "mlp":
        return MLPController(config, **kwargs)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}. Expected 'rnn' or 'mlp'.")
