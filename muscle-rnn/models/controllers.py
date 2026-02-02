"""Controllers for Muscle-Driven Arm Control."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Union

from core.config import ModelConfig
from core.base import BaseController
from core.types import ObservationTensor, ProprioceptionTensor
from .modules import SensoryModule, MotorModule, TargetEncoder, RNNCore, MLPCore


class ControllerBase(BaseController):
    """
    Base implementation for controllers with common functionality.

    Accepts both structured (ObservationTensor) and flat tensor inputs.
    Implements monosynaptic stretch reflex arcs (Ia/II -> Alpha).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.sensory = SensoryModule(
            config.num_muscles, use_bias=config.proprioceptive_bias
        )
        self.target_encoder = TargetEncoder(
            grid_size=config.target_grid_size,
            sigma=config.target_sigma,
            workspace_bounds=config.workspace_bounds,
        )
        self.motor = MotorModule(
            config.rnn_hidden_size, config.num_muscles, config.output_bias
        )

        # Monosynaptic stretch reflex (Ia/II -> Alpha MN)
        self.Ia_to_alpha = nn.Linear(
            config.num_muscles, config.num_muscles, bias=config.reflex_bias
        )
        self.II_to_alpha = nn.Linear(
            config.num_muscles, config.num_muscles, bias=config.reflex_bias
        )
        self._init_reflex_weights()

        # State tracking
        self._gamma_static = None
        self._gamma_dynamic = None
        self._prev_alpha = None

    def _init_reflex_weights(self) -> None:
        """Initialize reflex connections with diagonal structure."""
        n = self.config.num_muscles
        self.Ia_to_alpha.weight.data = torch.eye(n) * 0.3 + torch.randn(n, n) * 0.05
        self.II_to_alpha.weight.data = torch.eye(n) * 0.2 + torch.randn(n, n) * 0.05

    def compute_alpha_with_reflex(
        self,
        alpha_cortical: torch.Tensor,
        type_Ia: torch.Tensor,
        type_II: torch.Tensor,
    ) -> torch.Tensor:
        """Combine cortical alpha with reflex contributions."""
        return torch.sigmoid(
            alpha_cortical + self.Ia_to_alpha(type_Ia) + self.II_to_alpha(type_II)
        )

    def parse_input(
        self, obs: Union[torch.Tensor, ObservationTensor]
    ) -> Tuple[ProprioceptionTensor, torch.Tensor]:
        """Parse input to structured proprioception and encoded target."""
        if isinstance(obs, ObservationTensor):
            return obs.proprio, self.target_encoder.encode(obs.target)

        # Flat tensor - parse by position
        n = self.config.num_muscles
        proprio = ProprioceptionTensor(
            lengths=obs[:, :n],
            velocities=obs[:, n : 2 * n],
            forces=obs[:, 2 * n : 3 * n],
        )
        target_xyz = obs[:, 3 * n : 3 * n + 3]
        return proprio, self.target_encoder.encode(target_xyz)


class RNNController(ControllerBase):
    """RNN-based controller for muscle-driven arm control."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        rnn_input_dim = (
            config.num_sensors + config.num_target_units + config.num_muscles
        )
        self.core = RNNCore(
            input_size=rnn_input_dim,
            hidden_size=config.rnn_hidden_size,
            rnn_type=config.rnn_type,
            num_layers=config.num_rnn_layers,
            input_projection_bias=config.input_projection_bias,
        )
        self._hidden = None

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
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
        batch_size = (
            obs.batch_size if isinstance(obs, ObservationTensor) else obs.shape[0]
        )
        device = obs.device if isinstance(obs, ObservationTensor) else obs.device

        if hidden is None:
            if self._hidden is None:
                self.init_hidden(batch_size, device)
            hidden = self._hidden

        if prev_alpha is None:
            prev_alpha = (
                self._prev_alpha
                if self._prev_alpha is not None
                else torch.zeros(batch_size, self.config.num_muscles, device=device)
            )

        gamma_static = (
            self._gamma_static
            if self._gamma_static is not None
            else torch.ones(batch_size, self.config.num_muscles, device=device)
        )
        gamma_dynamic = (
            self._gamma_dynamic
            if self._gamma_dynamic is not None
            else torch.ones(batch_size, self.config.num_muscles, device=device)
        )

        proprio, target_encoded = self.parse_input(obs)

        sensory_out, sensory_outputs = self.sensory(
            proprio.lengths,
            proprio.velocities,
            proprio.forces,
            gamma_static,
            gamma_dynamic,
        )

        rnn_input = torch.cat([sensory_out, target_encoded, prev_alpha], dim=1)
        rnn_output, new_hidden = self.core(rnn_input, hidden)

        alpha_cortical, gamma_static_new, gamma_dynamic_new = self.motor(rnn_output)
        alpha = self.compute_alpha_with_reflex(
            alpha_cortical, sensory_outputs["type_Ia"], sensory_outputs["type_II"]
        )

        # Update state
        self._hidden = new_hidden
        self._prev_alpha = alpha.detach()
        self._gamma_static = gamma_static_new.detach()
        self._gamma_dynamic = gamma_dynamic_new.detach()

        return (
            alpha,
            new_hidden,
            {
                "alpha": alpha,
                "alpha_cortical": alpha_cortical,
                "gamma_static": gamma_static_new,
                "gamma_dynamic": gamma_dynamic_new,
                "sensory_outputs": sensory_outputs,
                "rnn_hidden": rnn_output,
            },
        )

    def forward_sequence(
        self, obs_sequence: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Forward pass for sequence of observations."""
        batch_size, seq_len, _ = obs_sequence.shape
        device = obs_sequence.device

        if hidden is None:
            self.init_hidden(batch_size, device)
            hidden = self._hidden

        actions, infos = [], []
        prev_alpha = torch.zeros(batch_size, self.config.num_muscles, device=device)

        for t in range(seq_len):
            action, hidden, info = self.forward(
                obs_sequence[:, t, :], hidden, prev_alpha
            )
            actions.append(action)
            infos.append(info)
            prev_alpha = info["alpha"].detach()

        return torch.stack(actions, dim=1), hidden, infos


class MLPController(ControllerBase):
    """MLP-based controller (no recurrence). Used as teacher in distillation."""

    def __init__(self, config: ModelConfig, hidden_sizes: Optional[List[int]] = None):
        super().__init__(config)

        if hidden_sizes is None:
            hidden_sizes = config.mlp_hidden_sizes

        mlp_input_dim = config.num_sensors + config.num_target_units
        self.core = MLPCore(mlp_input_dim, config.rnn_hidden_size, hidden_sizes)

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
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
        batch_size = (
            obs.batch_size if isinstance(obs, ObservationTensor) else obs.shape[0]
        )
        device = obs.device if isinstance(obs, ObservationTensor) else obs.device

        gamma_static = torch.ones(batch_size, self.config.num_muscles, device=device)
        gamma_dynamic = torch.ones(batch_size, self.config.num_muscles, device=device)

        proprio, target_encoded = self.parse_input(obs)

        sensory_out, sensory_outputs = self.sensory(
            proprio.lengths,
            proprio.velocities,
            proprio.forces,
            gamma_static,
            gamma_dynamic,
        )

        mlp_output = self.core(torch.cat([sensory_out, target_encoded], dim=1))

        alpha_cortical, gamma_static_new, gamma_dynamic_new = self.motor(mlp_output)
        alpha = self.compute_alpha_with_reflex(
            alpha_cortical, sensory_outputs["type_Ia"], sensory_outputs["type_II"]
        )

        return (
            alpha,
            None,
            {
                "alpha": alpha,
                "alpha_cortical": alpha_cortical,
                "gamma_static": gamma_static_new,
                "gamma_dynamic": gamma_dynamic_new,
                "sensory_outputs": sensory_outputs,
            },
        )


def create_controller(
    config: ModelConfig, controller_type: str = "rnn", **kwargs
) -> BaseController:
    """Factory function to create controllers."""
    controllers = {"rnn": RNNController, "mlp": MLPController}
    cls = controllers.get(controller_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown controller type: {controller_type}. Expected 'rnn' or 'mlp'."
        )
    return cls(config, **kwargs) if controller_type.lower() == "mlp" else cls(config)
