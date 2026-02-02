"""
Neural Network Modules for Muscle-Driven Arm Control

Modules:
- TargetEncodingModule: Grid of Gaussian-tuned spatial units
- ProprioceptiveModule: Ia, II, Ib sensory neurons (one per muscle)
- OutputModule: Alpha MNs, Gamma static, Gamma dynamic
- RNNController: Main recurrent neural network
- MLPController: Feedforward MLP (for teacher in distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

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
class ModelConfig:
    """Configuration for the neural network architecture."""

    num_muscles: int
    num_sensors: int  # Total sensor inputs (e.g., length + velocity + force per muscle)
    num_target_units: int  # Grid size^2 for target encoding

    # Network dimensions - only RNN has hidden recurrent layer
    rnn_hidden_size: int = 32

    # RNN settings
    rnn_type: str = "rnn"  # 'rnn', 'gru', 'lstm'
    num_rnn_layers: int = 1

    # Bias settings for each module
    proprioceptive_bias: bool = True  # Bias in Ia, II, Ib sensory neurons
    target_encoding_bias: bool = True  # Bias in target grid layer
    input_projection_bias: bool = True  # Bias in RNN input projection
    output_bias: bool = True  # Bias in alpha/gamma output heads
    reflex_bias: bool = False  # Bias in Ia->Alpha, II->Alpha (typically False)

    @property
    def input_size(self) -> int:
        """Total observation dimension (sensors + target grid)."""
        return self.num_sensors + self.num_target_units


"""
.########....###....########...######...########.########
....##......##.##...##.....##.##....##..##..........##...
....##.....##...##..##.....##.##........##..........##...
....##....##.....##.########..##...####.######......##...
....##....#########.##...##...##....##..##..........##...
....##....##.....##.##....##..##....##..##..........##...
....##....##.....##.##.....##..######...########....##...
"""


class TargetEncodingModule(nn.Module):
    """
    Target encoding module using grid of Gaussian-tuned spatial units.

    The input is already encoded as Gaussian activations from the environment.
    This module applies a single ReLU layer (no hidden layers).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Single layer: target grid -> same size output with ReLU
        self.fc = nn.Linear(
            config.num_target_units,
            config.num_target_units,
            bias=config.target_encoding_bias,
        )

    def forward(self, target_encoding: torch.Tensor) -> torch.Tensor:
        """
        Process target encoding.

        Args:
            target_encoding: Gaussian-tuned grid activations [batch, num_target_units]

        Returns:
            Processed target features [batch, num_target_units]
        """
        return F.relu(self.fc(target_encoding))


"""
..######..########.##....##..######...#######..########..##....##
.##....##.##.......###...##.##....##.##.....##.##.....##..##..##.
.##.......##.......####..##.##.......##.....##.##.....##...####..
..######..######...##.##.##..######..##.....##.########.....##...
.......##.##.......##..####.......##.##.....##.##...##......##...
.##....##.##.......##...###.##....##.##.....##.##....##.....##...
..######..########.##....##..######...#######..##.....##....##...
"""


class SensoryModule(nn.Module):
    """
    Proprioceptive sensory neuron module.

    Implements:
    - Type Ia neurons: respond to muscle velocity (dynamic)
    - Type II neurons: respond to muscle length (static)
    - Type Ib neurons: respond to muscle force (Golgi tendon organs)

    Each is a single ReLU unit per muscle (no hidden layers).

    Gamma motor neurons modulate spindle sensitivity:
    - Gamma static: modulates Type II response to length
    - Gamma dynamic: modulates Type Ia response to velocity
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        num_muscles = config.num_muscles
        use_bias = config.proprioceptive_bias

        # Sensory neuron layers (one per muscle, per type) - single layer ReLU
        # Type Ia: velocity -> ReLU (1 input, 1 output per muscle)
        self.type_Ia = nn.ModuleList(
            [nn.Linear(1, 1, bias=use_bias) for _ in range(num_muscles)]
        )

        # Type II: length -> ReLU (1 input, 1 output per muscle)
        self.type_II = nn.ModuleList(
            [nn.Linear(1, 1, bias=use_bias) for _ in range(num_muscles)]
        )

        # Type Ib: force -> ReLU (Golgi tendon organ, 1 input, 1 output per muscle)
        self.type_Ib = nn.ModuleList(
            [nn.Linear(1, 1, bias=use_bias) for _ in range(num_muscles)]
        )

        # Initialize with biologically plausible weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with biologically plausible values."""
        for i in range(self.config.num_muscles):
            # Type Ia: positive response to velocity
            nn.init.uniform_(self.type_Ia[i].weight, 0.5, 1.5)
            if self.type_Ia[i].bias is not None:
                nn.init.zeros_(self.type_Ia[i].bias)

            # Type II: positive response to length
            nn.init.uniform_(self.type_II[i].weight, 0.5, 1.5)
            if self.type_II[i].bias is not None:
                nn.init.zeros_(self.type_II[i].bias)

            # Type Ib: positive response to force
            nn.init.uniform_(self.type_Ib[i].weight, 0.5, 1.5)
            if self.type_Ib[i].bias is not None:
                nn.init.zeros_(self.type_Ib[i].bias)

    def forward(
        self,
        lengths: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        gamma_static: Optional[torch.Tensor] = None,
        gamma_dynamic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process proprioceptive inputs through sensory neurons.

        Args:
            lengths: Muscle lengths [batch, num_muscles]
            velocities: Muscle velocities [batch, num_muscles]
            forces: Muscle forces [batch, num_muscles]
            gamma_static: Gamma static modulation [batch, num_muscles]
            gamma_dynamic: Gamma dynamic modulation [batch, num_muscles]

        Returns:
            concatenated: All sensory outputs concatenated [batch, n_muscles * 3]
            sensory_outputs: Dictionary of individual sensory outputs
        """
        batch_size = lengths.shape[0]

        if gamma_static is None:
            gamma_static = torch.ones_like(lengths)
        if gamma_dynamic is None:
            gamma_dynamic = torch.ones_like(velocities)

        type_Ia_outputs = []
        type_II_outputs = []
        type_Ib_outputs = []

        for i in range(self.config.num_muscles):
            # Get inputs for this muscle
            length_i = lengths[:, i : i + 1]
            velocity_i = velocities[:, i : i + 1]
            force_i = forces[:, i : i + 1]

            # Apply gamma modulation before sensory processing
            modulated_length = length_i * gamma_static[:, i : i + 1]
            modulated_velocity = velocity_i * gamma_dynamic[:, i : i + 1]

            # Type Ia: responds to velocity only (modulated by gamma dynamic)
            Ia_output = F.relu(self.type_Ia[i](modulated_velocity))
            type_Ia_outputs.append(Ia_output)

            # Type II: responds to length only (modulated by gamma static)
            II_output = F.relu(self.type_II[i](modulated_length))
            type_II_outputs.append(II_output)

            # Type Ib: responds to force (not modulated by gamma)
            Ib_output = F.relu(self.type_Ib[i](force_i))
            type_Ib_outputs.append(Ib_output)

        # Concatenate all sensory outputs
        type_Ia_all = torch.cat(type_Ia_outputs, dim=1)  # [batch, num_muscles]
        type_II_all = torch.cat(type_II_outputs, dim=1)  # [batch, num_muscles]
        type_Ib_all = torch.cat(type_Ib_outputs, dim=1)  # [batch, num_muscles]

        # Concatenate for output (no hidden layer)
        concatenated = torch.cat([type_Ia_all, type_II_all, type_Ib_all], dim=1)

        sensory_outputs = {
            "type_Ia": type_Ia_all,
            "type_II": type_II_all,
            "type_Ib": type_Ib_all,
        }

        return concatenated, sensory_outputs


"""
..#######..##.....##.########.########..##.....##.########
.##.....##.##.....##....##....##.....##.##.....##....##...
.##.....##.##.....##....##....##.....##.##.....##....##...
.##.....##.##.....##....##....########..##.....##....##...
.##.....##.##.....##....##....##........##.....##....##...
.##.....##.##.....##....##....##........##.....##....##...
..#######...#######.....##....##.........#######.....##...
"""


class OutputModule(nn.Module):
    """
    Output module producing motor commands.

    Single ReLU layer for each output type (no hidden layers):
    - Alpha motor neurons: direct muscle activations
    - Gamma static: modulates Type II (length) sensory responses
    - Gamma dynamic: modulates Type Ia (velocity) sensory responses

    Alpha MNs receive direct projections from Ia and II sensory neurons.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Separate single-layer output heads (ReLU applied in forward)
        self.alpha_head = nn.Linear(
            config.rnn_hidden_size, config.num_muscles, bias=config.output_bias
        )
        self.gamma_static_head = nn.Linear(
            config.rnn_hidden_size,
            config.num_muscles,
            bias=config.output_bias,
        )
        self.gamma_dynamic_head = nn.Linear(
            config.rnn_hidden_size,
            config.num_muscles,
            bias=config.output_bias,
        )

        # Direct Ia/II -> Alpha connections (stretch reflex)
        self.Ia_to_alpha = nn.Linear(
            config.num_muscles, config.num_muscles, bias=config.reflex_bias
        )
        self.II_to_alpha = nn.Linear(
            config.num_muscles, config.num_muscles, bias=config.reflex_bias
        )

        self._init_reflex_weights()

    def _init_reflex_weights(self):
        """Initialize reflex connections with diagonal structure."""
        # Initialize with near-diagonal structure (each muscle's Ia/II projects mainly to same alpha MN)
        n = self.config.num_muscles
        Ia_weight = torch.eye(n) * 0.3 + torch.randn(n, n) * 0.05
        II_weight = torch.eye(n) * 0.2 + torch.randn(n, n) * 0.05

        self.Ia_to_alpha.weight.data = Ia_weight
        self.II_to_alpha.weight.data = II_weight

    def forward(
        self, rnn_hidden: torch.Tensor, type_Ia: torch.Tensor, type_II: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate motor outputs.

        Args:
            rnn_hidden: RNN hidden state [batch, rnn_hidden_size]
            type_Ia: Type Ia sensory neuron activations [batch, num_muscles]
            type_II: Type II sensory neuron activations [batch, num_muscles]

        Returns:
            alpha: Alpha MN activations [batch, num_muscles]
            gamma_static: Gamma static outputs [batch, num_muscles]
            gamma_dynamic: Gamma dynamic outputs [batch, num_muscles]
        """
        # Generate outputs with ReLU (single layer each)
        alpha_cortical = F.relu(self.alpha_head(rnn_hidden))
        gamma_static = (
            F.relu(self.gamma_static_head(rnn_hidden)) * 2.0
        )  # Scale to 0-2 range
        gamma_dynamic = F.relu(self.gamma_dynamic_head(rnn_hidden)) * 2.0

        # Add direct reflex pathways to alpha
        alpha_reflex_Ia = self.Ia_to_alpha(type_Ia)
        alpha_reflex_II = self.II_to_alpha(type_II)

        # Combine cortical and reflex contributions, then sigmoid to 0-1
        alpha = torch.sigmoid(alpha_cortical + alpha_reflex_Ia + alpha_reflex_II)

        return alpha, gamma_static, gamma_dynamic


"""
..######..########..####.##....##....###....##......
.##....##.##.....##..##..###...##...##.##...##......
.##.......##.....##..##..####..##..##...##..##......
..######..########...##..##.##.##.##.....##.##......
.......##.##.........##..##..####.#########.##......
.##....##.##.........##..##...###.##.....##.##......
..######..##........####.##....##.##.....##.########
"""


class RNNController(nn.Module):
    """
    Main RNN controller integrating all modules.

    Architecture:
    1. Proprioceptive module processes muscle sensors (single layer per type)
    2. Target encoding module processes target grid (single layer)
    3. RNN integrates over time (the only recurrent hidden layer)
    4. Output module generates motor commands (single layer per output)
    5. Alpha MNs project back to RNN

    Note: No explicit time or phase encoding - the RNN learns temporal
    structure through its recurrent dynamics.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Sub-modules (all single-layer, no hidden)
        self.proprioceptive = SensoryModule(config)
        self.target_encoding = TargetEncodingModule(config)
        self.output_module = OutputModule(config)

        # Input projection to RNN
        # Includes: proprioceptive outputs (num_sensors for Ia, II, Ib),
        #           target encoding, and alpha feedback
        rnn_input_dim = (
            config.num_sensors  # Ia, II, Ib outputs (one per muscle each)
            + config.num_target_units  # Target encoding output
            + config.num_muscles  # Alpha MN feedback
        )

        self.input_projection = nn.Linear(
            rnn_input_dim, config.rnn_hidden_size, bias=config.input_projection_bias
        )

        # RNN - the only recurrent hidden layer
        if config.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=config.rnn_hidden_size,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.num_rnn_layers,
                batch_first=True,
                nonlinearity="relu",
            )
        elif config.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=config.rnn_hidden_size,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.num_rnn_layers,
                batch_first=True,
            )
        elif config.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=config.rnn_hidden_size,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.num_rnn_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {config.rnn_type}")

        # For tracking hidden state across steps
        self._hidden = None
        self._prev_alpha = None
        self._gamma_static = None
        self._gamma_dynamic = None

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Initialize hidden state for new episodes."""
        if self.config.rnn_type == "lstm":
            self._hidden = (
                torch.zeros(
                    self.config.num_rnn_layers,
                    batch_size,
                    self.config.rnn_hidden_size,
                    device=device,
                ),
                torch.zeros(
                    self.config.num_rnn_layers,
                    batch_size,
                    self.config.rnn_hidden_size,
                    device=device,
                ),
            )
        else:
            self._hidden = torch.zeros(
                self.config.num_rnn_layers,
                batch_size,
                self.config.rnn_hidden_size,
                device=device,
            )

        self._prev_alpha = torch.zeros(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_static = torch.ones(
            batch_size, self.config.num_muscles, device=device
        )
        self._gamma_dynamic = torch.ones(
            batch_size, self.config.num_muscles, device=device
        )

    def parse_observation(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse flat observation into components.

        Assumes sensors are ordered as: [lengths, velocities, forces] for all muscles,
        followed by target encoding.
        """
        num_muscles = self.config.num_muscles

        # Proprioceptive: assume length, velocity, force per muscle
        # TODO: Make this more flexible based on actual sensor configuration
        lengths = obs[:, 0:num_muscles]
        velocities = obs[:, num_muscles : 2 * num_muscles]
        forces = obs[:, 2 * num_muscles : 3 * num_muscles]

        # Target encoding starts after sensors
        target_start = self.config.num_sensors
        target_encoding = obs[
            :, target_start : target_start + self.config.num_target_units
        ]

        return {
            "lengths": lengths,
            "velocities": velocities,
            "forces": forces,
            "target_encoding": target_encoding,
        }

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        prev_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for single timestep.

        Args:
            obs: Observation [batch, obs_dim]
            hidden: Optional hidden state
            prev_alpha: Previous alpha MN output for feedback

        Returns:
            action: Alpha MN activations [batch, num_muscles]
            new_hidden: Updated hidden state
            info: Dictionary of intermediate outputs
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Use stored hidden if not provided
        if hidden is None:
            if self._hidden is None:
                self.init_hidden(batch_size, device)
            hidden = self._hidden

        if prev_alpha is None:
            if self._prev_alpha is None:
                self._prev_alpha = torch.zeros(
                    batch_size, self.config.num_alpha_outputs, device=device
                )
            prev_alpha = self._prev_alpha

        # Get gamma values from previous step
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

        # Parse observation
        parsed = self.parse_observation(obs)

        # Process proprioceptive input (with gamma modulation from previous step)
        proprio_out, sensory_outputs = self.proprioceptive(
            parsed["lengths"],
            parsed["velocities"],
            parsed["forces"],
            gamma_static,
            gamma_dynamic,
        )

        # Process target encoding
        target_out = self.target_encoding(parsed["target_encoding"])

        # Combine inputs for RNN (no phase or time - RNN learns temporal dynamics)
        rnn_input = torch.cat(
            [
                proprio_out,  # [batch, n_muscles * 3]
                target_out,  # [batch, n_target_units]
                prev_alpha,  # [batch, num_muscles] - Alpha MN feedback
            ],
            dim=1,
        )

        # Project to RNN dimension
        rnn_input = F.relu(self.input_projection(rnn_input))
        rnn_input = rnn_input.unsqueeze(1)  # Add time dimension

        # RNN forward
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        rnn_output = rnn_output.squeeze(1)  # Remove time dimension

        # Generate outputs
        alpha, gamma_static_new, gamma_dynamic_new = self.output_module(
            rnn_output, sensory_outputs["type_Ia"], sensory_outputs["type_II"]
        )

        # Update stored state
        self._hidden = new_hidden
        self._prev_alpha = alpha.detach()
        self._gamma_static = gamma_static_new.detach()
        self._gamma_dynamic = gamma_dynamic_new.detach()

        # Action for environment is only alpha MNs
        action = alpha

        info = {
            "alpha": alpha,
            "gamma_static": gamma_static_new,
            "gamma_dynamic": gamma_dynamic_new,
            "sensory_outputs": sensory_outputs,
            "rnn_hidden": rnn_output,
        }

        return action, new_hidden, info

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
        prev_alpha = torch.zeros(
            batch_size, self.config.num_alpha_outputs, device=device
        )

        for t in range(seq_len):
            obs_t = obs_sequence[:, t, :]
            action, hidden, info = self.forward(obs_t, hidden, prev_alpha)
            actions.append(action)
            infos.append(info)
            prev_alpha = info["alpha"].detach()

        actions = torch.stack(actions, dim=1)

        return actions, hidden, infos

    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as a flat numpy array (for CMA-ES)."""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat numpy array (for CMA-ES)."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat_params[idx : idx + size].reshape(p.shape),
                dtype=p.dtype,
                device=p.device,
            )
            idx += size

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


"""
.##.....##.##.......########.
.###...###.##.......##.....##
.####.####.##.......##.....##
.##.###.##.##.......########.
.##.....##.##.......##.......
.##.....##.##.......##.......
.##.....##.########.##.......
"""


class MLPController(nn.Module):
    """
    Feedforward MLP controller (for teacher in distillation learning).

    Same input/output interface as RNNController but without recurrence.
    Uses larger hidden layers to compensate for lack of memory.
    """

    def __init__(self, config: ModelConfig, hidden_sizes: List[int] = None):
        super().__init__()
        self.config = config

        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        # Sub-modules (same as RNN - single layer each)
        self.proprioceptive = SensoryModule(config)
        self.target_encoding = TargetEncodingModule(config)

        # MLP input dimension (no phase/time)
        mlp_input_dim = (
            config.num_muscles * 3  # Proprioceptive outputs
            + config.num_target_units  # Target encoding output
        )

        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output heads (single layer each)
        self.alpha_head = nn.Linear(
            prev_dim, config.num_alpha_outputs, bias=config.output_bias
        )
        self.gamma_static_head = nn.Linear(
            prev_dim, config.num_gamma_static_outputs, bias=config.output_bias
        )
        self.gamma_dynamic_head = nn.Linear(
            prev_dim, config.num_gamma_dynamic_outputs, bias=config.output_bias
        )

        # Direct reflex connections
        self.Ia_to_alpha = nn.Linear(
            config.num_muscles, config.num_alpha_outputs, bias=config.reflex_bias
        )
        self.II_to_alpha = nn.Linear(
            config.num_muscles, config.num_alpha_outputs, bias=config.reflex_bias
        )

        self._init_reflex_weights()

    def _init_reflex_weights(self):
        """Initialize reflex connections."""
        n = self.config.num_muscles
        self.Ia_to_alpha.weight.data = torch.eye(n) * 0.3 + torch.randn(n, n) * 0.05
        self.II_to_alpha.weight.data = torch.eye(n) * 0.2 + torch.randn(n, n) * 0.05

    def parse_observation(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse flat observation into components."""
        num_muscles = self.config.num_muscles

        # Proprioceptive: assume length, velocity, force per muscle
        lengths = obs[:, 0:num_muscles]
        velocities = obs[:, num_muscles : 2 * num_muscles]
        forces = obs[:, 2 * num_muscles : 3 * num_muscles]

        # Target encoding starts after sensors
        target_start = self.config.num_sensors
        target_encoding = obs[
            :, target_start : target_start + self.config.num_target_units
        ]

        return {
            "lengths": lengths,
            "velocities": velocities,
            "forces": forces,
            "target_encoding": target_encoding,
        }

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        prev_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            obs: Observation [batch, obs_dim]
            hidden: Ignored (for interface compatibility)
            prev_alpha: Ignored

        Returns:
            action: Alpha MN activations [batch, num_muscles]
            None: (no hidden state)
            info: Dictionary of intermediate outputs
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Parse observation
        parsed = self.parse_observation(obs)

        # Process proprioceptive input (no gamma modulation for MLP - uses default 1.0)
        gamma_static = torch.ones(batch_size, self.config.num_muscles, device=device)
        gamma_dynamic = torch.ones(batch_size, self.config.num_muscles, device=device)

        proprio_out, sensory_outputs = self.proprioceptive(
            parsed["lengths"],
            parsed["velocities"],
            parsed["forces"],
            gamma_static,
            gamma_dynamic,
        )

        # Process target encoding
        target_out = self.target_encoding(parsed["target_encoding"])

        # Combine inputs (no phase/time)
        mlp_input = torch.cat(
            [
                proprio_out,
                target_out,
            ],
            dim=1,
        )

        # MLP forward
        hidden_out = self.mlp(mlp_input)

        # Generate outputs with ReLU
        alpha_cortical = F.relu(self.alpha_head(hidden_out))
        gamma_static = F.relu(self.gamma_static_head(hidden_out)) * 2.0
        gamma_dynamic = F.relu(self.gamma_dynamic_head(hidden_out)) * 2.0

        # Add reflex pathways
        alpha_reflex_Ia = self.Ia_to_alpha(sensory_outputs["type_Ia"])
        alpha_reflex_II = self.II_to_alpha(sensory_outputs["type_II"])
        alpha = torch.sigmoid(alpha_cortical + alpha_reflex_Ia + alpha_reflex_II)

        # Action for environment is only alpha MNs
        action = alpha

        info = {
            "alpha": alpha,
            "gamma_static": gamma_static,
            "gamma_dynamic": gamma_dynamic,
            "sensory_outputs": sensory_outputs,
        }

        return action, None, info

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """No-op for interface compatibility."""
        pass

    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as a flat numpy array."""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat numpy array."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat_params[idx : idx + size].reshape(p.shape),
                dtype=p.dtype,
                device=p.device,
            )
            idx += size

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_controller(
    config: ModelConfig, controller_type: str = "rnn", **kwargs
) -> nn.Module:
    """
    Factory function to create a controller.

    Args:
        config: Model configuration
        controller_type: 'rnn' or 'mlp'
        **kwargs: Additional arguments for the controller

    Returns:
        Controller module
    """
    if controller_type == "rnn":
        return RNNController(config)
    elif controller_type == "mlp":
        return MLPController(config, **kwargs)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


if __name__ == "__main__":
    # Test the modules
    config = ModelConfig(
        n_muscles=4,
        n_joints=2,
        n_target_units=25,  # 5x5 grid
        rnn_hidden_size=128,
    )

    print(f"Config:")
    print(f"  n_muscles: {config.num_muscles}")
    print(f"  n_joints: {config.n_joints}")
    print(f"  proprioceptive_input_dim: {config.proprioceptive_input_dim}")
    print(f"  total_input_dim: {config.input_size}")
    print(f"  total_output_dim: {config.total_output_dim}")

    # Create RNN controller
    rnn = RNNController(config)
    print(f"\nRNN Controller:")
    print(f"  Total parameters: {rnn.count_parameters()}")

    # Test forward pass
    batch_size = 8
    obs = torch.randn(batch_size, config.input_size)

    rnn.init_hidden(batch_size, obs.device)
    action, hidden, info = rnn.forward(obs)

    print(f"  Input shape: {obs.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Alpha shape: {info['alpha'].shape}")
    print(f"  Gamma static shape: {info['gamma_static'].shape}")

    # Test sequence forward
    seq_len = 10
    obs_seq = torch.randn(batch_size, seq_len, config.input_size)
    rnn.init_hidden(batch_size, obs_seq.device)
    actions, final_hidden, infos = rnn.forward_sequence(obs_seq)
    print(f"\n  Sequence input shape: {obs_seq.shape}")
    print(f"  Sequence action shape: {actions.shape}")

    # Create MLP controller
    mlp = MLPController(config)
    print(f"\nMLP Controller:")
    print(f"  Total parameters: {mlp.count_parameters()}")

    action_mlp, _, info_mlp = mlp.forward(obs)
    print(f"  Action shape: {action_mlp.shape}")

    # Test flat params
    flat = rnn.get_flat_params()
    print(f"\n  Flat params shape: {flat.shape}")
