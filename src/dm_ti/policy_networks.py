import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from networks import LeakyRNN, ModalitySpecificEncoder


class RecurrentActorNetwork(nn.Module):
    """
    Recurrent actor network for continuous action PPO.
    Uses population encoding and custom LeakyRNN.
    """
    def __init__(
        self, 
        obs_shape: Sequence[int], 
        action_shape: Sequence[int], 
        hidden_size: int = 128,
        num_layers: int = 1,
        device: Union[str, torch.device] = "cpu",
        debug: bool = False,
        encoder=None
    ):
        super().__init__()
        
        # Get input and output dimensions
        self.obs_dim = obs_shape[0] if len(obs_shape) == 1 else obs_shape
        self.action_dim = action_shape[0] if len(action_shape) == 1 else action_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.debug = debug
        
        # Add max_action attribute required by Tianshou for action_scaling
        # Since we use tanh activation, output is in [-1, 1] range
        self.max_action = 1.0  
        
        # Input encoder - processes raw observations into neural representations
        self.encoder = encoder or ModalitySpecificEncoder(target_size=40)
        encoded_dim = self.encoder.output_size
        
        # RNN layer - processes temporal sequences
        self.rnn = LeakyRNN(
            input_size=encoded_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            tau_mem=50.0,
            use_layer_norm=True,
            debug=debug
        )
        
        # Actor head - outputs mean of action distribution
        self.mu = nn.Linear(hidden_size, self.action_dim)
        
        # Log standard deviation - learnable but not state-dependent
        self.sigma = nn.Parameter(torch.zeros(self.action_dim))
        self.sigma_min = -20  # Prevent sigma from going too negative
  
        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[torch.Tensor] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the actor network.
        
        Args:
            obs: Observations, shape [batch_size, *obs_shape] or [batch_size, seq_len, *obs_shape]
            state: RNN hidden state (3D format: [num_layers, batch, hidden]) or None
            info: Additional info dict
            
        Returns:
            Tuple of ((mu, sigma), new_state)
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._get_device())
        
        # Check if input already has sequence dimension
        has_seq_dim = len(obs.shape) > 2
        batch_size = obs.shape[0]
        
        if has_seq_dim:
            # Handle batched sequence input [batch_size, seq_len, obs_dim]
            seq_len = obs.shape[1]
            obs_reshaped = obs.reshape(batch_size * seq_len, -1)
            encoded = self.encoder(obs_reshaped)
            encoded = encoded.reshape(batch_size, seq_len, -1)
        else:
            # Single timestep - add sequence dimension
            encoded = self.encoder(obs)
            encoded = encoded.unsqueeze(1)  # [batch, 1, encoded_dim]
        
        # If we got a batch-first state [batch, num_layers, hidden], invert it:
        if state is not None and state.dim() == 3:
            # state: [batch, num_layers, hidden] → [num_layers, batch, hidden]
            state = state.transpose(0, 1)

        # Now call the RNN with the correctly shaped `state`
        rnn_out, h_n = self.rnn(encoded, state)  # h_n: [num_layers, batch, hidden]
        # Convert to batch-first: [batch, num_layers, hidden]
        h_n = h_n.transpose(0, 1)
        
        # Get output from the last timestep
        if has_seq_dim:
            last_output = rnn_out[:, -1]  # [batch, hidden]
        else:
            last_output = rnn_out.squeeze(1)  # [batch, hidden]
            
        # Action distribution parameters
        mu = self.mu(last_output)
        # Use softplus instead of exp for bounded positive values
        sigma = torch.nn.functional.softplus(self.sigma).expand_as(mu) + 1e-3

        # Check for NaNs - stabilization measure
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("WARNING: NaN detected in action mean - stabilizing")
            mu = torch.zeros_like(mu)
            
            # Reset RNN state when NaNs appear
            if hasattr(self, 'last_state') and self.last_state is not None:
                print("Resetting RNN state due to NaNs")
                self.last_state = torch.zeros_like(self.last_state)

        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            print("WARNING: NaN detected in action std - stabilizing")
            sigma = torch.ones_like(sigma) * 0.1
        
        return (mu, sigma), h_n
    
    def _get_device(self):
        """Get the device of this module."""
        return next(self.parameters()).device
        
    def init_state(self, batch_size: int):
        """Initialize hidden state in batch-first format."""
        # Get [num_layers, batch, hidden] then transpose
        h0 = self.rnn.init_hidden(batch_size, device=self._get_device())
        return h0.transpose(0, 1)
    
    def reset_state(self, batch_size=None, done_env_ids=None, state=None):
        """
        Reset RNN hidden state for Tianshou compatibility.
        Now expects state shape [batch, num_layers, hidden].
        """
        # If no state and no batch size, return None
        if batch_size is None and state is None:
            return None
            
        # Initialize new state if needed
        if state is None:
            return self.init_state(batch_size)
        
        # Handle done environments
        if done_env_ids:
            # Zero-out along batch dimension
            for env_id in done_env_ids:
                if env_id < state.size(0):
                    state[env_id].zero_()
        
        return state  # Return the state (potentially modified in-place)

    def forward_with_stimulation(self, obs, unit_idx=None, stimulation_strength=0.0, state=None):
        """
        Forward pass with optional unit stimulation.
        
        Args:
            obs: Observations
            unit_idx: Index of RNN unit to stimulate (None for no stimulation)
            stimulation_strength: Strength of stimulation to apply
            state: RNN hidden state
    
        Returns:
            Tuple of ((mu, sigma), new_state)
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._get_device())
    
        # Check if input already has sequence dimension
        has_seq_dim = len(obs.shape) > 2
        batch_size = obs.shape[0]
    
        if has_seq_dim:
            # Handle batched sequence input [batch_size, seq_len, obs_dim]
            seq_len = obs.shape[1]
            obs_reshaped = obs.reshape(batch_size * seq_len, -1)
            encoded = self.encoder(obs_reshaped)
            encoded = encoded.reshape(batch_size, seq_len, -1)
        else:
            # Single timestep - add sequence dimension
            encoded = self.encoder(obs)
            encoded = encoded.unsqueeze(1)  # [batch, 1, encoded_dim]
    
        # If we got a batch-first state [batch, num_layers, hidden], invert it:
        if state is not None and state.dim() == 3:
            # state: [batch, num_layers, hidden] → [num_layers, batch, hidden]
            state = state.transpose(0, 1)

        # Now call the RNN with the correctly shaped `state`
        rnn_out, h_n = self.rnn(encoded, state)  # h_n: [num_layers, batch, hidden]
    
        # Apply stimulation if specified
        if unit_idx is not None and stimulation_strength != 0.0:
            # Apply stimulation to the RNN output (not just final hidden state)
            rnn_out = rnn_out.clone()  # Don't modify original
            rnn_out[:, :, unit_idx] += stimulation_strength
        
            # Also stimulate the hidden state for consistency
            h_n = h_n.clone()
            h_n[0, :, unit_idx] += stimulation_strength  # First layer
    
        # Convert to batch-first: [batch, num_layers, hidden]
        h_n = h_n.transpose(0, 1)
    
        # Get output from the last timestep
        if has_seq_dim:
            last_output = rnn_out[:, -1]  # [batch, hidden]
        else:
            last_output = rnn_out.squeeze(1)  # [batch, hidden]
        
        # Action distribution parameters
        mu = self.mu(last_output)
        # Use softplus instead of exp for bounded positive values
        sigma = torch.nn.functional.softplus(self.sigma).expand_as(mu) + 1e-3

        # Check for NaNs - stabilization measure
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("WARNING: NaN detected in action mean - stabilizing")
            mu = torch.zeros_like(mu)

        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            print("WARNING: NaN detected in action std - stabilizing")
            sigma = torch.ones_like(sigma) * 0.1
    
        return (mu, sigma), h_n


class CriticNetwork(nn.Module):
    """
    Recurrent critic network for PPO with gradient isolation.
    """
    def __init__(
        self,
        obs_shape: Sequence[int],
        hidden_size: int = 128,
        num_layers: int = 1,
        device: Union[str, torch.device] = "cpu",
        encoder=None  # Add this parameter
    ):
        super().__init__()
        
        # Get input dimension
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else obs_shape
        
        # Use the provided encoder or create a new one
        self.encoder = encoder or ModalitySpecificEncoder(target_size=40)
        encoded_dim = self.encoder.output_size
        
        # Value head - directly processes encoded observations
        # Bypass RNN completely for the critic to avoid influencing RNN weights
        self.value_net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[torch.Tensor] = None,  # we ignore state here
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:               # <-- return type is now just Tensor
        # convert numpy → torch
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32,
                               device=next(self.parameters()).device)
        # handle sequences
        if obs.dim() > 2:
            obs = obs[:, -1]        # only last timestep
        encoded = self.encoder(obs)
        value = self.value_net(encoded)
        return value                
    
    def reset_state(self, *args, **kwargs):
        return None



