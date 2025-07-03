import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from networks import ModalitySpecificEncoder

class GRUActorNetwork(nn.Module):
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
        
        # FIXED dimension handling
        self.obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else obs_shape
        self.action_dim = action_shape[0] if isinstance(action_shape, (tuple, list)) else action_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.debug = debug
        
        print(f"GRU Actor: obs_dim={self.obs_dim}, action_dim={self.action_dim}, hidden_size={hidden_size}")
        
        self.max_action = 1.0  
        
        self.encoder = encoder or ModalitySpecificEncoder(target_size=40)
        encoded_dim = self.encoder.output_size
        
        print(f"GRU Actor: encoded_dim={encoded_dim}")
        
        self.gru = nn.GRU(
            input_size=encoded_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )
        
        self.mu = nn.Linear(hidden_size, self.action_dim)
        self.sigma = nn.Parameter(torch.zeros(self.action_dim))
        
        self._init_weights()
        
    def _init_weights(self):
        # Conservative initialization
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        # Very conservative output layer
        nn.init.orthogonal_(self.mu.weight, gain=0.001)  # Even smaller gain
        nn.init.zeros_(self.mu.bias)
        
        # Conservative sigma
        nn.init.constant_(self.sigma, -2.0)  # exp(-2) ≈ 0.135
        
    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._get_device())
        
        batch_size = obs.shape[0]
        
        # Always treat as single timestep for evaluation
        if len(obs.shape) == 2:
            encoded = self.encoder(obs)
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
        else:
            # Handle sequence input
            seq_len = obs.shape[1]
            obs_reshaped = obs.reshape(batch_size * seq_len, -1)
            encoded = self.encoder(obs_reshaped)
            encoded = encoded.reshape(batch_size, seq_len, -1)
        
        # Handle GRU state conversion
        if state is not None and state.dim() == 3:
            state = state.transpose(0, 1).contiguous()
        
        gru_out, h_n = self.gru(encoded, state)
        h_n = h_n.transpose(0, 1).contiguous()
        
        # Get last output
        if len(obs.shape) == 2:
            last_output = gru_out.squeeze(1)
        else:
            last_output = gru_out[:, -1]
            
        mu = self.mu(last_output)
        sigma = F.softplus(self.sigma).expand_as(mu) + 1e-4
        sigma = torch.clamp(sigma, min=1e-4, max=1.0)  # Tighter bounds
        
        # Clamp mu to reasonable range
        mu = torch.clamp(mu, min=-2.0, max=2.0)
        
        if self.debug:
            print(f"GRU forward: mu range=[{mu.min():.3f}, {mu.max():.3f}], sigma range=[{sigma.min():.3f}, {sigma.max():.3f}]")
        
        return (mu, sigma), h_n
    
    def _get_device(self):
        return next(self.parameters()).device
        
    def init_state(self, batch_size: int):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self._get_device())
        return h0.transpose(0, 1).contiguous()
    
    def reset_state(self, batch_size=None, done_env_ids=None, state=None):
        if batch_size is None and state is None:
            return None
            
        if state is None:
            return self.init_state(batch_size)
        
        if done_env_ids:
            for env_id in done_env_ids:
                if env_id < state.size(0):
                    state[env_id].zero_()
        
        return state