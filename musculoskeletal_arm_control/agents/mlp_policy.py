"""
MLP policy network for RL training.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class MLPPolicy(nn.Module):
    """Multi-layer perceptron policy for muscle control."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu"
    ):
        """
        Initialize MLP policy.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (number of muscles)
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        # Output layer (muscle activations)
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Sigmoid())  # Constrain to [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Value network for critic (for PPO)
        value_layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            value_layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                value_layers.append(nn.ReLU())
            elif activation == "tanh":
                value_layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        value_layers.append(nn.Linear(input_dim, 1))
        self.value_network = nn.Sequential(*value_layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get action.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            
        Returns:
            action: Actions (batch_size, action_dim)
        """
        return self.network(obs)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for state.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            
        Returns:
            value: Value estimates (batch_size, 1)
        """
        return self.value_network(obs)
    
    def get_action_and_value(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both action and value estimate.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            
        Returns:
            action: Actions (batch_size, action_dim)
            value: Value estimates (batch_size, 1)
        """
        action = self.forward(obs)
        value = self.get_value(obs)
        return action, value
    
    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action from observation (numpy interface).
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Action array
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.forward(obs_tensor)
        return action.squeeze(0).cpu().numpy()
