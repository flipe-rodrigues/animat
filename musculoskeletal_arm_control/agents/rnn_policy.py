"""
RNN policy network with simple ReLU units.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SimpleRNNCell(nn.Module):
    """Simple RNN cell with ReLU activation."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize RNN cell.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        # Hidden to hidden (recurrent connection)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through RNN cell.
        
        Args:
            x: Input (batch_size, input_size)
            hidden: Previous hidden state (batch_size, hidden_size)
            
        Returns:
            new_hidden: New hidden state (batch_size, hidden_size)
        """
        if hidden is None:
            hidden = torch.zeros(
                x.size(0), self.hidden_size, device=x.device, dtype=x.dtype
            )
        
        # h_t = ReLU(W_ih @ x_t + W_hh @ h_{t-1})
        new_hidden = torch.relu(self.W_ih(x) + self.W_hh(hidden))
        
        return new_hidden


class RNNPolicy(nn.Module):
    """RNN policy with simple ReLU units for muscle control."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1
    ):
        """
        Initialize RNN policy.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (number of muscles)
            hidden_size: RNN hidden state size
            num_layers: Number of RNN layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn_cells = nn.ModuleList([
            SimpleRNNCell(
                input_size=obs_dim if i == 0 else hidden_size,
                hidden_size=hidden_size
            )
            for i in range(num_layers)
        ])
        
        # Output layer (muscle activations)
        self.output_layer = nn.Linear(hidden_size, action_dim)
        
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through RNN.
        
        Args:
            obs: Observations (batch_size, obs_dim) or (seq_len, batch_size, obs_dim)
            hidden: Tuple of hidden states for each layer
            
        Returns:
            action: Actions (batch_size, action_dim) or (seq_len, batch_size, action_dim)
            new_hidden: Updated hidden states
        """
        # Handle both single step and sequence inputs
        if obs.dim() == 2:
            # Single step: (batch_size, obs_dim)
            is_sequence = False
            obs = obs.unsqueeze(0)  # (1, batch_size, obs_dim)
        else:
            # Sequence: (seq_len, batch_size, obs_dim)
            is_sequence = True
        
        seq_len, batch_size, _ = obs.size()
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = tuple(
                torch.zeros(batch_size, self.hidden_size, device=obs.device)
                for _ in range(self.num_layers)
            )
        
        # Process sequence
        outputs = []
        new_hidden = list(hidden)
        
        for t in range(seq_len):
            x = obs[t]  # (batch_size, obs_dim)
            
            # Pass through RNN layers
            for i, rnn_cell in enumerate(self.rnn_cells):
                x = rnn_cell(x, new_hidden[i])
                new_hidden[i] = x
            
            # Generate output (muscle activations)
            output = torch.sigmoid(self.output_layer(x))
            outputs.append(output)
        
        # Stack outputs
        actions = torch.stack(outputs, dim=0)  # (seq_len, batch_size, action_dim)
        
        if not is_sequence:
            actions = actions.squeeze(0)  # (batch_size, action_dim)
        
        return actions, tuple(new_hidden)
    
    @torch.no_grad()
    def predict(
        self,
        obs: np.ndarray,
        hidden: Optional[Tuple[np.ndarray, ...]] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Predict action from observation (numpy interface).
        
        Args:
            obs: Observation array (obs_dim,)
            hidden: Tuple of hidden state arrays
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Action array (action_dim,)
            new_hidden: Updated hidden states
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        
        if hidden is not None:
            hidden_tensors = tuple(
                torch.FloatTensor(h).unsqueeze(0) for h in hidden
            )
        else:
            hidden_tensors = None
        
        action, new_hidden = self.forward(obs_tensor, hidden_tensors)
        
        action_np = action.squeeze(0).cpu().numpy()
        new_hidden_np = tuple(h.squeeze(0).cpu().numpy() for h in new_hidden)
        
        return action_np, new_hidden_np
    
    def reset_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
        """
        Reset hidden states to zeros.
        
        Args:
            batch_size: Batch size
            
        Returns:
            hidden: Tuple of zero hidden states
        """
        return tuple(
            torch.zeros(batch_size, self.hidden_size)
            for _ in range(self.num_layers)
        )
