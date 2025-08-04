import torch
import numpy as np
from networks.rnn import RNNPolicy

class ESRNNAdapter:
    """Adapter to make PyTorch RNN work with ES parameter interface."""
    
    def __init__(self, input_dim, action_dim, hidden_size=25, alpha=0.1):
        # Create PyTorch RNN using your modular RNN
        self.policy = RNNPolicy(
            input_dim=input_dim,
            action_dim=action_dim, 
            hidden_size=hidden_size,
            alpha=alpha
        )
        self.hidden_state = None
        self.num_params = self._count_parameters()
        
    def _count_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.policy.parameters())
    
    def get_params(self):
        """Get flattened parameters as numpy array."""
        params = []
        for param in self.policy.parameters():
            params.append(param.data.flatten())
        return torch.cat(params).numpy()
    
    def set_params(self, flat_params):
        """Set parameters from flattened numpy array."""
        flat_params = torch.FloatTensor(flat_params)
        idx = 0
        
        for param in self.policy.parameters():
            param_size = param.numel()
            param.data = flat_params[idx:idx + param_size].reshape(param.shape)
            idx += param_size
    
    def from_params(self, flat_params):
        """Create new RNN instance with given parameters."""
        new_adapter = ESRNNAdapter(
            input_dim=self.policy.input_dim,
            action_dim=self.policy.output_dim,
            hidden_size=self.policy.hidden_size,
            alpha=self.policy.alpha
        )
        new_adapter.set_params(flat_params)
        return new_adapter
    
    def init_state(self):
        """Initialize hidden state."""
        self.hidden_state = None
    
    def step(self, obs):
        """Forward pass through RNN."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, self.hidden_state = self.policy.forward(obs_tensor, self.hidden_state)
            return action.squeeze(0).numpy()


