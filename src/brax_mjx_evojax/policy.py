import jax
import jax.numpy as jnp
from evojax.policy.base import PolicyNetwork
from typing import Tuple, Any

class TanhRNNPolicy(PolicyNetwork):
    """A simple Tanh RNN policy for the 2-joint limb task."""
    
    def __init__(self, hidden_dim: int = 64):
        """Initialize the policy."""
        # Initialize the base class without parameters
        super().__init__()
        
        # Set dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = 15  # 12 sensors + 3D target position
        self.output_dim = 4  # 4 muscle activations
        
        # Define neural network parameters size
        self.params_size = (
            # input -> hidden
            self.input_dim * self.hidden_dim +
            # hidden -> hidden
            self.hidden_dim * self.hidden_dim +
            # hidden -> output
            self.hidden_dim * self.output_dim +
            # biases
            self.hidden_dim + self.output_dim
        )
    
    @property
    def input_shape(self) -> Tuple:
        """Input tensor shape."""
        return (self.input_dim,)
        
    @property
    def output_shape(self) -> Tuple:
        """Output tensor shape."""
        return (self.output_dim,)
    
    def init_states(self, batch_size: int) -> jnp.ndarray:
        """Initialize the RNN hidden states for a new batch."""
        # Create a jnp array to hold the hidden state
        return jnp.zeros((batch_size, self.hidden_dim))
    
    def reset_states(self, states: jnp.ndarray, done: jnp.ndarray) -> jnp.ndarray:
        """Reset the RNN hidden states based on done mask."""
        # Reset hidden states for completed episodes
        return states * (1.0 - done[:, None])
    
    def get_actions(self, params: jnp.ndarray, states: jnp.ndarray, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get actions for the given observations."""
        # Reshape parameters
        idx = 0
        
        # Input -> Hidden weights
        n_weights = self.input_dim * self.hidden_dim
        Wxh = params[idx:idx+n_weights].reshape(self.input_dim, self.hidden_dim)
        idx += n_weights
        
        # Hidden -> Hidden weights
        n_weights = self.hidden_dim * self.hidden_dim
        Whh = params[idx:idx+n_weights].reshape(self.hidden_dim, self.hidden_dim)
        idx += n_weights
        
        # Hidden -> Output weights
        n_weights = self.hidden_dim * self.output_dim
        Why = params[idx:idx+n_weights].reshape(self.hidden_dim, self.output_dim)
        idx += n_weights
        
        # Hidden bias
        n_bias = self.hidden_dim
        bh = params[idx:idx+n_bias]
        idx += n_bias
        
        # Output bias
        n_bias = self.output_dim
        by = params[idx:idx+n_bias]
        
        # Use current hidden state
        h_prev = states
        
        # RNN forward pass
        h_new = jnp.tanh(jnp.dot(obs, Wxh) + jnp.dot(h_prev, Whh) + bh)
        
        # Output layer
        y = jnp.tanh(jnp.dot(h_new, Why) + by)
        
        # Rescale from [-1, 1] to [0, 1] for muscle activations
        actions = (y + 1.0) / 2.0
        
        return actions, h_new