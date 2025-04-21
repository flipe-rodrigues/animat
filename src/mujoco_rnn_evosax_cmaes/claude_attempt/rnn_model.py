"""
Simple tanh RNN implementation for controlling the 2-joint 4-muscle arm.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

class SimpleRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a simple tanh RNN for muscle control.
        
        Args:
            input_size: Size of the input (sensors + target position)
            hidden_size: Size of the hidden layer
            output_size: Size of the output (muscle activations)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def init_params(self, key: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Initialize the RNN parameters.
        
        Args:
            key: JAX random key
        
        Returns:
            Dictionary containing the RNN parameters
        """
        # Split the random key for different parameter initializations
        # Adjust for JAX 0.5.2 if needed
        keys = jax.random.split(key, 4)
        k1, k2, k3, k4 = keys
        
        # Initialize weights with scaled random values
        scale = 0.1
        
        # Input to hidden weights - compatible with JAX 0.5.2
        w_ih = scale * jax.random.normal(k1, (self.hidden_size, self.input_size))
        
        # Hidden to hidden (recurrent) weights
        w_hh = scale * jax.random.normal(k2, (self.hidden_size, self.hidden_size))
        
        # Hidden to output weights
        w_ho = scale * jax.random.normal(k3, (self.output_size, self.hidden_size))
        
        # Biases
        b_h = scale * jax.random.normal(k4, (self.hidden_size,))
        b_o = jnp.zeros((self.output_size,))
        
        return {
            'w_ih': w_ih,
            'w_hh': w_hh, 
            'w_ho': w_ho,
            'b_h': b_h,
            'b_o': b_o
        }
    
    def predict(self, params: Dict[str, jnp.ndarray], inputs: jnp.ndarray,
                h_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the RNN for a single timestep.
        
        Args:
            params: RNN parameters
            inputs: Input vector containing sensor readings and target position
            h_prev: Previous hidden state
        
        Returns:
            Tuple of (muscle activations, updated hidden state)
        """
        # Compute hidden state
        h_raw = jnp.dot(params['w_ih'], inputs) + jnp.dot(params['w_hh'], h_prev) + params['b_h']
        h = jnp.tanh(h_raw)  # Apply tanh activation
        
        # Compute output (muscle activations)
        outputs_raw = jnp.dot(params['w_ho'], h) + params['b_o']
        
        # Apply sigmoid to constrain outputs between 0 and 1 (muscle activations)
        # Ensure compatibility with JAX 0.5.2
        outputs = jax.nn.sigmoid(outputs_raw)
        
        return outputs, h
    
    def init_hidden(self) -> jnp.ndarray:
        """
        Initialize the hidden state with zeros.
        
        Returns:
            Zero-initialized hidden state
        """
        return jnp.zeros((self.hidden_size,))
    
    def flatten_params(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Flatten parameters dictionary into a single vector for optimization.
        
        Args:
            params: Dictionary of RNN parameters
            
        Returns:
            Flattened parameter vector
        """
        # Create flatten array - ensuring compatibility with JAX 0.5.2
        flattened = jnp.concatenate([
            params['w_ih'].flatten(),
            params['w_hh'].flatten(),
            params['w_ho'].flatten(),
            params['b_h'].flatten(),
            params['b_o'].flatten()
        ])
        
        return flattened
    
    def unflatten_params(self, flat_params: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Unflatten a parameter vector back into a dictionary.
        
        Args:
            flat_params: Flattened parameter vector
            
        Returns:
            Dictionary of RNN parameters
        """
        # Calculate sizes of each parameter
        w_ih_size = self.hidden_size * self.input_size
        w_hh_size = self.hidden_size * self.hidden_size
        w_ho_size = self.output_size * self.hidden_size
        b_h_size = self.hidden_size
        b_o_size = self.output_size
        
        # Split the flattened parameters - compatible with JAX 0.5.2
        idx = 0
        w_ih = jnp.reshape(flat_params[idx:idx+w_ih_size], (self.hidden_size, self.input_size))
        idx += w_ih_size
        
        w_hh = jnp.reshape(flat_params[idx:idx+w_hh_size], (self.hidden_size, self.hidden_size))
        idx += w_hh_size
        
        w_ho = jnp.reshape(flat_params[idx:idx+w_ho_size], (self.output_size, self.hidden_size))
        idx += w_ho_size
        
        b_h = flat_params[idx:idx+b_h_size]
        idx += b_h_size
        
        b_o = flat_params[idx:idx+b_o_size]
        
        return {
            'w_ih': w_ih,
            'w_hh': w_hh,
            'w_ho': w_ho,
            'b_h': b_h,
            'b_o': b_o
        }
    
    @property
    def param_count(self) -> int:
        """
        Calculate the total number of parameters in the RNN.
        
        Returns:
            Total parameter count
        """
        return (self.hidden_size * self.input_size +  # w_ih
                self.hidden_size * self.hidden_size +  # w_hh
                self.output_size * self.hidden_size +  # w_ho
                self.hidden_size +  # b_h
                self.output_size)  # b_o