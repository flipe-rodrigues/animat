import os
import jax.numpy as jnp
from pathlib import Path

def get_root_path():
    """Get the root path of the project."""
    # Return the directory where the current file (utils.py) is located
    return Path(__file__).parent.absolute()

def l1_norm(x):
    """JAX-compatible L1 norm (Manhattan distance)."""
    return jnp.sum(jnp.abs(x))

def l2_norm(x):
    """JAX-compatible L2 norm (Euclidean distance)."""
    return jnp.sqrt(jnp.sum(x**2))

def action_entropy(action, base=2):
    """Calculate entropy of action distribution (JAX compatible)."""
    action = jnp.maximum(action, 1e-10)  # Avoid log(0)
    action_pdf = action / jnp.sum(action)
    return -jnp.sum(action_pdf * jnp.log(action_pdf) / jnp.log(base))

# Additional utility functions that might be useful
def normalize(x, mean, std, epsilon=1e-8):
    """Normalize data with JAX."""
    return (x - mean) / (std + epsilon)