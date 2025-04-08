import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence, Dict, Any




class SACPolicy(nn.Module):
    """Actor network for SAC.
    
    Outputs action mean and log standard deviation for continuous control.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256, 256, 128)
    activation: callable = nn.relu
    
    @nn.compact
    def __call__(self, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Initial feature extraction
        x = nn.Dense(self.hidden_dims[0])(x)
        x = self.activation(x)
        
        # Add residual blocks for deeper networks
        for dim in self.hidden_dims[1:]:
            x = ResidualBlock(features=dim, activation=self.activation)(x)
        
        # Actor outputs
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0)
        )(x)
        actor_mean = nn.sigmoid(actor_mean)  # [0,1] range for muscle activations
        
        # Log standard deviations - learnable parameters
        actor_logstd = self.param(
            'actor_logstd',
            nn.initializers.constant(-1.0),
            (self.action_dim,)
        )
        
        return actor_mean, actor_logstd


class SACQFunction(nn.Module):
    """Critic network (Q-function) for SAC.
    
    Estimates Q(s,a) values for state-action pairs.
    """
    hidden_dims: Sequence[int] = (256, 256, 256, 128)
    activation: callable = nn.relu
    
    @nn.compact
    def __call__(self, obs, actions) -> jnp.ndarray:
        # Concatenate observation and action
        x = jnp.concatenate([obs, actions], axis=-1)
        
        # Q-network processing
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.activation(x)
            
        # Q-value output (scalar)
        q_value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0)
        )(x)
        
        return q_value

class ResidualBlock(nn.Module):
    """Residual block for more advanced network architectures."""
    features: int
    activation: callable = nn.relu
    
    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Dense(self.features)(x)
        y = self.activation(y)
        y = nn.Dense(self.features)(y)
        return self.activation(residual + y)
    
    
class RecurrentSACPolicy(nn.Module):
    """Recurrent policy network for SAC"""
    action_dim: int
    hidden_dim: int = 128
    
    def initialize_carry(self, batch_size):
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), 
            (batch_size,), 
            self.hidden_dim
        )
    
    @nn.compact
    def __call__(self, x, carry=None):
        if carry is None:
            carry = nn.LSTMCell.initialize_carry(
                jax.random.PRNGKey(0), 
                (x.shape[0],), 
                self.hidden_dim
            )
        
        lstm_cell = nn.LSTMCell()
        carry, y = lstm_cell(carry, x)
        
        actor_mean = nn.Dense(self.action_dim)(y)
        actor_mean = nn.sigmoid(actor_mean)
        actor_logstd = self.param('actor_logstd', nn.initializers.constant(-1.0), (self.action_dim,))
        
        return actor_mean, actor_logstd, carry


class RecurrentSACQFunction(nn.Module):
    """Recurrent Q-function for SAC"""
    hidden_dim: int = 128
    
    def initialize_carry(self, batch_size):
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), 
            (batch_size,), 
            self.hidden_dim
        )
    
    @nn.compact
    def __call__(self, obs, actions, carry=None):
        x = jnp.concatenate([obs, actions], axis=-1)
        
        if carry is None:
            carry = nn.LSTMCell.initialize_carry(
                jax.random.PRNGKey(0), 
                (x.shape[0],), 
                self.hidden_dim
            )
        
        lstm_cell = nn.LSTMCell()
        carry, y = lstm_cell(carry, x)
        
        q_value = nn.Dense(1)(y)
        
        return q_value, carry