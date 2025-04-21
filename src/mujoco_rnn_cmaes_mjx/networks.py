import copy
import jax
import jax.numpy as jnp
from utils import *


class RNN:
    def __init__(self, input_size, hidden_size, output_size, activation, alpha):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.alpha = alpha
        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init
        self.init_weights()
        self.init_biases()
        self.init_state()
        self.num_params = len(self.get_params())

    def __eq__(self, other):
        if isinstance(other, RNN):
            return jnp.all(self.get_params() == other.get_params())
        return False

    def init_weights(self):
        self.W_in = self.init_fcn(n_in=self.input_size, n_out=self.hidden_size)
        self.W_h = self.init_fcn(n_in=self.hidden_size, n_out=self.hidden_size)
        self.W_out = self.init_fcn(n_in=self.hidden_size, n_out=self.output_size)

    def init_biases(self):
        self.b_h = jnp.zeros(self.hidden_size)
        self.b_out = jnp.zeros(self.output_size)

    def init_state(self):
        """Reset hidden state between episodes"""
        self.h = jnp.zeros(self.hidden_size)
        self.out = jnp.zeros(self.output_size)

    def step(self, obs):
        """Compute one RNN step"""
        self.h = (1 - self.alpha) * self.h + self.alpha * self.activation(
            self.W_in @ obs + self.W_h @ self.h + self.b_h
        )
        self.out = (1 - self.alpha) * self.out + self.alpha * logistic(
            self.W_out @ self.h + self.b_out
        )
        return self.out

    def get_params(self):
        return jnp.concatenate(
            [
                self.W_in.flatten(),
                self.W_h.flatten(),
                self.W_out.flatten(),
                self.b_h.flatten(),
                self.b_out.flatten(),
            ]
        )

    def set_params(self, params):
        idx = 0
        W_in_size = self.input_size * self.hidden_size
        W_h_size = self.hidden_size * self.hidden_size
        W_out_size = self.hidden_size * self.output_size

        self.W_in = params[idx : idx + W_in_size].reshape(
            self.hidden_size, self.input_size
        ).T
        idx += W_in_size
        self.W_h = params[idx : idx + W_h_size].reshape(
            self.hidden_size, self.hidden_size
        ).T
        idx += W_h_size
        self.W_out = params[idx : idx + W_out_size].reshape(
            self.output_size, self.hidden_size
        ).T
        idx += W_out_size

        self.b_h = params[idx : idx + self.hidden_size]
        idx += self.hidden_size
        self.b_out = params[idx : idx + self.output_size]

    def from_params(self, params):
        """Return a new RNN with weights and biases from flattened parameters."""
        idx = 0

        def extract(shape):
            nonlocal idx
            size = jnp.prod(jnp.array(shape))
            param = params[idx : idx + size].reshape(shape)
            idx += size
            return param

        new_rnn = copy.deepcopy(self)
        new_rnn.W_in = extract((self.hidden_size, self.input_size)).T
        new_rnn.W_h = extract((self.hidden_size, self.hidden_size)).T
        new_rnn.W_out = extract((self.output_size, self.hidden_size)).T
        new_rnn.b_h = extract((self.hidden_size,))
        new_rnn.b_out = extract((self.output_size,))
        return new_rnn


# JAX-compiled RNN step function for efficiency
@jax.jit
def rnn_step(params, hidden_state, output_state, obs, alpha, activation_fn):
    """JAX-compatible RNN step function"""
    idx = 0
    
    # Extract parameters
    input_size = obs.shape[0]
    hidden_size = hidden_state.shape[0]
    output_size = output_state.shape[0]
    
    W_in_size = input_size * hidden_size
    W_h_size = hidden_size * hidden_size
    W_out_size = hidden_size * output_size
    
    W_in = params[idx:idx + W_in_size].reshape((hidden_size, input_size))
    idx += W_in_size
    
    W_h = params[idx:idx + W_h_size].reshape((hidden_size, hidden_size))
    idx += W_h_size
    
    W_out = params[idx:idx + W_out_size].reshape((output_size, hidden_size))
    idx += W_out_size
    
    b_h = params[idx:idx + hidden_size]
    idx += hidden_size
    
    b_out = params[idx:idx + output_size]
    
    # Update hidden state
    new_hidden = (1 - alpha) * hidden_state + alpha * activation_fn(
        jnp.dot(W_in, obs) + jnp.dot(W_h, hidden_state) + b_h
    )
    
    # Update output
    new_output = (1 - alpha) * output_state + alpha * logistic(
        jnp.dot(W_out, new_hidden) + b_out
    )
    
    return new_hidden, new_output


# Vectorized RNN step for parallel environments
@jax.vmap
def parallel_rnn_step(params, hidden_states, output_states, observations, alpha, activation_fn):
    """Vectorized RNN step for batch processing"""
    return rnn_step(params, hidden_states, output_states, observations, alpha, activation_fn)
