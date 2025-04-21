import jax
import jax.numpy as jnp
from typing import Tuple

def init_rnn_params(key, input_dim: int, hidden_dim: int, output_dim: int):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    Wx = jax.random.normal(k1, (hidden_dim, input_dim)) * 0.1
    Wh = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
    b = jnp.zeros((hidden_dim,))

    W_out = jax.random.normal(k3, (output_dim, hidden_dim)) * 0.1
    b_out = jnp.zeros((output_dim,))

    return (Wx, Wh, b), (W_out, b_out)

def rnn_step(rnn_params: Tuple, output_params: Tuple, h: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    Wx, Wh, b = rnn_params
    W_out, b_out = output_params

    h_new = jnp.tanh(jnp.dot(Wx, x) + jnp.dot(Wh, h) + b)
    output = jax.nn.sigmoid(jnp.dot(W_out, h_new) + b_out)  # muscle activations between [0, 1]

    return h_new, output

def unpack_candidate(flat_params: jnp.ndarray, input_dim=15, hidden_dim=32, output_dim=4) -> Tuple:
    """
    Unpacks a flat vector into RNN parameters:
    Wx: (H, I), Wh: (H, H), b: (H,), W_out: (O, H), b_out: (O,)
    """
    idx = 0

    Wx_size = hidden_dim * input_dim
    Wh_size = hidden_dim * hidden_dim
    b_size = hidden_dim
    W_out_size = output_dim * hidden_dim
    b_out_size = output_dim

    Wx = flat_params[idx : idx + Wx_size].reshape((hidden_dim, input_dim))
    idx += Wx_size

    Wh = flat_params[idx : idx + Wh_size].reshape((hidden_dim, hidden_dim))
    idx += Wh_size

    b = flat_params[idx : idx + b_size]
    idx += b_size

    W_out = flat_params[idx : idx + W_out_size].reshape((output_dim, hidden_dim))
    idx += W_out_size

    b_out = flat_params[idx : idx + b_out_size]
    idx += b_out_size

    rnn_params = (Wx, Wh, b)
    output_params = (W_out, b_out)

    return rnn_params, output_params
