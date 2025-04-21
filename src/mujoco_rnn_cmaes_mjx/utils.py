import os
import jax
import jax.numpy as jnp
from scipy.stats import beta
import numpy as np


def get_root_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path


def exponential_kernel(tau, time):
    """Generates an exponential kernel parameterized by its mean"""
    lambda_ = 1 / tau
    kernel = lambda_ * jnp.exp(-lambda_ * time)
    return kernel / kernel.sum()


def truncated_exponential(mu, a, b, size=1):
    """Sample from a truncated exponential distribution using inverse CDF method."""
    lambda_ = 1 / mu
    if hasattr(size, "__iter__"):
        U = np.random.uniform(0, 1, size=size)
    else:
        U = np.random.uniform(0, 1, size=size)
    exp_a, exp_b = np.exp(-lambda_ * a), np.exp(-lambda_ * b)
    return np.array(-np.log((1 - U) * (exp_a - exp_b) + exp_b) / lambda_)


def sample_entropy(samples, base=2):
    """Compute entropy directly from a vector of samples."""
    _, counts = jnp.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    return -jnp.sum(probs * jnp.log(probs) / jnp.log(base))


def beta_from_mean(mu, nu=5, num_samples=1):
    alpha = mu * nu
    beta_ = (1 - mu) * nu
    return beta.rvs(alpha, beta_, size=num_samples)


def logistic(x):
    return 1 / (1 + jnp.exp(-x))


def tanh(x):
    return jnp.tanh(x)


def relu(x):
    return jnp.maximum(0, x)


def softpus(x):
    return jnp.log(1 + jnp.exp(x))


def xavier_init(n_in, n_out):
    key = jax.random.PRNGKey(0)
    stddev = jnp.sqrt(1 / (n_in + n_out))
    return jax.random.normal(key, (n_out, n_in)) * stddev


def he_init(n_in, n_out):
    """He (Kaiming) initialization for ReLU weights."""
    key = jax.random.PRNGKey(0)
    stddev = jnp.sqrt(2 / n_in)
    return jax.random.normal(key, (n_out, n_in)) * stddev


def l1_norm(x):
    return jnp.sum(jnp.abs(x))


def l2_norm(x):
    return jnp.sqrt(jnp.sum(x**2))


def normalize01(x, xmin, xmax, default=0.5):
    valid = xmax > xmin
    xnorm = jnp.full_like(x, default)
    # Use JAX's functional update pattern
    xnorm = jnp.where(valid, (x - xmin) / (xmax - xmin), xnorm)
    return xnorm


def zscore(x, xmean, xstd, default=0):
    valid = xstd > 0
    xnorm = jnp.full_like(x, default)
    # Use JAX's functional update pattern
    xnorm = jnp.where(valid, (x - xmean) / xstd, xnorm)
    return xnorm


def action_entropy(action, base=2):
    action = jnp.clip(action, 1e-10, 1)  # Avoid log(0)
    action_pdf = action / action.sum()
    return -jnp.sum(action_pdf * jnp.log(action_pdf) / jnp.log(base))
