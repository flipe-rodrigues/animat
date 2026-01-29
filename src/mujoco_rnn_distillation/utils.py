import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from abc import ABC, abstractmethod
from typing import Union, Optional

# Cache commonly used constants
_LOG2 = np.log(2)
_SQRT2 = np.sqrt(2)


def get_root_path() -> str:
    """
    Find project root by searching for .git directory.

    Returns:
        Absolute path to project root
    """
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path


def exponential_kernel(tau: float, time: np.ndarray) -> np.ndarray:
    """
    Generate exponential kernel parameterized by its mean.

    Args:
        tau: Time constant (mean of exponential)
        time: Time array

    Returns:
        Normalized kernel values
    """
    lambda_ = 1 / tau
    kernel = lambda_ * np.exp(-lambda_ * time)
    return kernel / kernel.sum()


def truncated_exponential(
    mu: float, a: float, b: float, size: Union[int, tuple] = 1
) -> np.ndarray:
    """
    Sample from truncated exponential distribution using inverse CDF method.

    Optimized with validation and numerical stability checks.

    Args:
        mu: Mean of (untruncated) exponential
        a: Lower bound
        b: Upper bound
        size: Number of samples or shape

    Returns:
        Array of samples from truncated exponential
    """
    # Validation
    if mu <= 0:
        raise ValueError("mu must be positive")
    if a >= b:
        raise ValueError("a must be < b")

    lambda_ = 1 / mu
    U = np.random.uniform(0, 1, size)
    exp_a, exp_b = np.exp(-lambda_ * a), np.exp(-lambda_ * b)

    # Check for numerical issues
    denominator = exp_a - exp_b
    if np.abs(denominator) < 1e-10:
        # If a and b are very close, return midpoint
        return np.full(size if isinstance(size, int) else np.prod(size), (a + b) / 2)

    return -np.log((1 - U) * denominator + exp_b) / lambda_


def sample_entropy(samples: np.ndarray, base: float = 2) -> float:
    """
    Compute entropy directly from samples (optimized).

    Args:
        samples: Array of discrete samples
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy in specified units
    """
    _, counts = np.unique(samples, return_counts=True)
    probs = counts / len(samples)  # Slightly faster than counts.sum()

    # Use specialized functions for common bases
    if base == 2:
        return -np.sum(probs * np.log2(probs))
    elif base == np.e:
        return -np.sum(probs * np.log(probs))
    else:
        return -np.sum(probs * np.log(probs)) / np.log(base)


def beta_from_mean(
    mu: float, nu: float = 5, num_samples: int = 1
) -> Union[float, np.ndarray]:
    """
    Sample from beta distribution parameterized by mean and precision.

    Args:
        mu: Mean (0 < mu < 1)
        nu: Precision parameter (higher = less variance)
        num_samples: Number of samples

    Returns:
        Samples from beta distribution
    """
    alpha = mu * nu
    beta_param = (1 - mu) * nu
    return beta.rvs(alpha, beta_param, size=num_samples)


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic (sigmoid) activation function"""
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function"""
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit activation function"""
    return np.maximum(0, x)


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation function (smooth approximation of ReLU)"""
    return np.log(1 + np.exp(x))


def xavier_init(n_in: int, n_out: int) -> np.ndarray:
    """
    Xavier/Glorot initialization for neural network weights.

    Good for tanh/sigmoid activations.

    Args:
        n_in: Input dimension
        n_out: Output dimension

    Returns:
        Weight matrix (n_out x n_in)
    """
    stddev = np.sqrt(1 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * stddev


def he_init(n_in: int, n_out: int) -> np.ndarray:
    """
    He (Kaiming) initialization for neural network weights.

    Good for ReLU activations.

    Args:
        n_in: Input dimension
        n_out: Output dimension

    Returns:
        Weight matrix (n_out x n_in)
    """
    stddev = np.sqrt(2 / n_in)
    return np.random.randn(n_out, n_in) * stddev


def l1_norm(x: np.ndarray) -> float:
    """L1 (Manhattan) norm"""
    return np.sum(np.abs(x))


def l2_norm(x: np.ndarray) -> float:
    """L2 (Euclidean) norm"""
    return np.sqrt(np.sum(x**2))


def normalize01(
    x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, default: float = 0.5
) -> np.ndarray:
    """
    Normalize to [0, 1] range.

    Args:
        x: Values to normalize
        xmin: Minimum values
        xmax: Maximum values
        default: Default value when xmax == xmin

    Returns:
        Normalized values
    """
    valid = xmax > xmin
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmin[valid]) / (xmax[valid] - xmin[valid])
    return xnorm


def zscore(
    x: np.ndarray, xmean: np.ndarray, xstd: np.ndarray, default: float = 0.0
) -> np.ndarray:
    """
    Z-score normalization (optimized).

    Fast path when all values are valid, avoiding array allocation.

    Args:
        x: Values to normalize
        xmean: Mean values
        xstd: Standard deviation values
        default: Default value when xstd == 0

    Returns:
        Normalized values
    """
    # Fast path: all valid
    if np.all(xstd > 0):
        return (x - xmean) / xstd

    # Slow path: some invalid
    valid = xstd > 0
    result = np.full_like(x, default, dtype=np.float64)
    result[valid] = (x[valid] - xmean[valid]) / xstd[valid]
    return result


def action_entropy(action: np.ndarray, base: float = 2) -> float:
    """
    Compute entropy of action distribution.

    Args:
        action: Action values (treated as unnormalized probabilities)
        base: Logarithm base

    Returns:
        Entropy of action distribution
    """
    action = np.clip(action, 1e-10, 1)  # Avoid log(0)
    action_pdf = action / action.sum()

    if base == 2:
        return -np.sum(action_pdf * np.log2(action_pdf))
    else:
        return -np.sum(action_pdf * np.log(action_pdf)) / np.log(base)


def gamma_from_tau(tau: float, dt: float) -> float:
    """
    Convert time constant tau to discount factor gamma.

    Args:
        tau: Time constant
        dt: Timestep

    Returns:
        Discount factor gamma
    """
    return np.exp(-dt / tau)


def alpha_from_tau(tau: float, dt: float) -> float:
    """
    Convert time constant tau to discrete-time alpha parameter.

    alpha = 1 - exp(-dt/tau) represents the update rate.

    Args:
        tau: Time constant
        dt: Timestep

    Returns:
        Alpha parameter (update rate)
    """
    return 1.0 - np.exp(-dt / tau)


def tau_from_alpha(alpha: float, dt: float) -> float:
    """
    Convert discrete-time alpha parameter to time constant tau.

    Args:
        alpha: Update rate (0 < alpha < 1)
        dt: Timestep

    Returns:
        Time constant tau
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    return -dt / np.log(1 - alpha)