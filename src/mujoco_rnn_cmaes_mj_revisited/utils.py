import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from abc import ABC, abstractmethod


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
    kernel = lambda_ * np.exp(-lambda_ * time)
    return kernel / kernel.sum()


def truncated_exponential(mu, a, b, size=1):
    """Sample from a truncated exponential distribution using inverse CDF method."""
    lambda_ = 1 / mu
    U = np.random.uniform(0, 1, size)
    exp_a, exp_b = np.exp(-lambda_ * a), np.exp(-lambda_ * b)
    return np.array(-np.log((1 - U) * (exp_a - exp_b) + exp_b) / lambda_)


def sample_entropy(samples, base=2):
    """Compute entropy directly from a vector of samples."""
    _, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs) / np.log(base))


def beta_from_mean(mu, nu=5, num_samples=1):
    alpha = mu * nu
    beta_ = (1 - mu) * nu
    return beta.rvs(alpha, beta_, size=num_samples)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softpus(x):
    return np.log(1 + np.exp(x))


def xavier_init(n_in, n_out):
    stddev = np.sqrt(1 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * stddev


def he_init(n_in, n_out):
    """He (Kaiming) initialization for ReLU weights."""
    stddev = np.sqrt(2 / n_in)
    return np.random.randn(n_out, n_in) * stddev


def l1_norm(x):
    return np.sum(np.abs(x))


def l2_norm(x):
    return np.sqrt(np.sum(x**2))


def normalize01(x, xmin, xmax, default=0.5):
    valid = xmax > xmin
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmin[valid]) / (xmax[valid] - xmin[valid])
    return xnorm


def zscore(x, xmean, xstd, default=0):
    valid = xstd > 0
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmean[valid]) / xstd[valid]
    return xnorm


def action_entropy(action, base=2):
    action = np.clip(action, 1e-10, 1)  # Avoid log(0)
    action_pdf = action / action.sum()
    return -np.sum(action_pdf * np.log(action_pdf) / np.log(base))


def alpha_from_tau(tau, dt):
    """Convert time constant tau to discrete-time alpha parameter."""
    return 1 - np.exp(-dt / tau)


def tau_from_alpha(alpha, dt):
    """Convert discrete-time alpha parameter to time constant tau."""
    return -dt / np.log(1 - alpha)
