"""
Reward shaping functions for reaching tasks.

Two approaches:
1. Simple shaping: Direct cost/reward based on state
2. Potential-based: Uses state differences (guarantees policy invariance)
"""

import numpy as np


# ============================================================================
# SIMPLE SHAPING FUNCTIONS
# ============================================================================


def identity_shaping(x):
    """Identity: f(x) = x"""
    return x


def negation_shaping(x):
    """Negation: f(x) = -x"""
    return -x


def quadratic_shaping(x):
    """Quadratic: f(x) = x²"""
    return x**2


def huber_shaping(x, delta=0.15):
    """
    Huber: quadratic when close, linear when far.
    Good general-purpose choice for reaching tasks.
    """
    if x <= delta:
        return 0.5 * (x**2) / delta
    else:
        return x - 0.5 * delta