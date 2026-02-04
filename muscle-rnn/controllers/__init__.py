"""
Neural motor controller package.

A hierarchical motor control architecture with biologically-inspired
sensory processing.
"""

from .config import Activation, ControllerConfig, WorkspaceBounds
from .controller import Controller, create_mlp_controller, create_rnn_controller

__all__ = [
    # Config
    "Activation",
    "ControllerConfig",
    "WorkspaceBounds",
    # Controller
    "Controller",
    "create_rnn_controller",
    "create_mlp_controller",
]
