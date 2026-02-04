"""
Models Module

Neural network components for motor control:
- Config: Controller configuration
- Modules: BaseModule and all module implementations
- Controllers: Complete controller architectures
"""

from .controllers import ControllerConfig, BaseController, RNNController, MLPController
from .modules import (
    BaseModule,
    TargetEncoder,
    SensoryModule,
    MotorModule,
    BaseCore,
    RNNCore,
    MLPCore,
)

__all__ = [
    # Config
    "ControllerConfig",
    # Base
    "BaseModule",
    # Modules
    "TargetEncoder",
    "SensoryModule",
    "MotorModule",
    # Core networks
    "BaseCore",
    "RNNCore",
    "MLPCore",
    # Controllers
    "BaseController",
    "RNNController",
    "MLPController",
]
