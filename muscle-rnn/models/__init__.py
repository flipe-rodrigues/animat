"""
Neural Network Models for Muscle-Driven Arm Control

Controllers:
- RNNController: RNN-based controller with temporal integration
- MLPController: MLP-based controller (for teacher in distillation)
- ControllerBase: Common base class with shared functionality

Factory:
- create_controller: Create controller by type ('rnn' or 'mlp')

Individual modules are available in models.modules:
- SensoryModule: Proprioceptive sensory neurons (Ia, II, Ib)
- MotorModule: Alpha and Gamma motor neuron outputs with reflexes
- TargetEncoder: Encodes XYZ position as Gaussian grid activations
- RNNCore: Recurrent neural network core (the only recurrent layer)
- MLPCore: Feedforward MLP core

Note: ModelConfig and BaseController now imported from core module.
"""

from .controllers import (
    ControllerBase,
    RNNController,
    MLPController,
    create_controller,
)

from .modules import (
    SensoryModule,
    MotorModule,
    TargetEncoder,
    RNNCore,
    MLPCore,
)

__all__ = [
    # Controllers
    'ControllerBase',
    'RNNController',
    'MLPController',
    'create_controller',
    # Modules
    'SensoryModule',
    'MotorModule',
    'TargetEncoder',
    'RNNCore',
    'MLPCore',
]
