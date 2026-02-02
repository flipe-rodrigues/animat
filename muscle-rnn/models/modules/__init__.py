"""
Neural Network Modules

Building blocks for the controllers:

Sensory Processing:
- SensoryModule: Processes proprioceptive inputs (Ia, II, Ib afferents)
  - Type Ia: velocity-sensitive (dynamic spindle afferents)
  - Type II: length-sensitive (static spindle afferents)
  - Type Ib: force-sensitive (Golgi tendon organs)
  - Includes gamma modulation of spindle sensitivity

Target Encoding:
- TargetEncoder: Converts XYZ position to Gaussian grid activations
- TargetEncodingModule: Learned transformation of encoded target

Core Processing:
- RNNCore: Recurrent neural network (the only recurrent hidden layer)
- MLPCore: Feedforward MLP (for teacher networks)

Motor Output:
- MotorModule: Produces motor commands
  - Alpha MN: direct muscle activations
  - Gamma static: modulates Type II sensitivity
  - Gamma dynamic: modulates Type Ia sensitivity
  - Direct reflex pathways: Ia→Alpha, II→Alpha
"""

from .sensory import SensoryModule
from .motor import MotorModule
from .target import TargetEncoder
from .rnn import RNNCore
from .mlp import MLPCore

__all__ = [
    'SensoryModule',
    'MotorModule',
    'TargetEncoder',
    'RNNCore',
    'MLPCore',
]
