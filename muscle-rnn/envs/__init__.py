"""
Environment Components

- ReachingEnv: Reaching task implementation (trials, phases, rewards)
- TrialConfig: Configuration for individual trials
- MuJoCoPlant: Simple wrapper around MuJoCo physics

The environment provides raw observations:
- Proprioceptive: normalized muscle length, velocity, force
- Target: raw XYZ position (encoding done by controller)

Actions are alpha motor neuron activations only.
Gamma modulation is handled internally by the controller.
"""

from .reaching import ReachingEnv, TrialConfig
from .plant import (
    MuJoCoPlant,
    PlantState,
    ParsedModel,
    JointInfo,
    MuscleInfo,
    SensorInfo,
    BodyInfo,
    parse_mujoco_xml,
    get_model_dimensions,
    calibrate_sensors,
)

__all__ = [
    'ReachingEnv',
    'TrialConfig',
    'MuJoCoPlant',
    'PlantState',
    'ParsedModel',
    'JointInfo',
    'MuscleInfo', 
    'SensorInfo',
    'BodyInfo',
    'parse_mujoco_xml',
    'get_model_dimensions',
    'calibrate_sensors',
]
