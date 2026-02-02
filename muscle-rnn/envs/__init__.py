"""Environment components - Gym environment and MuJoCo plant wrapper."""

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
    "ReachingEnv",
    "TrialConfig",
    "MuJoCoPlant",
    "PlantState",
    "ParsedModel",
    "JointInfo",
    "MuscleInfo",
    "SensorInfo",
    "BodyInfo",
    "parse_mujoco_xml",
    "get_model_dimensions",
    "calibrate_sensors",
]
