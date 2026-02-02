"""
Core package for shared functionality.

Contains:
- Configuration classes
- Base classes and interfaces
- Observation types for structured data
- Common constants
"""

from .config import ModelConfig, TrainingConfig
from .base import BaseTrainer, BaseController
from .types import (
    Proprioception,
    ProprioceptionTensor,
    Observation,
    ObservationTensor,
)
from . import constants

__all__ = [
    'ModelConfig',
    'TrainingConfig',
    'BaseTrainer',
    'BaseController',
    'Proprioception',
    'ProprioceptionTensor', 
    'Observation',
    'ObservationTensor',
    'constants',
]
