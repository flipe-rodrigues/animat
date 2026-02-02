"""
Training Algorithms and Utilities

CMA-ES (Covariance Matrix Adaptation Evolution Strategy):
- run_cmaes_training: Main entry point for evolutionary optimization
- CMAESConfig: Configuration dataclass
- CMAES: CMA-ES optimizer implementation

Distillation Learning:
- run_distillation_training: Train RNN by imitating MLP teacher
- DistillationConfig: Configuration dataclass

Shared Utilities:
- evaluation: Common evaluation functions
- checkpoint: Checkpoint saving/loading utilities

Usage (CMA-ES):
    from training.train_cmaes import run_cmaes_training
    
    results = run_cmaes_training(
        xml_path='mujoco/arm.xml',
        num_generations=500,
        population_size=64,
    )

Usage (Distillation):
    from training.train_distillation import run_distillation_training
    
    results = run_distillation_training(
        xml_path='mujoco/arm.xml',
        teacher_epochs=100,
        student_epochs=200,
    )

Command Line:
    python run.py train mujoco/arm.xml --method cmaes --generations 500
    python run.py train mujoco/arm.xml --method distillation
"""

from .train_cmaes import (
    run_cmaes_training,
    CMAESConfig,
    CMAES,
)

from .train_distillation import (
    run_distillation_training,
    DistillationConfig,
)

from . import evaluation
from . import checkpoint

__all__ = [
    # CMA-ES
    'run_cmaes_training',
    'CMAESConfig',
    'CMAESTrainer',
    'CMAES',
    # Distillation
    'run_distillation_training',
    'DistillationConfig',
]
