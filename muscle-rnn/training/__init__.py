"""Training algorithms - CMA-ES and distillation learning."""

from .train_cmaes import run_cmaes_training, CMAESConfig, CMAES, CMAESTrainer
from .train_distillation import run_distillation_training, DistillationConfig
from .evaluation import evaluate_controller, evaluate_fitness, collect_trajectory_data

__all__ = [
    "run_cmaes_training",
    "CMAESConfig",
    "CMAES",
    "CMAESTrainer",
    "run_distillation_training",
    "DistillationConfig",
    "evaluate_controller",
    "evaluate_fitness",
    "collect_trajectory_data",
]
