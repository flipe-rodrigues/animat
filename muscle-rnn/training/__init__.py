"""Training algorithms - CMA-ES and distillation learning."""

from .train_cmaes import cmaes_optimize, evaluate_controller
from .train_distillation import (
    train_teacher_bc,
    train_student_distillation,
    train_with_sb3_bc,
    collect_random_data,
    collect_expert_data,
    TrajectoryDataset,
    BehaviorCloningDataset,
)

__all__ = [
    "cmaes_optimize",
    "evaluate_controller",
    "train_teacher_bc",
    "train_student_distillation",
    "train_with_sb3_bc",
    "collect_random_data",
    "collect_expert_data",
    "TrajectoryDataset",
    "BehaviorCloningDataset",
]
