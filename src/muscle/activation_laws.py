from abc import ABC, abstractmethod
from spindles import *


class MuscleActivationLaw(ABC):
    """Abstract base class for stretch reflex models"""

    def __init__(self, model, data, spindle: Spindle):
        self.model = model
        self.data = data
        self.spindle = spindle

    @abstractmethod
    def step(self, alpha_drive, gamma_static_drive, gamma_dynamic_drive):
        """Compute reflex force based on drives"""
        pass


class FeldmanActivationLaw(MuscleActivationLaw):

    def __init__(self, model, data, spindle, lambda_extra=0.5):
        super().__init__(model, data, spindle)
        self.initialize_lambda_range(lambda_extra)

    def initialize_lambda_range(self, lambda_extra):
        muscle_id = self.model.actuator("muscle").id
        [length_min, length_max] = self.model.actuator_lengthrange[muscle_id]
        length_range = length_max - length_min
        self.lambda_range = length_range * (1 + lambda_extra)
        self.lambda_min = length_min - lambda_extra / 2 * length_range

    def step(
        self,
        alpha_drive,
        gamma_static_drive,
        gamma_dynamic_drive,
    ):
        self.spindle.step(gamma_static_drive, gamma_dynamic_drive)
        self.alpha_drive = alpha_drive
        self.gamma_static_drive = gamma_static_drive
        self.gamma_dynamic_drive = gamma_dynamic_drive
        self.lambda_ = (1 - gamma_static_drive) * self.lambda_range + self.lambda_min
        self.mu_ = gamma_dynamic_drive
        self.lambda_star = (
            self.lambda_ - self.mu_ * self.spindle.velocity - self.alpha_drive
        )
        force = max(0, self.spindle.length - self.lambda_star)
        return force
