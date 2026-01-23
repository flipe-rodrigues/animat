from abc import ABC, abstractmethod
from muscle.spindles import *
from muscle.golgi_tendon_organs import *


class ActivationLaw(ABC):
    """Abstract base class for muscle activation laws"""

    def __init__(self, model, data, spindle: Spindle, gto: GolgiTendonOrgan = None):
        self.model = model
        self.data = data
        self.spindle = spindle
        self.gto = gto

    @abstractmethod
    def step(self, alpha_drive, gamma_static_drive, gamma_dynamic_drive):
        """Update activation law state with new drives"""
        pass


class FeldmanActivationLaw(ActivationLaw):

    def __init__(self, model, data, spindle, gto=None, lambda_extra=0.5):
        super().__init__(model, data, spindle, gto)
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
        gamma_static,
        gamma_dynamic,
    ):
        self.spindle.step(gamma_static, gamma_dynamic)
        self.gto.step() if self.gto else None
        self.alpha_drive = alpha_drive
        self.gamma_static = gamma_static
        self.gamma_dynamic = gamma_dynamic
        self.lambda_ = (1 - (gamma_static + alpha_drive)) * self.lambda_range + self.lambda_min
        self.mu_ = gamma_dynamic
        self.rho_ = self.gto.force if self.gto else 0
        self.epsilon_ = 0
        self.lambda_star = (
            self.lambda_ - self.mu_ * self.spindle.velocity + self.rho_ + self.epsilon_
        )
        force = max(0, self.spindle.length - self.lambda_star)
        return force
