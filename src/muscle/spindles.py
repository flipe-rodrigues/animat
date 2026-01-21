from abc import ABC, abstractmethod


class Spindle(ABC):
    """Abstract base class for muscle spindles"""

    def __init__(self, model, data, gamma_static=0.0, gamma_dynamic=0.0):
        self.model = model
        self.data = data
        self.gamma_static = gamma_static
        self.gamma_dynamic = gamma_dynamic

    @abstractmethod
    def step(self, gamma_static, gamma_dynamic):
        """Update spindle state with new gamma inputs"""
        pass

    @abstractmethod
    def compute_afferent_signals(self):
        """Compute afferent firing rates"""
        pass


class SimpleSpindle(Spindle):
    def __init__(self, model, data, gamma_static=0.0, gamma_dynamic=0.0):
        super().__init__(model, data, gamma_static, gamma_dynamic)
        self.length_sensor_id = model.sensor("muscle_length").id
        self.velocity_sensor_id = model.sensor("muscle_velocity").id

    def step(self, gamma_static, gamma_dynamic):
        self.length = self.data.sensordata[self.length_sensor_id]
        self.velocity = self.data.sensordata[self.velocity_sensor_id]
        self.gamma_static = gamma_static
        self.gamma_dynamic = gamma_dynamic

    def compute_afferent_signals(self):
        spindle_Ia = (
            self.length + self.gamma_static + self.gamma_dynamic * self.velocity
        )
        spindle_II = self.length + self.gamma_static
        return spindle_Ia, spindle_II
