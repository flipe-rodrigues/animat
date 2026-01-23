from abc import ABC, abstractmethod


class GolgiTendonOrgan(ABC):
    """Abstract base class for Golgi tendon organs"""

    def __init__(self, model, data, kf=1.0):
        self.model = model
        self.data = data
        self.muscle_id = model.actuator("muscle").id
        self.force_sensor_id = model.sensor("muscle_force").id
        self.kf = kf

    @abstractmethod
    def step(self):
        """Update Golgi tendon organ state"""
        pass

    @abstractmethod
    def compute_afferents(self):
        """Compute afferent firing rates"""
        pass


class SimpleGolgiTendonOrgan(GolgiTendonOrgan):
    """A simple Golgi tendon organ model"""

    def step(self):
        self.force = self.data.ctrl[self.muscle_id]  * self.kf

    def compute_afferents(self):
        Ib_afferent = self.force
        return Ib_afferent
