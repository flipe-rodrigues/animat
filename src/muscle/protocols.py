from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Step(ABC):
    """Abstract base class for protocol steps"""

    duration: float


@dataclass
class Cycle(ABC):
    """Abstract base class for protocol cycles"""

    steps: List[Step]

    def get_duration(self) -> float:
        return sum(step.duration for step in self.steps)


class Protocol(ABC):
    """Abstract base class for protocols"""

    def __init__(self):
        self.cycles: List[Cycle] = []
        self.steps: List[Step] = []

    @abstractmethod
    def add_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_cycle(self, *args, **kwargs):
        pass

    def get_current_step(self, time) -> Optional[Step]:
        cumulative_time = 0
        for step in self.steps:
            cumulative_time += step.duration
            if time < cumulative_time:
                return step
        return None

    def get_all_steps(self) -> List[Step]:
        return self.steps

    def get_duration(self) -> float:
        return sum(step.duration for step in self.steps)


@dataclass
class StretchStep(Step):
    """Represents a single step in a stretch protocol"""

    stretch_velocity: float
    alpha_drive: float
    gamma_static: float
    gamma_dynamic: float
    duration: float


@dataclass
class StretchCycle(Cycle):
    """Represents a cycle in a stretch protocol"""

    steps: List[StretchStep]


class StretchProtocol(Protocol):
    """Concrete implementation of a stretch protocol"""

    def __init__(self):
        super().__init__()
        self.cycles: List[StretchCycle] = []
        self.steps: List[StretchStep] = []

    def add_step(
        self,
        stretch_velocity,
        alpha_drive,
        gamma_static,
        gamma_dynamic,
        duration,
    ):
        self.steps.append(
            StretchStep(
                stretch_velocity=stretch_velocity,
                alpha_drive=alpha_drive,
                gamma_static=gamma_static,
                gamma_dynamic=gamma_dynamic,
                duration=duration,
            )
        )
        return self

    def add_cycle(
        self,
        stretch_speed=0.1,
        alpha_drive=0,
        gamma_static=0,
        gamma_dynamic=0,
        duration=5,
    ):
        """Add a standard stretch cycle: 0 → +speed → 0 → -speed → 0"""
        steps = []
        for stretch_velocity in [
            0,
            stretch_speed,
            0,
            -stretch_speed,
            0,
        ]:
            steps.append(
                StretchStep(
                    stretch_velocity=stretch_velocity,
                    alpha_drive=alpha_drive,
                    gamma_static=gamma_static,
                    gamma_dynamic=gamma_dynamic,
                    duration=duration,
                )
            )

        cycle = StretchCycle(steps)
        self.cycles.append(cycle)
        self.steps.extend(steps)
        return self
