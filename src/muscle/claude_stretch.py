"""
Muscle Stretch Simulation with Lambda Model
Refactored for improved organization and extensibility
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# DOMAIN MODELS
# ============================================================================

@dataclass
class Step:
    """Represents a single step in a stretch protocol"""
    stretcher: float
    alpha: float
    gamma_static: float
    gamma_dynamic: float
    duration: float


@dataclass
class MuscleDynamicsParams:
    """Parameters for muscle dynamics computation"""
    lambda_min: float
    lambda_range: float
    
    def compute_lambda(self, gamma_static: float) -> float:
        """Compute lambda from gamma_static"""
        return (1 - gamma_static) * self.lambda_range + self.lambda_min
    
    def compute_lambda_star(self, lambda_: float, mu: float, 
                           velocity: float, alpha: float) -> float:
        """Compute dynamic lambda"""
        return lambda_ - mu * velocity - alpha
    
    def compute_force(self, length: float, lambda_star: float) -> float:
        """Compute muscle force"""
        return max(0, length - lambda_star)
    
    def compute_spindle_ia(self, length: float, mu: float, velocity: float) -> float:
        """Compute spindle Ia afferent signal"""
        return length + mu * velocity
    
    def compute_spindle_ii(self, length: float, gamma_static: float) -> float:
        """Compute spindle II afferent signal"""
        return length + gamma_static


@dataclass
class MujocoModelElements:
    """Container for MuJoCo model element IDs"""
    stretcher_id: int
    soleus_id: int
    length_sensor_id: int
    velocity_sensor_id: int
    force_sensor_id: int
    
    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> 'MujocoModelElements':
        """Extract model elements from MuJoCo model"""
        return cls(
            stretcher_id=model.actuator("stretcher").id,
            soleus_id=model.actuator("soleus").id,
            length_sensor_id=model.sensor("soleus_length").id,
            velocity_sensor_id=model.sensor("soleus_velocity").id,
            force_sensor_id=model.sensor("soleus_force").id
        )


# ============================================================================
# PROTOCOL BUILDER
# ============================================================================

class StretchProtocol:
    """Defines a sequence of stretch steps and cycles"""
    
    def __init__(self):
        self.steps: List[Step] = []
    
    def add_step(self, stretcher: float = 0, alpha: float = 0, 
                 gamma_static: float = 0, gamma_dynamic: float = 0, 
                 duration: float = 1) -> 'StretchProtocol':
        """Add a single step to the protocol"""
        self.steps.append(Step(stretcher, alpha, gamma_static, 
                              gamma_dynamic, duration))
        return self
    
    def add_cycle(self, magnitude: float = 0.1, gamma_static: float = 0,
                  gamma_dynamic: float = 0, alpha: float = 0, 
                  duration: float = 5) -> 'StretchProtocol':
        """Add a standard stretch cycle: 0 → +mag → 0 → -mag → 0"""
        cycle_steps = [
            (0, duration),
            (magnitude, duration),
            (0, duration),
            (-magnitude, duration),
            (0, duration),
        ]
        
        for mag, dur in cycle_steps:
            self.add_step(mag, alpha, gamma_static, gamma_dynamic, dur)
        
        return self
    
    def get_step_at_time(self, time: float) -> Optional[Step]:
        """Get the step active at the given time"""
        cumulative_time = 0
        for step in self.steps:
            cumulative_time += step.duration
            if time < cumulative_time:
                return step
        return None
    
    def get_total_duration(self) -> float:
        """Get total duration of the protocol"""
        return sum(step.duration for step in self.steps)


# ============================================================================
# DATA LOGGING
# ============================================================================

class ExperimentLog:
    """Logs experimental data during simulation"""
    
    FIELDS = [
        "time", "stretcher", "alpha", "gamma_static", "gamma_dynamic",
        "length", "velocity", "lambda", "mu", "lambda_star", "force",
        "spindle_Ia", "spindle_II"
    ]
    
    def __init__(self):
        self.data: Dict[str, List[float]] = {field: [] for field in self.FIELDS}
    
    def append(self, **kwargs):
        """Append a data row. Expects all FIELDS as kwargs"""
        for field in self.FIELDS:
            if field not in kwargs:
                raise ValueError(f"Missing required field: {field}")
            self.data[field].append(kwargs[field])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert log to pandas DataFrame"""
        return pd.DataFrame(self.data)
    
    def compute_statistics(self) -> pd.DataFrame:
        """Compute statistics for all logged variables"""
        df = self.to_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        return pd.DataFrame({
            "min": df[numeric_cols].min(),
            "max": df[numeric_cols].max(),
            "mean": df[numeric_cols].mean(),
            "std": df[numeric_cols].std(),
        })


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class StretchExperiment:
    """Runs a muscle stretch experiment with MuJoCo"""
    
    def __init__(self, protocol: StretchProtocol, model: mujoco.MjModel,
                 data: mujoco.MjData, dynamics_params: MuscleDynamicsParams,
                 model_elements: MujocoModelElements):
        self.protocol = protocol
        self.model = model
        self.data = data
        self.dynamics = dynamics_params
        self.elements = model_elements
        self.log = ExperimentLog()
    
    def _apply_control(self, step: Step):
        """Apply control signals for the current step"""
        self.data.ctrl[self.elements.stretcher_id] = step.stretcher
    
    def _compute_muscle_dynamics(self, step: Step) -> Dict[str, float]:
        """Compute muscle dynamics for the current step"""
        # Read sensors
        length = self.data.sensordata[self.elements.length_sensor_id]
        velocity = self.data.sensordata[self.elements.velocity_sensor_id]
        
        # Compute stretch reflex parameters
        lambda_ = self.dynamics.compute_lambda(step.gamma_static)
        mu = step.gamma_dynamic
        
        # Compute dynamic lambda and force
        lambda_star = self.dynamics.compute_lambda_star(
            lambda_, mu, velocity, step.alpha
        )
        force = self.dynamics.compute_force(length, lambda_star)
        
        # Apply muscle force
        self.data.ctrl[self.elements.soleus_id] = force
        
        # Compute spindle afferents
        spindle_ia = self.dynamics.compute_spindle_ia(length, mu, velocity)
        spindle_ii = self.dynamics.compute_spindle_ii(length, step.gamma_static)
        
        return {
            "length": length,
            "velocity": velocity,
            "lambda": lambda_,
            "mu": mu,
            "lambda_star": lambda_star,
            "force": self.data.sensordata[self.elements.force_sensor_id],
            "spindle_Ia": spindle_ia,
            "spindle_II": spindle_ii,
        }
    
    def _log_step(self, step: Step, dynamics: Dict[str, float]):
        """Log the current simulation state"""
        self.log.append(
            time=self.data.time,
            stretcher=step.stretcher,
            alpha=step.alpha,
            gamma_static=step.gamma_static,
            gamma_dynamic=step.gamma_dynamic,
            **dynamics
        )
    
    def run(self, render: bool = False, viewer: Optional[Any] = None):
        """Run the experiment simulation"""
        duration = self.protocol.get_total_duration()
        
        if render and viewer:
            self._configure_viewer(viewer)
        
        while self.data.time < duration:
            # Render if needed
            if render and viewer and viewer.is_running():
                viewer.sync()
                time.sleep(self.model.opt.timestep)
            
            # Get current protocol step
            step = self.protocol.get_step_at_time(self.data.time)
            if step is None:
                break
            
            # Apply controls and compute dynamics
            self._apply_control(step)
            dynamics = self._compute_muscle_dynamics(step)
            
            # Log data
            self._log_step(step, dynamics)
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
    
    def _configure_viewer(self, viewer):
        """Configure viewer settings"""
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        viewer.cam.lookat[:] = [0, 0.25, 0]
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10


# ============================================================================
# VISUALIZATION
# ============================================================================

class ExperimentPlotter:
    """Handles plotting of experiment results"""
    
    PLOT_CONFIG = [
        ("stretcher", "Stretcher", "Control (a.u.)"),
        ("alpha", r"$\alpha$", "Control (a.u.)"),
        ("gamma_static", r"$\gamma_{static}$", "Control (a.u.)"),
        ("gamma_dynamic", r"$\gamma_{dynamic}$", "Control (a.u.)"),
        ("length", "Muscle Length", "Length (a.u.)"),
        ("velocity", "Muscle Velocity", "Velocity (a.u.)"),
        ("lambda", r"$\lambda$", "Length (a.u.)"),
        ("mu", r"$\mu$", "a.u."),
        ("lambda_star", r"$\lambda^*$", "Length (a.u.)"),
        ("force", "Muscle Force", "Force (a.u.)"),
        ("spindle_Ia", "Spindle Afferent Ia", "Firing Rate (a.u.)"),
        ("spindle_II", "Spindle Afferent II", "Firing Rate (a.u.)"),
    ]
    
    @staticmethod
    def plot_experiment(df: pd.DataFrame, duration: float):
        """Create a complete plot of experiment results"""
        num_plots = len(ExperimentPlotter.PLOT_CONFIG)
        width = duration / 8
        height = 2 * num_plots
        
        fig, axes = plt.subplots(num_plots, figsize=(width, height), sharex=True)
        
        for idx, (field, title, ylabel) in enumerate(ExperimentPlotter.PLOT_CONFIG):
            axes[idx].plot(df["time"], df[field], color="black")
            axes[idx].set_title(title)
            axes[idx].set_ylabel(ylabel)
            axes[idx].spines["top"].set_visible(False)
            axes[idx].spines["right"].set_visible(False)
            axes[idx].set_frame_on(False)
        
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
        
        plt.tight_layout()
        return fig


# ============================================================================
# CONFIGURATION & FACTORY
# ============================================================================

class ExperimentConfig:
    """Configuration for creating experiments"""
    
    def __init__(self, model_path: str, lambda_extra: float = 0.5):
        self.model_path = model_path
        self.lambda_extra = lambda_extra
    
    def create_experiment(self, protocol: StretchProtocol) -> StretchExperiment:
        """Factory method to create a configured experiment"""
        # Load model
        model = mujoco.MjModel.from_xml_path(self.model_path)
        data = mujoco.MjData(model)
        
        # Extract model elements
        elements = MujocoModelElements.from_model(model)
        
        # Compute dynamics parameters
        length_range_values = model.actuator_lengthrange[elements.soleus_id]
        length_min, length_max = length_range_values
        length_range = length_max - length_min
        
        lambda_range = length_range * (1 + self.lambda_extra)
        lambda_min = length_min - (self.lambda_extra / 2) * length_range
        
        dynamics_params = MuscleDynamicsParams(lambda_min, lambda_range)
        
        return StretchExperiment(protocol, model, data, 
                                dynamics_params, elements)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Define protocol
    protocol = (StretchProtocol()
                .add_cycle(magnitude=0.1, duration=4)
                .add_cycle(magnitude=0.1, gamma_static=0.5, duration=4)
                .add_cycle(magnitude=0.1, gamma_dynamic=0.5, duration=4)
                .add_cycle(magnitude=0.1, gamma_static=0.5, 
                          gamma_dynamic=0.5, duration=4)
                .add_cycle(magnitude=0.1, alpha=0.5, duration=4))
    
    # Configure and create experiment
    script_dir = Path(__file__).parent
    model_path = script_dir / "../../mujoco/muscle.xml"
    
    config = ExperimentConfig(str(model_path))
    experiment = config.create_experiment(protocol)
    
    # Run experiment
    experiment.run(render=False)
    
    # Analyze results
    df = experiment.log.to_dataframe()
    print("\nExperiment Statistics:")
    print(experiment.log.compute_statistics())
    
    # Plot results
    duration = protocol.get_total_duration()
    fig = ExperimentPlotter.plot_experiment(df, duration)
    plt.show()


if __name__ == "__main__":
    main()