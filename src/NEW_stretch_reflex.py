# %%
"""
.####.##.....##.########...#######..########..########
..##..###...###.##.....##.##.....##.##.....##....##...
..##..####.####.##.....##.##.....##.##.....##....##...
..##..##.###.##.########..##.....##.########.....##...
..##..##.....##.##........##.....##.##...##......##...
..##..##.....##.##........##.....##.##....##.....##...
.####.##.....##.##.........#######..##.....##....##...
"""
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from muscle.spindles import *
from muscle.activation_laws import *
from muscle.protocols import *


# %%
# can eventually replace with logging module !!!!!!!!!!!!!!!!!
"""
.##........#######...######....######...########.########.
.##.......##.....##.##....##..##....##..##.......##.....##
.##.......##.....##.##........##........##.......##.....##
.##.......##.....##.##...####.##...####.######...########.
.##.......##.....##.##....##..##....##..##.......##...##..
.##.......##.....##.##....##..##....##..##.......##....##.
.########..#######...######....######...########.##.....##
"""


class StretchLogger:
    """Logs experimental data during simulation"""

    FIELDS = [
        "time",
        "stretcher",
        "length",
        "velocity",
        "force",
        "alpha_",
        "gamma_static",
        "gamma_dynamic",
        "lambda_",
        "mu_",
        "lambda_star",
        "spindle_Ia",
        "spindle_II",
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

        return pd.DataFrame(
            {
                "min": df[numeric_cols].min(),
                "max": df[numeric_cols].max(),
                "mean": df[numeric_cols].mean(),
                "std": df[numeric_cols].std(),
            }
        )


# %%
"""
.########.##.....##.########..########.########..####.##.....##.########.##....##.########
.##........##...##..##.....##.##.......##.....##..##..###...###.##.......###...##....##...
.##.........##.##...##.....##.##.......##.....##..##..####.####.##.......####..##....##...
.######......###....########..######...########...##..##.###.##.######...##.##.##....##...
.##.........##.##...##........##.......##...##....##..##.....##.##.......##..####....##...
.##........##...##..##........##.......##....##...##..##.....##.##.......##...###....##...
.########.##.....##.##........########.##.....##.####.##.....##.########.##....##....##...
"""


class StretchExperiment:
    """Runs a stretch reflex experiment in Mujoco"""

    def __init__(
        self,
        model,
        data,
        protocol: Protocol,
        activation_law: ActivationLaw,
    ):
        self.model = model
        self.data = data
        self.protocol = protocol
        self.activation_law = activation_law
        self.log = StretchLogger()
        self.stretcher_id = model.actuator("stretcher").id
        self.muscle_id = model.actuator("muscle").id
        self.alpha_drive_id = model.actuator("alpha_drive").id
        self.gamma_static_drive_id = model.actuator("gamma_static_drive").id
        self.gamma_dynamic_drive_id = model.actuator("gamma_dynamic_drive").id
        self.force_sensor_id = model.sensor("muscle_force").id

    def run(self, render=False, render_speed=1.0):
        duration = self.protocol.get_duration()
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.cam.lookat[:] = [0, 0.25, 0]
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -10
            self.viewer.sync()

        while self.data.time < duration:

            # Get current instruction
            current_step = self.protocol.get_current_step(self.data.time)

            # Apply instruction to controls
            self.data.ctrl[self.stretcher_id] = current_step.stretch_velocity
            alpha_drive = current_step.alpha_drive
            gamma_static_drive = current_step.gamma_static_drive
            gamma_dynamic_drive = current_step.gamma_dynamic_drive

            # stretch reflex parameters
            force = self.activation_law.step(
                alpha_drive,
                gamma_static_drive,
                gamma_dynamic_drive,
            )

            # Apply muscle force
            self.data.ctrl[self.muscle_id] = force

            # Compute spindle afferent signals
            spindle_Ia, spindle_II = (
                self.activation_law.spindle.compute_afferent_signals()
            )

            # Log data
            self.log.append(
                time=self.data.time,
                stretcher=current_step.stretch_velocity,
                length=self.activation_law.spindle.length,
                velocity=self.activation_law.spindle.velocity,
                force=self.data.sensordata[self.force_sensor_id],
                alpha_=current_step.alpha_drive,
                gamma_static=current_step.gamma_static_drive,
                gamma_dynamic=current_step.gamma_dynamic_drive,
                lambda_=self.activation_law.lambda_,
                mu_=self.activation_law.mu_,
                lambda_star=self.activation_law.lambda_star,
                spindle_Ia=spindle_Ia,
                spindle_II=spindle_II,
            )

            # Render if needed
            if render and self.viewer.is_running():
                self.viewer.sync()
                self.data.ctrl[self.alpha_drive_id] = alpha_drive
                self.data.ctrl[self.gamma_static_drive_id] = gamma_static_drive
                self.data.ctrl[self.gamma_dynamic_drive_id] = gamma_dynamic_drive
                time.sleep(self.model.opt.timestep / render_speed)

            # Step the simulation
            mujoco.mj_step(self.model, self.data)
    
        if render and self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# %%
"""
.##.....##....###....####.##....##
.###...###...##.##....##..###...##
.####.####..##...##...##..####..##
.##.###.##.##.....##..##..##.##.##
.##.....##.#########..##..##..####
.##.....##.##.....##..##..##...###
.##.....##.##.....##.####.##....##
"""

# Define stretch protocol
protocol = (
    StretchProtocol()
    .add_cycle(stretch_speed=0.1, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.5, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_dynamic_drive=0.5, duration=4)
    .add_cycle(stretch_speed=0.4, gamma_dynamic_drive=0.5, duration=1)
    .add_cycle(stretch_speed=0.4, gamma_static_drive=0.5, gamma_dynamic_drive=0.5, duration=1)
    .add_cycle(stretch_speed=0.1, alpha_drive=0.5, duration=4)
    .add_cycle(stretch_speed=0.1, alpha_drive=0.5, gamma_static_drive=0.5, duration=4)
)

# Load model and data
os.chdir(os.path.dirname(__file__))
MODEL_XML_PATH = "../mujoco/muscle.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)
# plant class? with model and data? gets xml as input?

# Create and run experiment
spindle = SimpleSpindle(model, data)
stretch_reflex = FeldmanActivationLaw(
    model,
    data,
    spindle,
    lambda_extra=0.5,
)
experiment = StretchExperiment(model, data, protocol, stretch_reflex)

# Run with rendering
experiment.run(render=False, render_speed=1)

# %%
"""
.########..##........#######..########
.##.....##.##.......##.....##....##...
.##.....##.##.......##.....##....##...
.########..##.......##.....##....##...
.##........##.......##.....##....##...
.##........##.......##.....##....##...
.##........########..#######.....##...
"""
df = experiment.log.to_dataframe()
num_fields = len(df.columns) - 1  # exclude time column
print(f"Logged {num_fields} fields:")

# Initialize figure and axes
duration = experiment.protocol.get_duration()
width = duration / 8
height = 2 * num_fields
fig, axes = plt.subplots(num_fields, figsize=(width, height), sharex=True)
idx = 0

axes[idx].plot(df["time"], df["stretcher"], color="black")
axes[idx].set_title("Stretcher")
axes[idx].set_ylabel("Control (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["length"], color="black")
axes[idx].set_title("Muscle Length")
axes[idx].set_ylabel("Length (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["velocity"], color="black")
axes[idx].set_title("Muscle Velocity")
axes[idx].set_ylabel("Velocity (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["force"], color="black")
axes[idx].set_title("Muscle Force")
axes[idx].set_ylabel("Force (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["alpha_"], color="black")
axes[idx].set_title(r"$\alpha$ drive")
axes[idx].set_ylabel("Control (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["gamma_static"], color="black")
axes[idx].set_title(r"$\gamma_{static}$ drive")
axes[idx].set_ylabel("Control (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["gamma_dynamic"], color="black")
axes[idx].set_title(r"$\gamma_{dynamic}$ drive")
axes[idx].set_ylabel("Control (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["lambda_"], color="black")
axes[idx].set_title(r"$\lambda$")
axes[idx].set_ylabel("Length (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["mu_"], color="black")
axes[idx].set_title(r"$\mu$")
axes[idx].set_ylabel("a.u.")
idx += 1

axes[idx].plot(df["time"], df["lambda_star"], color="black")
axes[idx].set_title(r"$\lambda^*$")
axes[idx].set_ylabel("Length (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["spindle_Ia"], color="black")
axes[idx].set_title("Spindle Afferent Ia")
axes[idx].set_ylabel("Firing Rate (a.u.)")
idx += 1

axes[idx].plot(df["time"], df["spindle_II"], color="black")
axes[idx].set_title("Spindle Afferent II")
axes[idx].set_ylabel("Firing Rate (a.u.)")
axes[idx].set_xlabel("Time (s)")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_frame_on(False)

plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(8, 6))
plt.scatter(df["length"], df["force"], c=df["time"], cmap="viridis", s=10, alpha=0.6)
plt.colorbar(label="Time (s)")
plt.xlabel("Muscle Length (a.u.)")
plt.ylabel("Muscle Force (a.u.)")
plt.title("Muscle Force vs Length")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
protocol = (
    StretchProtocol()
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.1, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.2, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.3, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.4, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.5, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.6, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.7, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.8, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=0.9, duration=4)
    .add_cycle(stretch_speed=0.1, gamma_static_drive=1.0, duration=4)
)

# Load model and data
os.chdir(os.path.dirname(__file__))
MODEL_XML_PATH = "../mujoco/muscle.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# Create and run experiment
spindle = SimpleSpindle(model, data)
stretch_reflex = FeldmanActivationLaw(
    model,
    data,
    spindle,
    lambda_extra=0.5,
)
experiment = StretchExperiment(model, data, protocol, stretch_reflex)

experiment.run(render=False)
df = experiment.log.to_dataframe()

# Separate data by velocity sign
positive_vel = df[df["velocity"] >= 0]
negative_vel = df[df["velocity"] < 0]

plt.figure(figsize=(8, 6))

# Filled markers for positive velocity
plt.scatter(
    positive_vel["length"],
    -positive_vel["force"],
    c=positive_vel["gamma_static"],
    cmap="viridis",
    s=10,
    alpha=0.6,
    marker="o",
    label="v ≥ 0",
)

# Open markers for negative velocity
# plt.scatter(
#     negative_vel["length"],
#     -negative_vel["force"],
#     c=negative_vel["gamma_static"],
#     cmap="viridis",
#     s=10,
#     alpha=0.6,
#     marker="x",
#     facecolors="none",
#     edgecolors=plt.cm.viridis(negative_vel["gamma_static"]),
#     label="v < 0",
# )

plt.colorbar(label=r"$\gamma_{static}$ drive")
plt.xlabel("Muscle Length (a.u.)")
plt.ylabel("Muscle Force (a.u.)")
plt.title("Muscle Force vs Length (color-coded by γ-static)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
