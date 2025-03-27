from dm_control import mjcf
from dm_control.composer import Task, Environment, Entity, Observables
from dm_control.composer.observation import observable
import numpy as np
import PIL.Image

# Load the arm model from the XML file
arm_model = mjcf.from_path("mujoco/arm_model.xml")

