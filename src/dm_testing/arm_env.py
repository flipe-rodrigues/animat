from dm_control import suite
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite.utils import randomizers
from dm_control.utils import rewards

from dm_testing.dm_control_test import ArmEntity, ReachTargetTask
from dm_control import mjcf
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 20  # Maximum episode length in seconds
_CONTROL_TIMESTEP = 0.02  # Time between agent actions in seconds

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    model = mjcf.from_path("../mujoco/arm_model.xml")
    return model.to_xml_string(), {}  # Convert MJCF to XML string

class Reach(base.Task):
    """A reach task for the arm model."""

    def __init__(self, random=None):
        """Initialize the reach task."""
        super().__init__(random=random)

        # Create the arm entity and task
        self._arm_model = mjcf.from_path("../mujoco/arm_model.xml")
        self._arm_entity = ArmEntity(self._arm_model)
        self._reach_task = ReachTargetTask(self._arm_entity)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self._reach_task.initialize_episode(physics, self.random)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = {}
        
        # Add hand and target positions
        obs['hand_position'] = physics.named.data.geom_xpos['hand']
        obs['target_position'] = physics.named.data.mocap_pos['target']
        
        # Add muscle states
        for actuator in self._reach_task._actuators:
            obs[f'{actuator}_length'] = physics.named.data.actuator_length[actuator]
            obs[f'{actuator}_velocity'] = physics.named.data.actuator_velocity[actuator]
            obs[f'{actuator}_force'] = physics.named.data.actuator_force[actuator]
        
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return self._reach_task.get_reward(physics)

def make(task_name='reach', task_kwargs=None, environment_kwargs=None, random=None):
    """Returns a new arm environment."""
    if task_kwargs is None:
        task_kwargs = {}
    if environment_kwargs is None:
        environment_kwargs = {}
    
    physics = mjcf.Physics.from_xml_string(*get_model_and_assets())
    task = Reach(random=random, **task_kwargs)
    environment_kwargs['time_limit'] = _DEFAULT_TIME_LIMIT
    return control.Environment(physics, task, control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

def load(task_name='reach', random=None):
    """Returns an environment from a task name and optional random seed."""
    return make(task_name, random=random)

if __name__ == "__main__":
    # Test the environment
    env = load()
    
    # Reset and run a few steps
    time_step = env.reset()
    for _ in range(100):
        action = np.random.uniform(-1, 1, size=env.action_spec().shape)
        time_step = env.step(action)
        print(f'Reward = {time_step.reward}')