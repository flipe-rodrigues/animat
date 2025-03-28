from dm_control import composer, mjcf
from dm_testing.dm_control_test import ArmEntity, ReachTargetTask
import numpy as np

_DEFAULT_TIME_LIMIT = 5.0
_CONTROL_TIMESTEP = 0.02

def load():
    """Create and return a composer Environment with the arm task."""
    print("Running the correct load() function")  # Debug print
    # Create the arm entity
    arm_model = mjcf.from_path("../mujoco/arm_model.xml")
    arm_entity = ArmEntity(arm_model)
    
    # Create the reach task
    task = ReachTargetTask(arm_entity)
    
    # Create the environment with a time limit
    env = composer.Environment(task, time_limit=_DEFAULT_TIME_LIMIT)
    
    # Enable all observables explicitly
    for name, observable in arm_entity.observables.as_dict().items():
        observable.enabled = True
    
    # Debug: Print observation spec to verify it's not empty
    print(f"Observation spec: {env.observation_spec()}")
    
    return env

if __name__ == "__main__":
    # Test the environment
    env = load()
    
    # Reset and run a few steps
    time_step = env.reset()
    for _ in range(100):
        action = np.random.uniform(-1, 1, size=env.action_spec().shape)
        time_step = env.step(action)
        print(f'Reward = {time_step.reward}')