# PyMJCF
from dm_control import mjcf

from dm_control import suite

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Composer low level imports

from dm_control.composer import Task, Environment, Entity, Observables


# General
import pandas
import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np
import pickle

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
# Internal loading of video libraries.


# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Inline video helper function
if os.environ.get('COLAB_NOTEBOOK_TEST', False):
  # We skip video generation during tests, as it is quite expensive.
  display_video = lambda *args, **kwargs: None
else:
  def display_video(frames, filename, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    anim.save(filename, writer="pillow", fps=framerate)
    print("Animation saved as animation.gif")

# Seed numpy's global RNG so that cell outputs are deterministic. We also try to
# use RandomState instances that are local to a single cell wherever possible.
np.random.seed(42)

# Load the arm model from the XML file
arm_model = mjcf.from_path("../mujoco/arm_model.xml")

# Wrap the MJCF model in a composer.Entity
class ArmEntity(Entity):
    def _build(self, model):
        """Define the structure of the entity."""
        # Set the MJCF root directly
        self._model = model
        

    def _build_observables(self):
        """Use CreatureObservables to define observables for the arm entity."""
        #print("Building observables for ArmEntity")
        observables = CreatureObservables(self)

        # Debug: List all observable attributes
        observable_names = [
            attr for attr in dir(observables)
            if isinstance(getattr(observables, attr), observable.Observable)
        ]
        #print(f"Registered observables: {observable_names}")

        return observables

    @property
    def mjcf_model(self):
        return self._model

# Add simple observable features for joint angles and velocities.
class CreatureObservables(Observables):
    def __init__(self, entity):
        super().__init__(entity)
        #print("Initializing CreatureObservables")
        # Enable all observables by default
        for name, observable in self.as_dict().items():
            observable.enabled = True

    # Add target position as an observable
    @composer.observable
    def target_position(self):
        print("Defining observable: target_position")
        return observable.MJCFFeature('mocap_pos', self._entity.mjcf_model.find('body', 'target'))

    @composer.observable
    def deltoid_length(self):
        print("Defining observable: deltoid_length")
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'deltoid_length'))

    @composer.observable
    def latissimus_length(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'latissimus_length'))

    @composer.observable
    def biceps_length(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'biceps_length'))

    @composer.observable
    def triceps_length(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'triceps_length'))

    @composer.observable
    def deltoid_velocity(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'deltoid_velocity'))

    @composer.observable
    def latissimus_velocity(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'latissimus_velocity'))

    @composer.observable
    def biceps_velocity(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'biceps_velocity'))

    @composer.observable
    def triceps_velocity(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'triceps_velocity'))

    @composer.observable
    def deltoid_force(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'deltoid_force'))

    @composer.observable
    def latissimus_force(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'latissimus_force'))

    @composer.observable
    def biceps_force(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'biceps_force'))

    @composer.observable
    def triceps_force(self):
        return observable.MJCFFeature('sensordata', self._entity.mjcf_model.find('sensor', 'triceps_force'))

arm_entity = ArmEntity(arm_model)


class ReachTargetTask(Task):
    def __init__(self, arm_entity):
        self._arm_entity = arm_entity
        self._min_activation = 0.0
        self._max_activation = 1.0
        
        # Get absolute path to mujoco directory
        mj_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 
            'mujoco'
        ))
        
        # Load sensor stats
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, 'rb') as f:
            self._sensor_stats = pickle.load(f)

        # Load target stats
        target_stats_path = os.path.join(mj_dir, "target_stats.pkl")
        with open(target_stats_path, 'rb') as f:
            self._target_stats = pickle.load(f)

        # Load valid target positions
        reachable_positions_path = os.path.join(mj_dir, "reachable_positions.pkl")
        with open(reachable_positions_path, 'rb') as f:
            self._reachable_positions = pickle.load(f)
            # Convert DataFrame to numpy array if needed
            if hasattr(self._reachable_positions, 'values'):
                self._reachable_positions = self._reachable_positions.values

        print(f"Loaded reachable positions shape: {self._reachable_positions.shape}")
        print(f"Sample position: {self._reachable_positions[0]}")
        
        self._actuators = [
            'deltoid',      # shoulder flexion
            'latissimus',   # shoulder extension
            'biceps',       # elbow flexion
            'triceps'       # elbow extension
        ]

    @property
    def root_entity(self):
        return self._arm_entity

    def initialize_episode(self, physics, random_state):
        # Reset the simulation state
        physics.reset()

        # Select a random reachable position from the loaded data
        num_positions = len(self._reachable_positions)
        target_idx = random_state.randint(0, num_positions)
        new_target_position = self._reachable_positions[target_idx]
        
        # Apply the new target position to the MuJoCo mocap body
        physics.named.data.mocap_pos["target"] = new_target_position

        # Initialize muscles to a relaxed state
        for actuator in self._actuators:
            physics.named.data.ctrl[actuator] = self._min_activation

    def before_step(self, physics, action, random_state):
        """Process the action before applying it to the physics simulation.
        
        Args:
            physics: A dm_control physics object
            action: Array of muscle activations
            random_state: Numpy random state object
        """
        # Clip actions to valid muscle activation range [0, 1]
        action = np.clip(action, self._min_activation, self._max_activation)
        
        # Apply muscle activations to the four muscles
        for i, actuator in enumerate(self._actuators):
            physics.named.data.ctrl[actuator] = action[i]

    def get_reward(self, physics):
        # Get current positions
        hand_position = physics.named.data.geom_xpos["hand"]
        target_position = physics.named.data.mocap_pos["target"]
        
        # Calculate distance to target
        distance = np.linalg.norm(hand_position - target_position)
        
        # Base reward is negative distance (closer is better)
        reward = -distance
        
        # Add penalty for excessive muscle activation to encourage efficiency
        muscle_activations = [physics.named.data.ctrl[actuator] for actuator in self._actuators]
        effort_penalty = -0.1 * np.sum(np.array(muscle_activations)**2)
        
        return 10 * reward + effort_penalty

    def should_terminate_episode(self, physics):
        hand_position = physics.named.data.geom_xpos["hand"]
        target_position = physics.named.data.mocap_pos["target"]
        distance = np.linalg.norm(hand_position - target_position)
        
        #print(f"Checking termination: distance={distance:.4f}, terminate={distance < 0.1}")
        return distance < 0.1



def run_simulation(env, duration=5.0):
    """Run a simulation and return collected data"""
    frames = []
    rewards = []
    observations = []
    ticks = []

    spec = env.action_spec()
    time_step = env.reset()

    while env.physics.data.time < duration:
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        frames.append(env.physics.render())
        rewards.append(time_step.reward)
        observations.append(copy.deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)

    return frames, rewards, observations, ticks

if __name__ == "__main__":
    # Create environment
    arm_model = mjcf.from_path("mujoco/arm_model.xml")
    arm_entity = ArmEntity(arm_model)
    task = ReachTargetTask(arm_entity)
    env = composer.Environment(task)

    # Run simulation
    frames, rewards, observations, ticks = run_simulation(env)

    # Display video
    display_video(frames, framerate=1./env.control_timestep())

    # Plot results
    num_sensors = len(observations[0])
    fig, axes = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
    
    # Plot rewards
    axes[0].plot(ticks, rewards)
    axes[0].set_ylabel('reward')
    axes[-1].set_xlabel('time')

    # Plot observations
    for i, key in enumerate(observations[0].keys()):
        data = [obs[key] for obs in observations]
        axes[i+1].plot(ticks, data)
        axes[i+1].set_ylabel(key)

    plt.show()

