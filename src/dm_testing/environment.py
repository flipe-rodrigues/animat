import numpy as np
from dm_control import mujoco, composer
from dm_control.composer.observation import observable
from dm_control import mjcf
import os
import pickle
import pandas as pd


def get_root_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path

NUM_SUBSTEPS = 10

class ArmEntity(composer.Entity):
    """A robotic arm that can reach targets."""
    
    def _build(self):
        # Create the MJCF model
        model_path = os.path.join(get_root_path(), "mujoco", "arm.xml")
        self._mjcf_root = mjcf.from_path(model_path)
        
        # Store references to key components
        self._target = self._mjcf_root.find('body', 'target')
        self._hand = self._mjcf_root.find('body', 'hand')
        
    def _build_observables(self):
        return ArmObservables(self)
    
    @property
    def mjcf_model(self):
        return self._mjcf_root
    
    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))
    
    @property
    def target(self):
        return self._target
    
    @property
    def hand(self):
        return self._hand

class ArmObservables(composer.Observables):
    """Observables for the arm entity."""
    
    @composer.observable
    def muscle_sensors(self):
        # Collect all sensors on the MJCF model
        sensors = self._entity._mjcf_root.find_all('sensor')
        # 'sensordata' is the MuJoCo name for physics.data.sensordata
        return observable.MJCFFeature('sensordata', sensors)

    @composer.observable
    def target_position(self):
        # Returns the mocap_pos of the target body
        return observable.MJCFFeature('mocap_pos', [self._entity._target])
    


class ArmReachingTask(composer.Task):
    """Task for controlling a 2D arm to reach targets."""
    
    def __init__(self, random_state=None):
        # Create the arm entity
        self._arm = ArmEntity()
        
        # Store random state
        self.random_state = random_state
        
        # Set appropriate time steps
        self.set_timesteps(control_timestep=0.02, physics_timestep=0.002)
        
        
        # Keep muscle sensors and target position enabled
        self._arm.observables.muscle_sensors.enabled = True
        self._arm.observables.target_position.enabled = True
        
        # Load candidate targets
        mj_dir = os.path.join(get_root_path(), "mujoco")
        candidate_targets_path = os.path.join(mj_dir, "candidate_targets.pkl")
        try:
            with open(candidate_targets_path, "rb") as f:
                data = pickle.load(f)
                # Convert DataFrame to list of numpy arrays if needed
                if isinstance(data, pd.DataFrame):
                    # Convert each row to a position array
                    self._reachable_positions = []
                    for _, row in data.iterrows():
                        # Get x, y, z coordinates
                        pos = np.array([row['x'], row['y'], row['z']])
                        self._reachable_positions.append(pos)
                else:
                    # Already in expected format
                    self._reachable_positions = data
            print(f"Loaded {len(self._reachable_positions)} candidate target positions")
        except Exception as e:
            print(f"WARNING: Failed to load candidate_targets.pkl ({e}). "
                  "Falling back to uniform grid of targets. "
                  "This may cause a distribution mismatch between training and evaluation!")
            # Create fallback targets in a grid pattern
            self._reachable_positions = []
            for r in np.linspace(0.2, 0.5, 5):
                for theta in np.linspace(0, 2*np.pi, 8):
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    self._reachable_positions.append(np.array([x, y, 0.05]))
            print(f"Generated {len(self._reachable_positions)} fallback target positions")
        
        # Initialize reward shaping state
        self._discount = 0.99
        self._prev_potential = None
        
        # Enable task observables
        for obs in self.observables.values():
            obs.enabled = True
    
    @property
    def root_entity(self):
        """Returns the root entity for this task."""
        return self._arm


    def initialize_episode(self, physics, random_state):
        """Initialize a new episode after physics compilation."""
        # Reset the arm to a randomized position (-60 to 60 degrees)
        with physics.reset_context():
            # Convert -60 to 60 degrees to radians
            min_angle = np.deg2rad(-60)
            max_angle = np.deg2rad(60)
            
            # Sample random angles for each joint
            shoulder_angle = random_state.uniform(min_angle, max_angle)
            elbow_angle = random_state.uniform(min_angle, max_angle)
            
            # Set the joint positions
            physics.named.data.qpos['shoulder'] = shoulder_angle
            physics.named.data.qpos['elbow'] = elbow_angle
            physics.named.data.qvel[:] = 0.0
        
        # Select a random target position
        if len(self._reachable_positions) > 0:
            idx = random_state.randint(0, len(self._reachable_positions))
            target_pos = self._reachable_positions[idx]
            
            # Set the target position using the physics engine
            physics.bind(self._arm.target).mocap_pos[:] = target_pos
        
        # Store initial distance for reward calculation
        hand_pos = physics.bind(self._arm.hand).xpos
        target_pos = physics.bind(self._arm.target).mocap_pos
        self._prev_potential = -np.linalg.norm(hand_pos - target_pos)
    
    def get_reward(self, physics):
        """Return the reward for the current state."""
        # Current state potential φ(s') = -distance
        hand_pos = physics.bind(self._arm.hand).xpos
        target_pos = physics.bind(self._arm.target).mocap_pos
        distance = np.linalg.norm(hand_pos - target_pos)
        curr_pot = -distance

        # Potential-based shaping: F = γ·φ(s') - φ(s)
        shaping = self._discount * curr_pot - self._prev_potential
        
        # Cap shaping reward to avoid overwhelming the success reward
        shaping = np.clip(shaping, -1.0, 1.0)

        # True reward: success bonus + energy penalty
        success = 1.0 if distance < 0.08 else 0.0
        energy = -0.001 * np.sum(physics.data.act**2)

        # Update stored potential for next step
        self._prev_potential = curr_pot

        # Return total reward
        return success + energy + shaping
    
    def should_terminate_episode(self, physics):
        """Determine if episode should end."""
        # Only terminate if target reached after a minimum number of steps
        #MIN_STEPS = 1  # Adjust as needed
        
        hand_pos = physics.bind(self._arm.hand).xpos
        target_pos = physics.bind(self._arm.target).mocap_pos
        distance = np.linalg.norm(hand_pos - target_pos)
        
        # Only allow termination after minimum steps
        if distance < 0.08:
        #if physics.data.time / self.control_timestep >= MIN_STEPS and distance < 0.08:
            return True
        return False


def make_arm_env(random_seed=None):
    """Create and return a composer Environment with the arm task."""
    random_state = np.random.RandomState(random_seed)
    task = ArmReachingTask(random_state=random_state)
    env = composer.Environment(
        task=task,
        time_limit=3.0,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
    return env