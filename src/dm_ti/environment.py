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
        self._step_count = 0
        self._goal_reached = False
        self._decay_steps = 1000000  # Tune based on your training length
        self._goal_threshold = 0.08  # Match your termination condition
        self._target_dwell_count = 0
        self._target_dwell_required = 8  # Require 5 consecutive timesteps at target
        
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

        # Reset episode-specific reward tracking
        self._target_dwell_count = 0
        self._goal_reached = False
        self._step_count = 0
    
    def get_reward(self, physics):
        """Return the reward for the current state."""
        # Current state potential φ(s') = -distance
        hand_pos = physics.bind(self._arm.hand).xpos
        target_pos = physics.bind(self._arm.target).mocap_pos
        distance = np.linalg.norm(hand_pos - target_pos)
        #curr_pot = -distance

        #hand_vel = physics.bind(self._arm.hand).cvel[3:6]  # Linear velocity components
        #speed = np.linalg.norm(hand_vel)
        
        # Calculate proximity factor (0 far from target, 1 at target)
        #proximity = max(0, 1 - distance / (4 * self._goal_threshold))
        
        # Velocity penalty that scales with proximity to target
        #vel_penalty = -0.2 * speed * proximity**2  # Stronger effect very close to target

        # Potential-based shaping: F = γ·φ(s') - φ(s) 
        # Using tanh for smooth scaling instead of hard clipping
        #shaping = np.tanh(self._discount * curr_pot - self._prev_potential)
        #shaping = self._discount * curr_pot - self._prev_potential
        #shaping = np.clip(shaping, -1.0, 1.0)
        #self._prev_potential = curr_pot
        #distance_component = -0.05 * distance  # Direct feedback

        #shaping = - distance
        
        # Sparse terminal bonus (one-time)
        sparse = 0.0

        """proximity_bonus = 0.0
        if distance < self._goal_threshold:

            self._target_dwell_count += 1
            proximity_bonus = 0.1

            if self._target_dwell_count >= self._target_dwell_required:
                sparse = 1.5
            
        else:
            self._target_dwell_count = 0"""
            
            
        
        if distance < self._goal_threshold:
            #self._goal_reached = True
            shaping = 0.0
            sparse = 0.1
        else:
            # Outside target: quadratic penalty grows with distance
            excess_distance = distance - self._goal_threshold
            shaping = - 0.1 * excess_distance  # Quadratic penalty

        #distance_penalty = -0.1 * distance
        
        # Energy penalty (keep your existing one)
        energy_scale = 0.005
        energy = -energy_scale * np.sum(physics.data.act**2)
        
        # Compute annealing factor λ (from 1.0 → 0.0)
        #lambda_t = max(0.2, 1.0 - self._step_count / self._decay_steps)
        
        # Combine rewards with annealing
        #reward = sparse + lambda_t * shaping + energy 
        reward = sparse + shaping + energy #+ distance_penalty #+ stability_bonus + vel_penalty # Use this for debugging
        
        # Increment step counter
        #self._step_count += 1
        
        flag = False
        if int(physics.data.time / self.control_timestep) % 50 == 0 and flag:
            # Print debug information every NUM_SUBSTEPS
            print(#f"Step {self._step_count}, "
                  f"Distance: {distance:.4f}, "
                  f"Shaping: {shaping:.4f}, "
                  f"Sparse: {sparse:.4f}, "
                  #f"Energy: {energy:.4f}, "
                    #f"Distance penalty: {distance_penalty:.4f}, "
                  #f"Reward: {reward:.4f}"
                    #f"Velocity: {speed:.4f}, "
                    #f"Proximity: {proximity:.4f}, "
                    #f"Velocity penalty: {vel_penalty:.4f}, "
                  )
            
        return reward
    
    def should_terminate_episode(self, physics):
        """Determine if episode should end."""
        MIN_STEPS = 10
        
        hand_pos = physics.bind(self._arm.hand).xpos
        target_pos = physics.bind(self._arm.target).mocap_pos
        distance = np.linalg.norm(hand_pos - target_pos)
        
        # Check hand velocity
        #hand_vel = physics.bind(self._arm.hand).cvel[3:6]  # Linear velocity components
        #speed = np.linalg.norm(hand_vel)
        
        # Only terminate when:
        # 1. Minimum steps passed
        # 2. Within distance threshold
        # 3. Sufficient dwell time
        # 4. Low velocity (new condition)
        """if (int(physics.data.time / self.control_timestep) >= MIN_STEPS and 
                distance < self._goal_threshold and 
                self._target_dwell_count >= self._target_dwell_required):  # Speed must be low
            return True"""
        return False


def make_arm_env(random_seed=None):
    """Create and return a composer Environment with the arm task."""
    random_state = np.random.RandomState(random_seed)
    task = ArmReachingTask(random_state=random_state)
    env = composer.Environment(
        task=task,
        time_limit=2.0,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
    return env