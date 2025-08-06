import os
import pickle
import pandas as pd
import numpy as np

from dm_control import composer

from envs.entities import ArmEntity
from utils import get_root_path

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
            #print(f"Loaded {len(self._reachable_positions)} candidate target positions")
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
        self._goal_threshold = 0.04  # Match your termination condition
        self._target_dwell_count = 0
        self._target_dwell_required = 8  # Require 5 consecutive timesteps at target
        
        # Enable task observables
        for obs in self.observables.values():
            obs.enabled = True
    
    @property
    def root_entity(self):
        """Returns the root entity for this task."""
        return self._arm


    def _randomize_hand_mass(self, physics, random_state):
        """Randomize hand mass for this episode."""
        
        # Find the hand body index
        hand_body_id = physics.model.name2id('hand', 'body')
        
        # Original hand mass (sphere with r=0.05m, default density)
        # Volume = (4/3)π(0.05)³ ≈ 0.000524 m³
        # Default mass ≈ 0.524 grams
        base_mass = 0.0005  # 0.5 grams base mass
        
        # Vary from 0.2g (light/empty hand) to 2.0g (heavy object in hand)
        mass_multiplier = random_state.uniform(0.4, 4.0)  # 0.4x to 4x base mass
        new_mass = base_mass * mass_multiplier
        
        # Set the new mass
        physics.model.body_mass[hand_body_id] = new_mass
        
        # Store for diagnostics
        self._current_hand_mass = new_mass
        

    def initialize_episode(self, physics, random_state):
        """Initialize a new episode after physics compilation."""
        
        # 🏋️ RANDOMIZE HAND MASS (instead of gravity)
        self._randomize_hand_mass(physics, random_state)
        
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

        # Reset success bonus flag
        self._success_bonus_given = False
    


    def _randomize_gravity(self, physics, random_state):
        """Randomize gravity vector for this episode."""
        
        # Option 1: Random gravity magnitude with fixed downward direction
        #gravity_magnitude = random_state.uniform(5.0, 15.0)  # 0.5x to 1.5x Earth gravity
        #gravity_magnitude = 0.0
        #physics.model.opt.gravity[0] = 0.0
        #physics.model.opt.gravity[1] = -gravity_magnitude 
        #physics.model.opt.gravity[2] = 0.0
        
        # Gravity in 2D plane only 
        gravity_magnitude = random_state.uniform(8.0, 12.0)
        angle = random_state.uniform(0, 2*np.pi)  # Random angle in XY plane
        physics.model.opt.gravity[0] = gravity_magnitude * np.cos(angle)
        physics.model.opt.gravity[1] = gravity_magnitude * np.sin(angle)
        physics.model.opt.gravity[2] = 0.0  # No Z component
        
        # Store for diagnostics
        self._current_gravity = physics.model.opt.gravity.copy()

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
        #shaping = np.clip(shaping * 5, -1.0, 1.0)
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
            sparse = 0.2
        else:
            # Outside target: quadratic penalty grows with distance
            excess_distance = distance - self._goal_threshold
            shaping = - 0.1 * excess_distance  # Quadratic penalty

        #distance_penalty = -0.1 * distance
        
        # Energy penalty (keep your existing one)
        energy_scale = 0.01
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
    
    def get_reward1(self, physics):
        """Return the reward focused only on success and stability."""

        hand_pos = physics.bind(self._arm.hand).xpos
        target_pos = physics.bind(self._arm.target).mocap_pos
        distance = np.linalg.norm(hand_pos - target_pos)
        
        # Get hand velocity for stability check
        hand_vel = physics.bind(self._arm.hand).cvel[:3]  # Linear velocity
        speed = np.linalg.norm(hand_vel)
        

        
        # Track success bonus (this tells us everything we need to know)
        if not hasattr(self, '_success_bonus_given'):
            self._success_bonus_given = False
    
        currently_in_target = distance < self._goal_threshold
    
        if currently_in_target:
            # SUCCESS ZONE: Focus only on stability
            
            # ONE-TIME success bonus only on first entry
            if not self._success_bonus_given:
                success_reward = 3.0
                self._success_bonus_given = True
            else:
                success_reward = 0.0
            
            # Heavy velocity penalty inside target - must be stable
            velocity_penalty = -2.0 * speed
            
            # Bonus for being very stable
            stability_bonus = 0.1 if speed < 0.01 else 0.0
            
            reward = success_reward + velocity_penalty + stability_bonus
            
        else:
            # OUTSIDE TARGET ZONE: Only penalize if we're leaving
            
            if self._success_bonus_given:
                # LEAVING PENALTY: Severe punishment for exiting target
                leaving_penalty = -8.0
                reward = leaving_penalty 
    
        return reward

    def should_terminate_episode(self, physics):
        """Determine if episode should end."""
        # Fixed episode length 
        return False