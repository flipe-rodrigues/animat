import os
import pickle
import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
from functools import partial
import numpy as np
import time

from utils import get_root_path, l1_norm, l2_norm

class MJXSequentialReacher:
    def __init__(self, model_path="arm_model.xml", num_envs=128, num_targets=5, target_duration=3.0,
                 curriculum_level=0, use_curriculum=False):
        """Initialize MJX-based sequential reaching environment."""
        # Add curriculum parameters
        self.use_curriculum = use_curriculum
        self.curriculum_level = curriculum_level
        self.success_rate_threshold = 0.8  # Advance when 80% successful
        self.curriculum_success_counter = 0
        self.curriculum_total_counter = 0
        
        # Load MuJoCo model
        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, model_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        
        # Convert to MJX model for acceleration
        self.mjx_model = mjx.put_model(self.model)
        
        # Find hand geom ID for tracking
        self.hand_id = self.model.geom("hand").id
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.num_envs = num_envs
        
        # Load statistics for normalization
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)
        
        target_stats_path = os.path.join(mj_dir, "target_stats.pkl")
        with open(target_stats_path, "rb") as f:
            self.target_stats = pickle.load(f)
        
        # Load valid target positions
        reachable_positions_path = os.path.join(mj_dir, "reachable_positions.pkl")
        with open(reachable_positions_path, "rb") as f:
            self.reachable_positions = pickle.load(f)
        
        # Additional parameters
        self.num_targets = num_targets  # Fewer targets for faster completion
        self.target_duration = target_duration  # Shorter target duration
        self.timestep = float(self.model.opt.timestep)
        self.max_episode_steps = int(target_duration * num_targets / self.timestep)
        self.current_steps = jnp.zeros(num_envs, dtype=jnp.int32)
        
        # Initialize random key for sampling
        self.key = random.PRNGKey(0)
        
        # Define curriculum stages (fixed positions for each level) - MOVE THIS BEFORE RESET
        self.curriculum_targets = {
            # Level 0: Just left and right positions
            0: jnp.array([
                [-0.3, 0.0, 0.3],  # Left position 
                [0.3, 0.0, 0.3],   # Right position
            ]),
            # Level 1: Add top and bottom
            1: jnp.array([
                [-0.3, 0.0, 0.3],  # Left
                [0.3, 0.0, 0.3],   # Right
                [0.0, 0.0, 0.5],   # Top
                [0.0, 0.0, 0.1],   # Bottom
            ]),
            # Level 2: Add diagonal positions
            2: jnp.array([
                [-0.3, 0.0, 0.3],  # Left
                [0.3, 0.0, 0.3],   # Right
                [0.0, 0.0, 0.5],   # Top
                [0.0, 0.0, 0.1],   # Bottom
                [-0.2, 0.0, 0.5],  # Top-left
                [0.2, 0.0, 0.5],   # Top-right
                [-0.2, 0.0, 0.1],  # Bottom-left
                [0.2, 0.0, 0.1],   # Bottom-right
            ]),
            # Level 3: Full random sampling from reachable positions
            3: None  # Will use self.reachable_positions
        }
        
        # Initialize environments - NOW AFTER defining curriculum_targets
        self.reset()
        
    def reset(self, key=None):
        """Reset all environments and sample new targets"""
        # Handle explicit random keys for proper randomization
        if key is None:
            # Create a new random key if none provided (important for evaluation)
            key = random.PRNGKey(int(time.time() * 1000) % (2**32))
        
        # Split key for different random operations
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Initialize environments with randomized states
        self.mjx_data = self._create_mjx_data(subkey1)
        
        # Sample random targets (different for each episode)
        self.targets = self._sample_targets(subkey2)
        
        # Reset step counters
        self.current_steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
        
        # Update targets in environment
        self.update_targets()
        
        # Return initial observations
        return self.get_obs()
    
    def update_targets(self):
        """Update targets for all environments based on current target index"""
        # Get current target for each environment
        batch_idx = jnp.arange(self.num_envs)
        targets = self.targets[batch_idx, self.current_target_idx]
        
        # In MJX, we need to update mocap bodies
        # This requires setting the mocap_pos in the mjx_data structure
        def set_mocap_pos(data, target):
            return data.replace(mocap_pos=data.mocap_pos.at[0].set(target))
        
        # Apply the update to all environments
        self.mjx_data = jax.vmap(set_mocap_pos)(self.mjx_data, targets)
    
    def get_obs(self):
        """Get observations from all environments with consistent normalization"""
        # Extract sensor data from all environments
        sensor_data = self.mjx_data.sensordata
        
        # Get current targets for each environment
        batch_idx = jnp.arange(self.num_envs)
        targets = self.targets[batch_idx, self.current_target_idx]
        
        # Normalize BOTH sensors and targets using pre-computed statistics
        sensor_mean = jnp.array(self.sensor_stats["mean"].values)
        sensor_std = jnp.array(self.sensor_stats["std"].values)
        normalized_sensors = (sensor_data - sensor_mean) / jnp.maximum(sensor_std, 1e-6)
        
        # Also normalize target positions
        target_mean = jnp.array(self.target_stats["mean"].values)
        target_std = jnp.array(self.target_stats["std"].values)
        normalized_targets = (targets - target_mean) / jnp.maximum(target_std, 1e-6)
        
        # Concatenate normalized target positions and sensor data
        observations = jnp.concatenate([normalized_targets, normalized_sensors], axis=1)
        
        return observations
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, actions):
        """Step all environments in parallel with the given actions"""
        # Ensure actions are properly shaped for batching
        if isinstance(actions, np.ndarray):
            actions = jnp.asarray(actions)
        
        # Step the physics using pure functions
        next_mjx_data, hand_positions, distances, rewards, next_current_steps, next_target_idx, dones, reward_components = self._step_impl(
            self.mjx_data, 
            actions, 
            self.current_steps,
            self.current_target_idx,
            self.targets
        )
        
        # Update state
        self.mjx_data = next_mjx_data
        self.current_steps = next_current_steps
        self.current_target_idx = next_target_idx
        
        # Get new observations
        next_obs = self.get_obs()
        
        # Properly structure the info dictionary with reward components
        info = {
            "euclidean_distances": distances["euclidean"],
            "manhattan_distances": distances["manhattan"],
            "energies": distances["energies"],
            "current_target_idx": self.current_target_idx,
            "reward_components": reward_components,
            "target_reached": distances["euclidean"] < 0.1  # Add this to info
        }
        
        # Return results - DON'T call update_curriculum here
        return next_obs, rewards, dones, info
    
    def _step_impl(self, mjx_data, actions, current_steps, current_target_idx, targets):
        """Implementation of step logic as pure functions"""
        # Set muscle activations
        mjx_data = jax.vmap(lambda d, a: d.replace(ctrl=a))(mjx_data, actions)
        
        # Step the physics simulation - fixed implementation
        mjx_data = jax.vmap(lambda d: mjx.step(self.mjx_model, d))(mjx_data)
        
        # Get hand positions
        hand_positions = jax.vmap(lambda d: d.geom_xpos[self.hand_id])(mjx_data)
        
        # Get current targets
        batch_idx = jnp.arange(self.num_envs)
        current_targets = targets[batch_idx, current_target_idx]
        
        # Calculate distances
        euclidean_distances = jax.vmap(l2_norm)(current_targets - hand_positions)
        manhattan_distances = jax.vmap(l1_norm)(current_targets - hand_positions)
        energies = jnp.mean(actions, axis=1)
        
        # Track previous distances for progress rewards (add as state variable)
        if not hasattr(self, 'prev_distances'):
            self.prev_distances = jnp.ones(self.num_envs) * 1.0  # Initialize with a default
        
        # Calculate progress (improvement in distance)
        progress = self.prev_distances - euclidean_distances
        # Update previous distances for next step
        self.prev_distances = euclidean_distances
        
        # Better reward shaping
        # 1. Distance component - scaled properly
        distance_reward = -euclidean_distances * 2.0
        
        # 2. Progress component - reward improvement
        progress_reward = jnp.clip(progress * 5.0, -1.0, 1.0)
        
        # 3. Energy efficiency - penalize high activations more
        energy_penalty = -energies * 0.5
        
        # 4. Target reached bonus - scaled by remaining time
        target_reached = euclidean_distances < 0.1
        steps_remaining = self.max_episode_steps - current_steps
        time_efficiency_factor = steps_remaining / self.max_episode_steps
        reach_bonus = jnp.where(target_reached, 10.0 * time_efficiency_factor, 0.0)
        
        # Combine all reward components
        rewards = reach_bonus + distance_reward + progress_reward + energy_penalty
        
        # Increment step counter
        next_steps = current_steps + 1
        
        # Check if target needs to be updated
        target_timeout = next_steps * self.timestep > self.target_duration
        update_mask = jnp.logical_or(target_reached, target_timeout)
        
        # Update target index
        next_target_idx = jnp.where(
            update_mask,
            jnp.minimum(current_target_idx + 1, self.num_targets - 1), 
            current_target_idx
        )
        
        # Update mocap bodies with new targets
        next_targets = targets[batch_idx, next_target_idx]
        mjx_data = jax.vmap(lambda d, t: d.replace(mocap_pos=d.mocap_pos.at[0].set(t)))(mjx_data, next_targets)
        
        # Check if episodes are done
        dones = jnp.logical_or(
            next_steps >= self.max_episode_steps,
            next_target_idx >= self.num_targets - 1
        )
        
        
        # Return all updated state components
        return mjx_data, hand_positions, {
            "euclidean": euclidean_distances, 
            "manhattan": manhattan_distances,
            "energies": energies
        }, rewards, next_steps, next_target_idx, dones, {
            "distance_reward": distance_reward,
            "progress_reward": progress_reward,
            "energy_penalty": energy_penalty,
            "reach_bonus": reach_bonus
        }
    
    def render(self, env_idx=0):
        """Render a single environment for visualization"""
        import mujoco.viewer
        
        # Extract state from MJX data for the specified environment
        mjx_data_single = jax.tree_util.tree_map(lambda x: np.array(x[env_idx]), self.mjx_data)
        
        # Create a MuJoCo data instance for rendering
        render_data = mujoco.MjData(self.model)
        
        # Copy state from MJX data
        render_data.qpos = mjx_data_single.qpos
        render_data.qvel = mjx_data_single.qvel
        render_data.ctrl = mjx_data_single.ctrl
        render_data.mocap_pos = mjx_data_single.mocap_pos
        
        # Render using MuJoCo viewer
        with mujoco.viewer.launch_passive(self.model, render_data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            viewer.cam.lookat[:] = [0, -1.5, -0.5]
            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
            viewer.sync()
            return viewer

    def _create_mjx_data(self, key):
        """Create MJX data objects for all environments with randomized initial states."""
        # Create keys for each environment
        keys = random.split(key, self.num_envs)
        
        # Create basic MJX data
        mjx_data = mjx.make_data(self.mjx_model)
        
        # Vectorize state randomization across environments
        def randomize_state(key, data):
            # Split key for different random operations
            key1, key2 = random.split(key)
            
            # Small random perturbation around the default position
            qpos_noise = random.uniform(key1, shape=data.qpos.shape, minval=-0.1, maxval=0.1)
            qpos = data.qpos + qpos_noise
            
            # Small initial velocities
            qvel_noise = random.uniform(key2, shape=data.qvel.shape, minval=-0.01, maxval=0.01)
            
            # Return updated data
            return data.replace(qpos=qpos, qvel=qvel_noise)
        
        # Apply randomization to each environment
        mjx_data_batch = jax.vmap(lambda k: randomize_state(k, mjx_data))(keys)
        
        return mjx_data_batch

    def _sample_targets(self, key):
        """Sample targets based on curriculum level."""
        # Split the key for different environments
        keys = random.split(key, self.num_envs)
        
        # Create empty array for targets
        all_targets = jnp.zeros((self.num_envs, self.num_targets, 3))
        
        if self.use_curriculum and self.curriculum_level < 3:
            # Use curriculum-defined targets
            curriculum_positions = self.curriculum_targets[self.curriculum_level]
            num_positions = len(curriculum_positions)
            
            # Function to sample from curriculum targets
            def sample_curriculum_targets(key):
                # Sample random indices from curriculum positions
                indices = random.choice(
                    key,
                    a=num_positions,
                    shape=(self.num_targets,),
                    replace=True
                )
                # Get the positions for these indices
                return curriculum_positions[indices]
            
            # Sample targets for all environments
            all_targets = jax.vmap(sample_curriculum_targets)(keys)
        else:
            # Original full sampling from reachable positions
            valid_positions = jnp.array(self.reachable_positions.values)
            num_positions = len(valid_positions)
            
            # Function to sample targets for a single environment
            def sample_env_targets(key):
                # Sample random indices
                indices = random.choice(
                    key, 
                    a=num_positions, 
                    shape=(self.num_targets,), 
                    replace=True
                )
                # Get the positions for these indices
                return valid_positions[indices]
            
            # Sample targets for all environments
            all_targets = jax.vmap(sample_env_targets)(keys)
        
        # Reset target index
        self.current_target_idx = jnp.zeros(self.num_envs, dtype=jnp.int32)
        
        return all_targets

    def update_curriculum(self, reached_targets):
        """Update curriculum based on performance.
        
        Args:
            reached_targets: Boolean array indicating which targets were reached
        """
        if not self.use_curriculum or self.curriculum_level >= 3:
            return
        
        # Convert from JAX to numpy if needed
        if hasattr(reached_targets, 'shape'):
            reached_targets = np.array(reached_targets)
        
        # Update success counters
        self.curriculum_success_counter += np.sum(reached_targets)
        self.curriculum_total_counter += reached_targets.size
        
        # Calculate success rate
        if self.curriculum_total_counter >= 100:  # Wait for enough samples
            success_rate = self.curriculum_success_counter / self.curriculum_total_counter
            
            # Advance curriculum if performance is good enough
            if success_rate >= self.success_rate_threshold:
                self.curriculum_level = min(3, self.curriculum_level + 1)
                print(f"Advancing to curriculum level {self.curriculum_level}!")
                
                # Reset counters for next level
                self.curriculum_success_counter = 0
                self.curriculum_total_counter = 0