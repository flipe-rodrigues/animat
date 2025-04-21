"""
MuJoCo MJX environment for 2-joint 4-muscle arm reaching task.
"""
import os
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx
import numpy as np

from typing import Tuple, Dict, Any, Optional, List

class ArmReachingEnv:
    def __init__(self, model_path: str, episode_length: int = 500, render: bool = False):
        """
        Initialize the arm reaching environment.
        
        Args:
            model_path: Path to the MuJoCo XML model file
            episode_length: Maximum number of steps per episode
            render: Whether to enable rendering
        """
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # Use mjx.device_put for MuJoCo MJX 3.3.1
        self.model_mjx = mujoco.mjx.put_model(self.model)
        
        # Create data instances
        self.data = mujoco.MjData(self.model)
        
        # Use make_data for MJX 3.3.1
        self.data_mjx = mujoco.mjx.make_data(self.model_mjx)
        
        # Environment parameters
        self.episode_length = episode_length
        self.render = render
        
        # Set up rendering if needed
        if self.render:
            self.renderer = mujoco.Renderer(self.model)
            self.viewer = None
            try:
                # MuJoCo 3.3.1 uses viewer.launch_passive if available
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except (AttributeError, ImportError):
                # Fallback if viewer module not available
                pass
        else:
            self.renderer = None
            self.viewer = None
        
        # Variables to track current state
        self.step_count = 0
        self.current_target = jnp.zeros(3)
        
        # Constants
        self.n_muscles = 4
        self.n_joints = 2
        self.n_sensors_per_muscle = 3  # length, velocity, force
        self.n_sensors = self.n_muscles * self.n_sensors_per_muscle
        self.input_dim = self.n_sensors + 3  # sensors + target xyz
        self.output_dim = self.n_muscles  # muscle activations
        
        # Sensor indices
        self.sensor_indices = {}
        for sensor_name in [
            # Muscle length sensors
            "shoulder_flexor_length", "shoulder_extensor_length", 
            "elbow_flexor_length", "elbow_extensor_length",
            # Muscle velocity sensors
            "shoulder_flexor_velocity", "shoulder_extensor_velocity",
            "elbow_flexor_velocity", "elbow_extensor_velocity",
            # Muscle force sensors
            "shoulder_flexor_force", "shoulder_extensor_force",
            "elbow_flexor_force", "elbow_extensor_force",
            # Position sensors
            "end_effector_pos", "target_pos"
        ]:
            try:
                sensor_id = self.model.sensor(sensor_name).id
                self.sensor_indices[sensor_name] = sensor_id
            except (AttributeError, KeyError) as e:
                # Handle case where sensor might not exist
                print(f"Warning: Sensor {sensor_name} not found: {e}")
                self.sensor_indices[sensor_name] = None
    
    def generate_random_target(self, key: jnp.ndarray) -> jnp.ndarray:
        """
        Generate a random target position within reachable space.
        
        Args:
            key: JAX random key
            
        Returns:
            3D target position (x, y, z)
        """
        # Create a random target position that's likely reachable
        # The arm's max reach is about 0.8 units (2 segments of 0.4 units)
        # Generate targets within a spherical space
        k1, k2 = jax.random.split(key)
        
        # Random radius between 0.3 and 0.7
        radius = 0.3 + 0.4 * jax.random.uniform(k1)
        
        # Random spherical coordinates
        theta = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)  # Azimuthal angle
        phi = jax.random.uniform(k2, minval=0, maxval=jnp.pi)       # Polar angle
        
        # Convert to Cartesian coordinates
        x = radius * jnp.sin(phi) * jnp.cos(theta)
        y = radius * jnp.sin(phi) * jnp.sin(theta)
        z = radius * jnp.cos(phi)
        
        return jnp.array([x, y, z])
    
    def set_target(self, target_pos: jnp.ndarray) -> None:
        """
        Set the target position in the simulation.
        
        Args:
            target_pos: 3D target position (x, y, z)
        """
        # Update the target site position in the data
        try:
            # MuJoCo 3.3.1 may use different accessors
            site_id = self.model.site("target").id
            self.data.site_xpos[site_id] = target_pos
        except (AttributeError, IndexError, KeyError):
            # Alternative access method if above doesn't work
            try:
                self.data.site("target").pos[:] = target_pos
            except (AttributeError, IndexError, KeyError) as e:
                print(f"Warning: Could not set target position: {e}")
                
        self.current_target = target_pos
    
    def reset(self, key: jnp.ndarray) -> jnp.ndarray:
        """
        Reset the environment and return the initial observation.
        
        Args:
            key: JAX random key
            
        Returns:
            Initial observation
        """
        # Reset data
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset MJX data
        self.data_mjx = mujoco.mjx.make_data(self.model_mjx)
        
        # Reset step counter
        self.step_count = 0
        
        # Generate new target
        k1, k2 = jax.random.split(key)
        target_pos = self.generate_random_target(k1)
        self.set_target(target_pos)
        
        # Get initial observation
        return self._get_observation()
    
    def _get_observation(self) -> jnp.ndarray:
        """
        Get the current observation.
        
        Returns:
            Observation vector containing all sensor readings and target position
        """
        # Safely collect all muscle sensors
        muscle_sensors = []
        
        # Length sensors
        for sensor_name in [
            "shoulder_flexor_length", "shoulder_extensor_length",
            "elbow_flexor_length", "elbow_extensor_length",
            # Velocity sensors
            "shoulder_flexor_velocity", "shoulder_extensor_velocity",
            "elbow_flexor_velocity", "elbow_extensor_velocity",
            # Force sensors
            "shoulder_flexor_force", "shoulder_extensor_force",
            "elbow_flexor_force", "elbow_extensor_force"
        ]:
            try:
                # MuJoCo 3.3.1 compatible sensor access
                sensor_value = self.data.sensor(sensor_name).data[0]
                muscle_sensors.append(sensor_value)
            except (AttributeError, IndexError, KeyError) as e:
                # Fallback to a default value
                print(f"Warning: Could not read sensor {sensor_name}: {e}")
                muscle_sensors.append(0.0)
        
        muscle_sensors = jnp.array(muscle_sensors)
        
        # Combine muscle sensors with target position
        observation = jnp.concatenate([muscle_sensors, self.current_target])
        
        return observation
    
    def _get_reward(self) -> float:
        """
        Calculate the reward based on distance to target.
        
        Returns:
            Negative Euclidean distance between end effector and target
        """
        # Get end effector and target positions
        try:
            end_effector_pos = self.data.sensor("end_effector_pos").data
            target_pos = self.data.sensor("target_pos").data
        except (AttributeError, IndexError, KeyError) as e:
            # Alternative access if the above fails
            try:
                ee_id = self.model.site("end_effector").id
                target_id = self.model.site("target").id
                end_effector_pos = self.data.site_xpos[ee_id]
                target_pos = self.data.site_xpos[target_id]
            except (AttributeError, IndexError, KeyError) as e:
                print(f"Warning: Could not get positions for reward calculation: {e}")
                return 0.0
        
        # Calculate Euclidean distance
        distance = jnp.sqrt(jnp.sum((end_effector_pos - target_pos)**2))
        
        # Return negative distance as reward (closer is better)
        return -float(distance)
    
    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Muscle activations (4 values between 0 and 1)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Apply the muscle activations - adjust for MuJoCo 3.3.1 API
        try:
            # First approach: access via actuator name
            self.data.actuator("shoulder_flexor").ctrl = action[0]
            self.data.actuator("shoulder_extensor").ctrl = action[1]
            self.data.actuator("elbow_flexor").ctrl = action[2]
            self.data.actuator("elbow_extensor").ctrl = action[3]
        except (AttributeError, IndexError, KeyError):
            # Alternative: set the entire ctrl array
            try:
                # Check if numpy array needs to be converted
                if isinstance(action, jnp.ndarray):
                    action_np = np.array(action)
                else:
                    action_np = action
                self.data.ctrl[:] = action_np
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not apply actions: {e}")
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get the updated observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Update step counter
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.episode_length
        
        # Additional info
        info = {
            "distance": -reward,
            "end_effector_pos": np.zeros(3),
            "target_pos": np.array(self.current_target),
        }
        
        # Get end effector position if possible
        try:
            info["end_effector_pos"] = np.array(self.data.sensor("end_effector_pos").data)
        except (AttributeError, IndexError, KeyError):
            try:
                ee_id = self.model.site("end_effector").id
                info["end_effector_pos"] = np.array(self.data.site_xpos[ee_id])
            except (AttributeError, IndexError, KeyError) as e:
                print(f"Warning: Could not get end effector position for info: {e}")
        
        # Render if needed
        if self.render:
            if self.viewer is not None:
                self.viewer.sync()
            elif self.renderer is not None:
                # Alternative rendering method if viewer not available
                self.renderer.update_scene(self.data)
        
        return observation, reward, done, info
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    # JAX-compatible versions of the environment functions
    @jax.jit
    def reset_jax(self, key: jnp.ndarray, data_mjx) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Reset the environment (JAX version).
        
        Args:
            key: JAX random key
            data_mjx: MuJoCo MJX data
            
        Returns:
            Tuple of (observation, updated data, target position)
        """
        # Reset data to defaults
        data_mjx = mujoco.mjx.make_data(self.model_mjx)
        
        # Generate new target
        k1, k2 = jax.random.split(key)
        target_pos = self.generate_random_target(k1)
        
        # Get the site_pos indices for the target site
        try:
            target_site_id = self.model_mjx.site("target").id
            # Update the target site position for MJX 3.3.1
            data_mjx = data_mjx.replace(site_xpos=data_mjx.site_xpos.at[target_site_id].set(target_pos))
        except (AttributeError, IndexError, KeyError) as e:
            print(f"Warning: Could not set target position in MJX: {e}")
        
        # Get observation
        observation = self._get_observation_jax(data_mjx, target_pos)
        
        return observation, data_mjx, target_pos
    
    def _get_observation_jax(self, data_mjx, target_pos: jnp.ndarray) -> jnp.ndarray:
        """
        Get the current observation (JAX version).
        
        Args:
            data_mjx: MuJoCo MJX data
            target_pos: Current target position
            
        Returns:
            Observation vector
        """
        # Extract all sensor readings - adjust for MJX 3.3.1
        sensor_data = data_mjx.sensordata
        
        # First 12 entries are our muscle sensors
        muscle_sensors = sensor_data[:12]
        
        # Concatenate with target position
        observation = jnp.concatenate([muscle_sensors, target_pos])
        
        return observation
    
    def _get_reward_jax(self, data_mjx, target_pos: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the reward (JAX version).
        
        Args:
            data_mjx: MuJoCo MJX data
            target_pos: Target position
            
        Returns:
            Negative distance between end effector and target
        """
        # Get sensor indices
        ee_sensor_id = self.sensor_indices.get("end_effector_pos")
        if ee_sensor_id is None:
            # Fallback if sensor not available
            return jnp.array(0.0)
        
        # Extract positions from sensor data
        ee_pos = data_mjx.sensordata[ee_sensor_id:ee_sensor_id+3]
        
        # Calculate distance to target position directly
        distance = jnp.sqrt(jnp.sum((ee_pos - target_pos)**2))
        
        return -distance
    
    @jax.jit
    def step_jax(self, data_mjx, action: jnp.ndarray, target_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Take a step in the environment (JAX version).
        
        Args:
            data_mjx: MuJoCo MJX data
            action: Muscle activations (4 values between 0 and 1)
            target_pos: Current target position
            
        Returns:
            Tuple of (next_observation, reward, updated_data, info)
        """
        # Apply the muscle activations - MJX 3.3.1 compatible
        data_mjx = data_mjx.replace(ctrl=action)
        
        # Step the simulation
        data_mjx = mujoco.mjx.step(self.model_mjx, data_mjx)
        
        # Get the updated observation
        observation = self._get_observation_jax(data_mjx, target_pos)
        
        # Calculate reward
        reward = self._get_reward_jax(data_mjx, target_pos)
        
        # Additional info
        ee_sensor_id = self.sensor_indices.get("end_effector_pos")
        ee_pos = jnp.zeros(3)
        if ee_sensor_id is not None:
            ee_pos = data_mjx.sensordata[ee_sensor_id:ee_sensor_id+3]
        
        info = {
            "distance": -reward,
            "end_effector_pos": ee_pos,
            "target_pos": target_pos,
        }
        
        return observation, reward, data_mjx, info
    
    @jax.jit
    def rollout_jax(self, 
                   key: jnp.ndarray, 
                   policy_fn, 
                   policy_params, 
                   n_steps: int = None) -> Dict[str, jnp.ndarray]:
        """
        Perform a complete episode rollout with a policy.
        
        Args:
            key: JAX random key
            policy_fn: Function that maps (parameters, observation, hidden_state) to (action, next_hidden_state)
            policy_params: Parameters for the policy
            n_steps: Number of steps to simulate (defaults to episode_length)
            
        Returns:
            Dictionary with results of the rollout
        """
        if n_steps is None:
            n_steps = self.episode_length
        
        # Initialize environment
        key, subkey = jax.random.split(key)
        observation, data_mjx, target_pos = self.reset_jax(subkey, None)
        
        # Initialize hidden state for RNN
        hidden_state = jnp.zeros((policy_params['w_hh'].shape[0],))
        
        # Initialize arrays to collect trajectory data
        rewards = jnp.zeros(n_steps)
        distances = jnp.zeros(n_steps)
        
        # Define scan function for rollout
        def rollout_step(carry, t):
            obs, data, h_state, tgt_pos, cumulative_reward = carry
            
            # Get action from policy
            action, h_state = policy_fn(policy_params, obs, h_state)
            
            # Step environment
            next_obs, reward, next_data, info = self.step_jax(data, action, tgt_pos)
            
            # Update cumulative reward
            cumulative_reward += reward
            
            # Store metrics
            rewards_t = rewards.at[t].set(reward)
            distances_t = distances.at[t].set(info["distance"])
            
            # Return updated state
            return (next_obs, next_data, h_state, tgt_pos, cumulative_reward), (rewards_t[t], distances_t[t])
        
        # Initial state
        initial_state = (observation, data_mjx, hidden_state, target_pos, 0.0)
        
        # Perform rollout
        (final_obs, final_data, final_h_state, _, total_reward), (rewards, distances) = jax.lax.scan(
            rollout_step, initial_state, jnp.arange(n_steps)
        )
        
        # Return results
        return {
            "total_reward": total_reward,
            "rewards": rewards,
            "distances": distances,
            "mean_distance": jnp.mean(distances),
            "final_distance": distances[-1]
        }