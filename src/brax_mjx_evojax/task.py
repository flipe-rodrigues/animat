import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from typing import Tuple, Dict, Any

class LimbReachingTask(VectorizedTask):
    """Task for 2-joint limb to reach a target."""
    
    def __init__(self, xml_file: str, max_steps: int = 100, target_change_prob: float = 0.1):
        """Initialize the task.
        
        Args:
            xml_file: Path to the MuJoCo XML file.
            max_steps: Maximum number of steps per episode.
            target_change_prob: Probability of changing the target position each step.
        """
        # Initialize the base class first (without parameters)
        super().__init__()
        
        self.xml_file = xml_file
        self.max_steps = max_steps
        self.target_change_prob = jnp.array(target_change_prob)
        
        # Define observation and action dimensions
        self.obs_dim = 15  # 12 sensors + 3D target position
        self.act_dim = 4   # 4 muscle activations
        
        # Load the MuJoCo model and put in MJX
        self.mj_model = mujoco.MjModel.from_xml_path(xml_file)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Create MJX model from MuJoCo model
        self.mjx_model = mujoco.mjx.put_model(self.mj_model)
        
        # Define target bounds
        self.target_bounds = jnp.array([
            [-0.5, 0.5],  # x
            [-0.5, 0.5],  # y
            [0.0, 0.3]    # z
        ])
        
        # Find relevant body indices
        hand_body_name = "hand"
        target_body_name = "target"
        self.hand_body_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
        self.target_body_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    
    @property
    def obs_shape(self) -> Tuple:
        """Observation shape."""
        return (self.obs_dim,)

    @property
    def act_shape(self) -> Tuple:
        """Action shape."""
        return (self.act_dim,)
    
    def reset_env(self, key: jnp.ndarray) -> Dict[str, Any]:
        """Reset a single environment.
        
        Args:
            key: JAX random key.
            
        Returns:
            Environment state dictionary.
        """
        # Create MJX data for this environment
        mjx_data = mujoco.mjx.make_data(self.mj_model)
        
        # Generate random target position
        key, target_key = jax.random.split(key)
        target_pos = self._generate_random_target(target_key)
        
        # Set target position
        mjx_data = self._set_target_position(mjx_data, target_pos)
        
        # Step simulation once to update sensors
        mjx_data = mujoco.mjx.step(self.mjx_model, mjx_data)
        
        # Return the environment state
        return {
            'mjx_data': mjx_data,
            'target_pos': target_pos,
            'steps': jnp.zeros(1, dtype=jnp.int32),
            'key': key
        }
    
    def get_obs(self, env_state: Dict[str, Any]) -> jnp.ndarray:
        """Get observations from environment state.
        
        Args:
            env_state: Environment state dictionary.
            
        Returns:
            Observations array.
        """
        # Get sensor readings
        sensor_data = env_state['mjx_data'].sensordata
        
        # Concatenate sensor data with target position
        obs = jnp.concatenate([sensor_data, env_state['target_pos']])
        
        return obs
    
    def step_env(self, env_state: Dict[str, Any], action: jnp.ndarray) -> Tuple[Dict[str, Any], jnp.ndarray, jnp.ndarray]:
        """Step a single environment.
        
        Args:
            env_state: Current environment state.
            action: Action to take (muscle activations).
            
        Returns:
            Tuple of (new_env_state, reward, done).
        """
        # Unpack state
        mjx_data = env_state['mjx_data']
        target_pos = env_state['target_pos']
        steps = env_state['steps']
        key = env_state['key']
        
        # Set muscle activations from action
        mjx_data = mjx_data.replace(ctrl=action)
        
        # Step the simulation
        mjx_data = mujoco.mjx.step(self.mjx_model, mjx_data)
        
        # Maybe change target position
        key, new_key = jax.random.split(key)
        change_target = jax.random.uniform(new_key) < self.target_change_prob
        
        key, target_key = jax.random.split(key)
        new_target_pos = jax.lax.cond(
            change_target,
            lambda: self._generate_random_target(target_key),
            lambda: target_pos
        )
        
        # Set new target position if changed
        mjx_data = jax.lax.cond(
            change_target,
            lambda: self._set_target_position(mjx_data, new_target_pos),
            lambda: mjx_data
        )
        
        # Get hand position from xpos (body Cartesian positions)
        hand_pos = mjx_data.xpos[self.hand_body_idx]
        
        # Calculate reward (negative distance to target)
        dist = jnp.linalg.norm(hand_pos - new_target_pos)
        reward = -dist
        
        # Check if episode is done
        steps = steps + 1
        done = steps >= self.max_steps
        
        # Create new state
        new_env_state = {
            'mjx_data': mjx_data,
            'target_pos': new_target_pos,
            'steps': steps,
            'key': key
        }
        
        return new_env_state, reward, done
    
    def _generate_random_target(self, key: jnp.ndarray) -> jnp.ndarray:
        """Generate a random target position within bounds."""
        keys = jax.random.split(key, 3)
        target_pos = jnp.array([
            jax.random.uniform(keys[0], minval=self.target_bounds[0, 0], maxval=self.target_bounds[0, 1]),
            jax.random.uniform(keys[1], minval=self.target_bounds[1, 0], maxval=self.target_bounds[1, 1]),
            jax.random.uniform(keys[2], minval=self.target_bounds[2, 0], maxval=self.target_bounds[2, 1])
        ])
        return target_pos
    
    def _set_target_position(self, mjx_data: mujoco.mjx.Data, target_pos: jnp.ndarray) -> mujoco.mjx.Data:
        """Set the position of the target body."""
        # Update the target mocap body position
        new_mocap_pos = mjx_data.mocap_pos.at[0].set(target_pos)
        return mjx_data.replace(mocap_pos=new_mocap_pos)
    
    def reset(self, key_list: jnp.ndarray) -> TaskState:
        """Reset environments (vectorized).
        
        Args:
            key_list: List of JAX random keys, one per environment.
            
        Returns:
            Initial task state.
        """
        # Initialize each environment
        env_states = jax.vmap(self.reset_env)(key_list)
        
        # Get observations for each environment
        obs = jax.vmap(self.get_obs)(env_states)
        
        # Create initial state
        state = TaskState()
        state.obs = obs
        state.reward = jnp.zeros(key_list.shape[0])
        state.done = jnp.zeros(key_list.shape[0], dtype=jnp.bool_)
        state.info = env_states
        
        return state
    
    def step(self, state: TaskState, action: jnp.ndarray) -> TaskState:
        """Take a step in all environments (vectorized).
        
        Args:
            state: Current task state.
            action: Actions to take for each environment.
            
        Returns:
            New task state.
        """
        # Step each environment
        new_env_states, rewards, dones = jax.vmap(self.step_env)(state.info, action)
        
        # Get observations for each environment
        obs = jax.vmap(self.get_obs)(new_env_states)
        
        # Create new state
        new_state = TaskState()
        new_state.obs = obs
        new_state.reward = rewards
        new_state.done = dones
        new_state.info = new_env_states
        
        return new_state
    
    def evaluate(self, params: jnp.ndarray, num_eval_episodes: int) -> Tuple[jnp.ndarray, Dict]:
        """Evaluate the policy parameters."""
        # Not implemented for simplicity, as we're focusing on training
        return jnp.array(0.0), {"score": jnp.array(0.0)}