import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from dm_testing.arm_env import load, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP
from dm_testing.dm_control_test import display_video
import os
import time

# Advanced parallelism settings
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

# PyTorch threading settings
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

# Enable MKL optimizations
torch.backends.mkldnn.enabled = True

class DMControlGymnasiumWrapper(gym.Env):
    """Convert dm_control environment to Gymnasium interface for Stable Baselines3."""
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode="rgb_array", verbose=False):
        """Initialize the wrapper."""
        self.env = load()
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Calculate max steps based on time limit and control timestep
        self.max_steps_per_episode = int(_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP)
        if self.verbose:
            print(f"Max steps per episode: {self.max_steps_per_episode} " +
                  f"(time limit: {_DEFAULT_TIME_LIMIT}s, timestep: {_CONTROL_TIMESTEP}s)")
        
        # Define action space (muscle activations between 0 and 1)
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        # Define observation space (flattened observations)
        total_size = 0
        for name, spec in self.env.observation_spec().items():
            if name == 'target_position':
                total_size += 3
            else:
                total_size += int(np.prod(spec.shape))
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
        
        # For tracking
        self.steps = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.target_changes = 0
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment and randomize target."""
        if seed is not None:
            np.random.seed(seed)
        
        time_step = self.env.reset()
        self.steps = 0
        self.episode_reward = 0
        self.episode_count += 1
        
        # FORCE TARGET RANDOMIZATION
        # Get access to reachable positions
        task = self.env.task
        
        if hasattr(task, '_reachable_positions') and len(task._reachable_positions) > 0:
            # Explicitly select a random target position
            num_positions = len(task._reachable_positions)
            target_idx = np.random.randint(0, num_positions)
            new_target_position = task._reachable_positions[target_idx]
            
            # Get the old target position if possible
            try:
                # In DM Control, mocap_pos is a structured array
                old_target = self.env.physics.named.data.mocap_pos["target"].copy()
                
                if self.verbose:
                    print(f"Episode {self.episode_count}: Target change - " +
                          f"From {old_target} to {new_target_position}")
                    
                # Only increment counter if we successfully changed targets
                if not np.array_equal(old_target, new_target_position):
                    self.target_changes += 1
            except (KeyError, IndexError, ValueError, AttributeError):
                if self.verbose:
                    print(f"Episode {self.episode_count}: Setting initial target to {new_target_position}")
            
            # Apply the new target position to the MuJoCo mocap body
            self.env.physics.named.data.mocap_pos["target"] = new_target_position
            
            # Update internal task distance tracking
            if hasattr(task, 'prev_distance'):
                hand_position = self.env.physics.named.data.geom_xpos["hand"]
                task.prev_distance = np.linalg.norm(hand_position - new_target_position)
        elif self.verbose:
            print("WARNING: Could not access reachable_positions to randomize target")
        
        observation = self._process_observation(time_step.observation)
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment."""
        time_step = self.env.step(action)
        self.steps += 1
        
        # Print debug info periodically
        if self.verbose and self.steps % 100 == 0:
            hand_position = self.env.physics.named.data.geom_xpos['hand']
            target_position = self.env.physics.named.data.mocap_pos['target']
            distance = np.linalg.norm(hand_position - target_position)
            print(f"Step {self.steps} (Ep {self.episode_count}):")
            print(f"  Hand: {hand_position}, Target: {target_position}")
            print(f"  Distance: {distance:.4f}, Y-plane: hand={hand_position[1]:.4f}, target={target_position[1]:.4f}")
            print(f"  Reward: {time_step.reward:.4f}")
        
        observation = self._process_observation(time_step.observation)
        reward = float(time_step.reward)
        self.episode_reward += reward
        
        # Check for episode termination:
        # 1. MuJoCo environment says episode is done (target reached or time limit)
        # 2. Explicit step count limit as backup
        terminated = time_step.last() 
        truncated = self.steps >= self.max_steps_per_episode
        
        info = {}
        if terminated or truncated:
            info['episode'] = {'r': self.episode_reward, 'l': self.steps, 'target_changes': self.target_changes}
            if self.verbose:
                print(f"Episode {self.episode_count} ended: " +
                      f"steps={self.steps}, reward={self.episode_reward:.2f}, " +
                      f"terminated={terminated}, truncated={truncated}")
        
        return observation, reward, terminated, truncated, info
    
    def _process_observation(self, observation):
        """Convert dm_control observation dict to flat array."""
        obs_list = []
        for name, obs in observation.items():
            if name == 'target_position':
                # Keep target position prominent in observations
                obs_list.append(obs.flatten()[:3].astype(np.float32))
            else:
                # Just flatten other observations
                obs_list.append(obs.flatten().astype(np.float32))
        
        # Simple concatenation - VecNormalize will handle normalization
        return np.concatenate(obs_list).astype(np.float32)
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.env.physics.render(camera_id=-1, width=640, height=480)
        return None
    
    def close(self):
        """Close the environment."""
        self.env.close()

class VideoRecorderCallback(BaseCallback):
    """Custom callback for saving videos during training."""
    
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        """Initialize the callback."""
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.timesteps = []
        
    def _on_step(self):
        """Called at each step."""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            obs, _ = self.eval_env.reset()
            frames = []
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Capture frames for video
                if step % 5 == 0 or step == 0:  # More frequent captures for smoother video
                    frames.append(self.eval_env.render())
                step += 1
            
            # Log reward
            self.episode_rewards.append(episode_reward)
            self.timesteps.append(self.num_timesteps)
            
            # Save video
            if len(frames) > 1:
                filename = f"sac_timestep_{self.num_timesteps}.gif"
                display_video(frames, filename=filename, framerate=30)
                print(f"Video saved as {filename} | Reward: {episode_reward:.2f}")
            
            # Save best model
            if episode_reward > self.best_mean_reward:
                self.best_mean_reward = episode_reward
                self.model.save("best_sac_model")
                print(f"New best model! Reward: {episode_reward:.2f}")
            
        return True

class TargetDebugCallback(BaseCallback):
    """Callback to track target changes."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.target_changes = 0
        self.episodes = 0
        self.last_info = None
    
    def _on_step(self):
        # Monitor info dictionaries for episode termination
        if self.locals.get('dones') is not None:
            dones = self.locals.get('dones')
            infos = self.locals.get('infos')
            
            # Process each environment that's done
            for i, done in enumerate(dones):
                if done and 'episode' in infos[i]:
                    self.episodes += 1
                    if 'target_changes' in infos[i]['episode']:
                        self.target_changes = max(self.target_changes, infos[i]['episode']['target_changes'])
                    
                    # Record last info for debugging
                    self.last_info = infos[i]
                    
                    # Print feedback at intervals
                    if self.episodes % 10 == 0:
                        print(f"Training progress: {self.episodes} episodes completed, " +
                              f"{self.target_changes} target changes, " +
                              f"avg reward: {infos[i]['episode']['r']:.2f}")
        
        return True

class RewardTrackingCallback(BaseCallback):
    """Callback to track rewards during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_rewards = []
        self.timesteps = []
        self.episode_count = 0
        self.running_reward = 0
    
    def _on_step(self):
        # Check for episode completion in any environment
        infos = self.locals.get('infos')
        if infos is not None:
            for info in infos:
                if 'episode' in info:
                    # Record the episode reward
                    self.episode_count += 1
                    episode_reward = info['episode']['r']
                    self.training_rewards.append(episode_reward)
                    self.timesteps.append(self.num_timesteps)
                    
                    # Log progress
                    if self.episode_count % 10 == 0:
                        recent_rewards = self.training_rewards[-10:]
                        avg_reward = sum(recent_rewards) / len(recent_rewards)
                        print(f"Episode {self.episode_count} | Avg reward: {avg_reward:.2f} | Step: {self.num_timesteps}")
                    
                    # Plot real-time learning curve every 50 episodes
                    if self.episode_count % 50 == 0 and len(self.training_rewards) > 1:
                        plt.figure(figsize=(10, 6))
                        plt.plot(self.timesteps, self.training_rewards)
                        plt.xlabel("Timesteps")
                        plt.ylabel("Reward")
                        plt.title(f"SAC Learning Curve - {self.episode_count} Episodes")
                        plt.grid(True)
                        plt.savefig("learning_progress.png")
                        plt.close()
        
        return True

def make_env(rank, verbose=False):
    """Create an environment wrapper."""
    def _init():
        # Only make the first environment verbose for cleaner logs
        env = DMControlGymnasiumWrapper(verbose=verbose and rank == 0)
        return Monitor(env)
    return _init

def train_sac(total_timesteps=200000):  # Increased to 200K steps
    """Train a SAC agent on the arm environment using multiple parallel environments."""
    # Create raw environment for checking
    raw_env = DMControlGymnasiumWrapper(verbose=True)
    check_env(raw_env)
    
    # Calculate number of environments based on CPU count
    # Use fewer environments than cores to prevent overload
    num_envs = max(1, min(os.cpu_count() - 2, 8))  # Cap at 8 envs to ensure quality
    print(f"Training with {num_envs} parallel environments")
    
    # Create environments using SubprocVecEnv for true parallelism
    env = SubprocVecEnv([make_env(i, verbose=i==0) for i in range(num_envs)])
    
    # Apply VecNormalize for automatic observation and reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,     # Normalize observations
        norm_reward=True,  # Normalize rewards
        clip_obs=10.0,     # Clip observations to reasonable ranges
        clip_reward=10.0,  # Clip rewards to avoid extremes
        gamma=0.99,        # Discount factor for reward normalization
    )
    
    # Create separate evaluation environment (non-vectorized)
    eval_env = DMControlGymnasiumWrapper(verbose=True)
    
    # Create the SAC agent with improved hyperparameters for better learning
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Lower learning rate
        buffer_size=1000000, # Much larger buffer
        batch_size=256,
        learning_starts=10000,  # More initial exploration
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 256],  # Deeper network
                qf=[512, 512, 256]
            ),
        )
    )
    
    # Create callbacks
    video_callback = VideoRecorderCallback(eval_env, eval_freq=20000)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./sac_checkpoints/",
        name_prefix="sac_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    target_debug_callback = TargetDebugCallback()
    reward_tracking_callback = RewardTrackingCallback()
    callbacks = [video_callback, checkpoint_callback, target_debug_callback, reward_tracking_callback]
    
    # Print training start information
    print("\n============ TRAINING CONFIGURATION ============")
    print(f"Time limit per episode: {_DEFAULT_TIME_LIMIT} seconds")
    print(f"Control timestep: {_CONTROL_TIMESTEP} seconds")
    print(f"Steps per episode: {int(_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP)}")
    print(f"Environments: {num_envs} parallel environments")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Estimated episodes: {total_timesteps / (int(_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP)):.1f}")
    print("================================================\n")
    
    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    total_time = time.time() - start_time
    
    # Print training summary
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Episodes completed: {target_debug_callback.episodes}")
    print(f"Target position changes: {target_debug_callback.target_changes}")
    print(f"Best reward achieved: {video_callback.best_mean_reward:.2f}")
    
    # Save the final model and normalization parameters
    model.save("final_sac_model")
    env.save("vec_normalize.pkl")
    
    # Plot learning curve after training
    if len(video_callback.episode_rewards) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(video_callback.timesteps, video_callback.episode_rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("SAC Learning Curve")
        plt.grid(True)
        plt.savefig("sac_learning_curve.png")
        plt.close()
    
    # After training, plot the final learning curve with proper data
    if len(reward_tracking_callback.training_rewards) > 1:
        plt.figure(figsize=(12, 8))
        
        # Plot with a smoothing window for readability
        rewards = np.array(reward_tracking_callback.training_rewards)
        timesteps = np.array(reward_tracking_callback.timesteps)
        
        # Raw data (light color)
        plt.plot(timesteps, rewards, 'c-', alpha=0.3, label='Raw rewards')
        
        # Smoothed curve (dark color)
        window_size = min(10, len(rewards)//5) if len(rewards) > 10 else 1
        if window_size > 1:
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = timesteps[window_size-1:]
            plt.plot(smoothed_steps, smoothed, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
        
        plt.xlabel("Timesteps", fontsize=14)
        plt.ylabel("Episode Reward", fontsize=14)
        plt.title("SAC Learning Curve", fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("sac_learning_curve.png", dpi=100)
        plt.close()

        print(f"Final learning curve saved with {len(rewards)} data points")
    
    # Record final performance
    obs, _ = eval_env.reset()
    frames = []
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        if step % 2 == 0:  # Even higher frequency captures for final demo
            frames.append(eval_env.render())
        step += 1
    
    display_video(frames, filename="final_performance.gif", framerate=30)
    print("Final performance video saved as 'final_performance.gif'")
    
    return model

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import stable_baselines3
        import gymnasium
    except ImportError:
        print("Installing required packages...")
        os.system("pip install gymnasium stable-baselines3[extra]")
        print("Packages installed. Please run the script again.")
        exit()
    
    # Print CPU info
    print(f"CPU Count: {os.cpu_count()} cores")
    
    # Test the environment
    env = DMControlGymnasiumWrapper(verbose=True)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Train the agent
    model = train_sac(total_timesteps=500000)  # 200k steps should show learning
    print(f"Training completed. Time limit: {_DEFAULT_TIME_LIMIT} seconds")