import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC  
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from dm_testing.arm_env import load, _DEFAULT_TIME_LIMIT
from dm_testing.dm_control_test import display_video
import os

# Set environment variables for Intel MKL optimizations
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

class DMControlGymnasiumWrapper(gym.Env):
    """Convert dm_control environment to Gymnasium interface for Stable Baselines3."""
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode="rgb_array"):
        """Initialize the wrapper."""
        self.env = load()
        self.render_mode = render_mode
        
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
                # Only use the first 3 values from target_position
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
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        time_step = self.env.reset()
        self.steps = 0
        self.episode_reward = 0
        
        return self._process_observation(time_step.observation), {}
    
    def step(self, action):
        """Take a step in the environment."""
        time_step = self.env.step(action)
        self.steps += 1
        
        observation = self._process_observation(time_step.observation)
        reward = float(time_step.reward)
        self.episode_reward += reward
        terminated = time_step.last()
        truncated = False  # dm_control doesn't have a concept of truncation
        info = {}
        
        if terminated:
            info['episode'] = {'r': self.episode_reward, 'l': self.steps}
        
        return observation, reward, terminated, truncated, info
    
    def _process_observation(self, observation):
        """Convert dm_control observation dict to flat array."""
        obs_list = []
        for name, obs in observation.items():
            if name == 'target_position':
                # Only use the first 3 values
                obs_list.append(obs.flatten()[:3])
            else:
                obs_list.append(obs.flatten())
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
                if step % 10 == 0 or step == 0:
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

def train_sac(total_timesteps=50000):
    """Train a SAC agent on the arm environment."""
    # Create raw environment for checking
    raw_env = DMControlGymnasiumWrapper()
    check_env(raw_env)  # Check the environment BEFORE vectorizing
    
    # Create environments
    env = DummyVecEnv([lambda: Monitor(DMControlGymnasiumWrapper())])
    eval_env = DMControlGymnasiumWrapper()
    
    # Create the SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50000,
        batch_size=128,
        tau=0.01,
        gamma=0.99,
        train_freq=8,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./sac_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],
                qf=[64, 64]
            ),
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Create callbacks
    video_callback = VideoRecorderCallback(eval_env, eval_freq=5000)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./sac_checkpoints/",
        name_prefix="sac_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [video_callback, checkpoint_callback]
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save("final_sac_model")
    
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
    
    # Record final performance
    obs, _ = eval_env.reset()
    frames = []
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        if step % 5 == 0:
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
        import os
        print("Installing required packages...")
        os.system("pip install gymnasium stable-baselines3[extra]")
        print("Packages installed. Please run the script again.")
        exit()
    
    # Test the environment
    env = DMControlGymnasiumWrapper()
    print(f"Observation space: {env.observation_space.shape}")  # Should be (15,)
    print(f"Action space: {env.action_space.shape}")  # Should be (4,)
    
    # Train the agent
    model = train_sac(total_timesteps=50000)
    print(f"Training completed. Time limit: {_DEFAULT_TIME_LIMIT} seconds")