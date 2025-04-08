import os
import time
import numpy as np
import torch
import gymnasium as gym
import jax
import jax.numpy as jnp
import inspect

# SB3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.logger import configure

# Import our custom adapter
from mjx_gym_adapter import MJXVecGymEnv

# Add these imports at the top of your file
from torch.utils.tensorboard import SummaryWriter

def safe_mean(arr):
    """Compute the mean of an array that might be empty"""
    return np.nan if len(arr) == 0 else np.mean(arr)


class SACAgent:
    """SAC agent using Stable-Baselines3 with MJX environment."""
    
    def __init__(
        self, 
        env, 
        learning_rate=3e-4, 
        num_envs=128, 
        gamma=0.99, 
        normalize_observations=False, 
        buffer_size=2000000,
        batch_size=1024,
        train_freq=1,
        gradient_steps=8,
        tau=0.005, 
        auto_entropy=True,
        entropy_coef=0.01,
        log_dir="./logs",
        device='auto'
    ):
        # Store configuration
        self.env = env
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.normalize_observations = normalize_observations
        self.log_dir = log_dir
        
        # Device selection
        self.device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else device
        print(f"Using device: {self.device}")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create Gymnasium-compatible environment
        self.gym_env = MJXVecGymEnv(env)  # Always use vectorized environment
        
        # Apply normalization if requested
        self.gym_env = VecNormalize(
            self.gym_env,
            norm_obs=normalize_observations,  # Use parameter directly
            norm_reward=True,                 # Always normalize rewards
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma
        )
        
        # Debug info on normalization
        print(f"Observation normalization in environment: True")
        print(f"Observation normalization in VecNormalize: {normalize_observations}")
        
        # Set up network architecture for SAC
        sac_policy_kwargs = {
            "net_arch": dict(
                pi=[256, 256, 256, 128],
                qf=[256, 256, 256, 128]
            )
        }
        
        # Initialize SAC
        self.model = SAC(
            "MlpPolicy",
            self.gym_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            tau=tau,
            gamma=gamma,
            ent_coef="auto" if auto_entropy else entropy_coef,
            target_update_interval=1,
            policy_kwargs=sac_policy_kwargs,
            tensorboard_log=log_dir,
            verbose=0,
            device=self.device
        )
        print(f"Initialized SAC model on {self.device}")
        
        # Save all hyperparameters to a file for reproducibility
        import json
        hyperparams = {
            'learning_rate': learning_rate,
            'num_envs': num_envs,
            'gamma': gamma,
            'normalize_observations': normalize_observations,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'tau': tau,
            'auto_entropy': auto_entropy,
            'entropy_coef': entropy_coef,
        }
        with open(os.path.join(log_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f)
    
    def train(self, num_timesteps):
        """Train the SAC model with TensorBoard logging."""
        # Configure logger with TensorBoard enabled
        self.model.tensorboard_log = self.log_dir
        
        # Force TensorBoard integration
        from stable_baselines3.common.logger import configure
        tb_formatter = configure(self.log_dir, ["tensorboard"])
        self.model.set_logger(tb_formatter)
        
        # Then replace CurriculumCallback with this version:
        class CurriculumCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=1)
                self.start_time = None
                self.last_log_time = time.time()
                self.rollout_buffer = []
                # Direct TB writer
                self.writer = None
                
            def _on_training_start(self):
                self.start_time = time.time()
                # Create our own SummaryWriter for direct TensorBoard access
                self.writer = SummaryWriter(self.logger.dir)
                print(f"Direct TensorBoard writer created at {self.logger.dir}")
                
                # Share this writer with environment for direct logging
                if hasattr(self.training_env, 'venv') and hasattr(self.training_env.venv, 'env'):
                    # Connect to the environment
                    self.training_env.venv.env.tb_writer = self.writer
                    print("Connected writer to environment")
                    
                    # Connect to the adapter
                    self.training_env.venv.writer = self.writer
                    print("Connected writer to adapter")
                
                # Force more frequent logging for episode rewards
                if hasattr(self.model, "replay_buffer"):
                    # Log the raw training rewards (before normalization)
                    def log_rewards(rewards):
                        if self.writer is not None:
                            batch_size = len(rewards)
                            if batch_size > 0:
                                mean_reward = float(np.mean(rewards))
                                self.writer.add_scalar("train/raw_batch_reward", mean_reward, self.num_timesteps)
                                self.writer.flush()
                        return rewards
                        
                    # Monkey patch SB3's replay buffer to capture rewards
                    original_add = self.model.replay_buffer.add
                    def add_with_logging(*args, **kwargs):
                        rewards = args[2]  # Rewards are 3rd argument
                        log_rewards(rewards)
                        return original_add(*args, **kwargs)
                        
                    self.model.replay_buffer.add = add_with_logging
                    print("Enhanced replay buffer with reward logging")
                
            def _on_rollout_end(self):
                """Log rollout data when each collection period ends."""
                # Process episodic data from the model's buffer
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                    
                    # Direct TensorBoard writing - bypassing SB3 logger
                    if self.writer is not None:
                        self.writer.add_scalar("rollout/ep_reward_mean", np.mean(rewards), self.num_timesteps)
                        self.writer.add_scalar("rollout/ep_reward_median", np.median(rewards), self.num_timesteps)
                        self.writer.add_scalar("rollout/ep_len_mean", np.mean(lengths), self.num_timesteps)
                        self.writer.add_scalar("rollout/ep_reward_min", np.min(rewards), self.num_timesteps)
                        self.writer.add_scalar("rollout/ep_reward_max", np.max(rewards), self.num_timesteps)
                        self.writer.flush()  # Force write to disk
                        
                    # Also log via SB3 for compatibility
                    self.logger.record("rollout/ep_reward_mean", np.mean(rewards))
                    self.logger.record("rollout/ep_reward_median", np.median(rewards))
                    self.logger.record("rollout/ep_len_mean", np.mean(lengths))
                    self.logger.record("rollout/ep_reward_min", np.min(rewards))
                    self.logger.record("rollout/ep_reward_max", np.max(rewards))
                    
                    # Force log write
                    self.logger.dump(self.num_timesteps)
                    
                    # Print status update
                    print(f"Step {self.num_timesteps}: Mean reward: {np.mean(rewards):.2f} over {len(rewards)} episodes")
                
                # Direct tracking of episode completions
                if hasattr(self.training_env, 'venv') and hasattr(self.training_env.venv, 'recent_dones'):
                    venv = self.training_env.venv
                    # Track all completed episodes
                    for i in range(venv.num_envs):
                        if venv.recent_dones[i] and venv.recent_episode_returns[i] > 0:
                            # Add to our local buffer
                            ep_reward = float(venv.recent_episode_returns[i])
                            ep_length = int(venv.episode_lengths[i])
                            
                            # Also directly log to TB - CRITICAL ADDITION
                            if self.writer is not None:
                                self.writer.add_scalar("debug/episode_reward", ep_reward, self.num_timesteps)
                                self.writer.add_scalar("debug/episode_length", ep_length, self.num_timesteps)
            
            def _on_step(self):
                """Track metrics each step with more aggressive episode detection"""
                # Check for episode completions EVERY step
                if hasattr(self.training_env, 'venv') and hasattr(self.training_env.venv, 'recent_dones'):
                    venv = self.training_env.venv
                    
                    # Check each environment for completion (CRITICAL ADDITION)
                    for i in range(venv.num_envs):
                        if venv.recent_dones[i] and venv.recent_episode_returns[i] != 0:
                            # Directly log completed episodes - NO BUFFERING
                            ep_reward = float(venv.recent_episode_returns[i])
                            ep_length = int(venv.episode_lengths[i])
                            
                            # Log to BOTH debug and rollout sections
                            if self.writer is not None:
                                self.writer.add_scalar("debug/episode_reward", ep_reward, self.num_timesteps)
                                self.writer.add_scalar("rollout/ep_reward", ep_reward, self.num_timesteps)
                                self.writer.flush()  # Force write immediately
                                
                                # Print occasional rewards
                                if i == 0 or ep_reward > 5.0:  # Only print env 0 or high rewards
                                    print(f"Episode completed: reward={ep_reward:.2f}, length={ep_length}")
                
                # Update curriculum metrics
                env = None
                if hasattr(self.training_env, 'venv') and hasattr(self.training_env.venv, 'env'):
                    env = self.training_env.venv.env
                    
                    # Log curriculum level 
                    if hasattr(env, 'curriculum_level'):
                        if self.writer is not None:
                            self.writer.add_scalar("curriculum/level", env.curriculum_level, self.num_timesteps)
                            
                    # Very frequent logging for better visualization
                    if self.n_calls % 50 == 0:  # Even more frequent updates
                        fps = int(self.n_calls / (time.time() - self.start_time)) if self.start_time else 0
                        
                        if self.writer is not None:
                            self.writer.add_scalar("time/fps", fps, self.num_timesteps)
                            
                            # Get episode rewards from buffer too (backup method)
                            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                                rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                                self.writer.add_scalar("rollout/ep_reward_mean", np.mean(rewards), self.num_timesteps)
                            
                            self.writer.flush()  # Force write
                        
                    # Process the info buffer to extract success rate
                    if hasattr(self.training_env.venv, 'info_buffer') and len(self.training_env.venv.info_buffer) > 0:
                        info = self.training_env.venv.info_buffer[0]
                        if isinstance(info, dict) and "target_reached" in info:
                            success_rate = float(np.mean(info["target_reached"]))
                            
                            if self.writer is not None:
                                self.writer.add_scalar("performance/target_success_rate", success_rate, self.num_timesteps)
                                self.writer.add_scalar("rollout/target_success_rate", success_rate, self.num_timesteps)
                
                return True
                
            def _on_training_end(self):
                # Close writer to flush remaining data
                if self.writer is not None:
                    self.writer.close()

        # Train with minimal callbacks
        start_time = time.time()
        self.model.learn(
            total_timesteps=num_timesteps,
            callback=CurriculumCallback(),
            progress_bar=True  # Built-in progress bar
        )
        
        # Save trained model
        self.save(os.path.join(self.log_dir, "final_model_sac"))
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    def save(self, path):
        """Save model and normalization stats."""
        self.model.save(path)
        if hasattr(self.gym_env, "save"):
            self.gym_env.save(f"{path}_vecnorm.pkl")
        print(f"Model saved to {path}")

    def load(self, path, vecnorm_path=None):
        """Load model and normalization stats."""
        self.model = SAC.load(path, env=self.gym_env, device=self.device)
        
        if vecnorm_path is not None and hasattr(self.gym_env, "load"):
            self.gym_env.load(vecnorm_path)
        print(f"Model loaded from {path}")