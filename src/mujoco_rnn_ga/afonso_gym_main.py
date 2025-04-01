#!/usr/bin/env python3
# filepath: /home/animat/src/mujoco_rnn_ga/gymnasium_arm_training.py

"""
Train an arm to reach specific targets using reinforcement learning with Stable Baselines 3.
Updated to use Gymnasium (the maintained successor to Gym) for better compatibility and features.
"""

import os
import time
import pickle
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from typing import Callable, Dict, Any, Tuple

# Import custom modules
from plants import SequentialReacher
from environments import SequentialReachingEnv
from utils import *

# CUDA optimizations for WSL
os.environ["MUJOCO_GL"] = "osmesa"  # Use software rendering for MuJoCo
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Check CUDA availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Optimize CUDA for WSL
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    device = "cuda"
else:
    print("CUDA not available, using CPU")
    device = "cpu"

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR = OUTPUT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train an arm to reach targets using RL')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac'], 
                        help='RL algorithm to use')
    parser.add_argument('--total_timesteps', type=int, default=2_000_000, 
                        help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for updates')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per update (PPO only)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Policy network hidden size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_targets', type=int, default=5, help='Targets per evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_envs', type=int, default=8, 
                        help='Number of parallel environments')
    parser.add_argument('--eval_freq', type=int, default=50000, 
                        help='Evaluate every n steps')
    parser.add_argument('--resume', help='Resume from saved model')
    return parser.parse_args()


class GymnasiumArmEnv(gym.Env):
    """
    Gymnasium wrapper for the arm environment that is compatible with Stable Baselines 3.
    Gymnasium is the maintained successor to Gym.
    """
    def __init__(self, env_maker):
        super(GymnasiumArmEnv, self).__init__()
        
        # Create the environment
        self.env = env_maker()
        
        # Define action and observation spaces required by gymnasium
        num_actuators = self.env.plant.num_actuators
        num_sensors = self.env.plant.num_sensors
        
        # 3D target position + sensor readings
        obs_dim = num_sensors + 3
        
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(num_actuators,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def step(self, action):
        """
        Take a step in the environment
        In Gymnasium, step returns (obs, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)
        
        # Convert to numpy if not already
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        # Split done into terminated and truncated for Gymnasium compatibility
        terminated = done and not info.get('timeout', False)  # Reached goal
        truncated = done and info.get('timeout', False)       # Reached time limit
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment
        In Gymnasium, reset returns (obs, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        obs = self.env.reset()
        
        # Convert to numpy if not already
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        return obs, {}  # Return observation and empty info dict

    def render(self):
        """Render the environment"""
        return self.env.render()

    def close(self):
        """Close the environment"""
        return self.env.close()


class EnvMaker:
    """Environment maker class that can be properly pickled"""
    def __init__(self, seed, num_targets, loss_weights):
        self.seed = seed
        self.num_targets = num_targets
        self.loss_weights = loss_weights
        
    def __call__(self):
        # Create a new plant instance
        plant = SequentialReacher()
        
        # Create a new environment with that plant
        base_env = SequentialReachingEnv(
            plant=plant,
            target_duration=50,
            num_targets=self.num_targets,
            loss_weights=self.loss_weights
        )
        
        # Define RLEnvWrapper inline to avoid import issues
        class RLEnvWrapper:
            """RL interface wrapper for the environment"""
            def __init__(self, env):
                self.env = env
                self.plant = env.plant
                self.current_target = None
                self.prev_distance = None
                self.episode_step = 0
                
            def reset(self):
                """Reset environment and return initial observation"""
                # Reset the underlying plant
                self.plant.reset()
                
                # Sample a new target position and update it in the plant
                if hasattr(self.env, 'sample_target'):
                    self.current_target = self.env.sample_target()
                else:  # Fallback to plant's target sampling
                    targets = self.plant.sample_targets(1)
                    self.current_target = targets[0]
                
                # Update the target in the plant
                self.plant.update_target(self.current_target)
                
                # Initialize episode variables
                self.episode_step = 0
                self.prev_distance = np.linalg.norm(
                    self.plant.get_hand_pos() - self.current_target
                )
                
                # Return initial observation
                return self._get_observation()
            
            def step(self, action):
                """Take action and return next_obs, reward, done, info"""
                # Apply action to the plant
                self.plant.step(action)
                
                # Calculate reward components
                hand_pos = self.plant.get_hand_pos()
                current_distance = np.linalg.norm(hand_pos - self.current_target)
                
                # Position component (negative distance)
                position_reward = -current_distance * self.env.loss_weights['position']
                
                # Progress component (getting closer)
                progress_reward = (self.prev_distance - current_distance) * self.env.loss_weights['progress']
                self.prev_distance = current_distance
                
                # Energy efficiency component
                energy_penalty = -np.sum(np.square(action)) * self.env.loss_weights['energy']
                
                # Time penalty (encourage faster completion)
                time_penalty = -0.01 * self.env.loss_weights['time']
                
                # Total reward
                reward = position_reward + progress_reward + energy_penalty + time_penalty
                
                # Check if done (reached target or max steps)
                target_reached = current_distance < 0.05  # 5cm threshold
                self.episode_step += 1
                timeout = self.episode_step >= 1000  # Max episode length
                done = target_reached or timeout
                
                # Extra info
                info = {
                    'distance': current_distance,
                    'target_reached': target_reached,
                    'timeout': timeout
                }
                
                return self._get_observation(), reward, done, info
            
            def render(self):
                """Render the environment"""
                return self.plant.render()
            
            def _get_observation(self):
                """Create observation vector without hand position"""
                # Get plant observations
                plant_obs = self.plant.get_obs()
                
                # Handle tuple observation structure
                if isinstance(plant_obs, tuple) and len(plant_obs) == 2:
                    # Extract only the second element (sensor data) and ignore hand position
                    sensor_data = plant_obs[1]
                    # Ensure it's a flattened numpy array
                    sensor_data = np.array(sensor_data, dtype=np.float32).flatten()
                else:
                    # If not a tuple with expected structure, use original handling
                    sensor_data = np.array(plant_obs, dtype=np.float32).flatten()
                
                # Ensure target is a flattened numpy array
                target_pos = np.array(self.current_target, dtype=np.float32).flatten()
                
                # Combine sensor data with target position (excluding hand position)
                return np.concatenate([sensor_data, target_pos])
            
            def close(self):
                """Close the environment"""
                if hasattr(self.plant, 'close'):
                    self.plant.close()
        
        # Wrap with RL interface
        env = RLEnvWrapper(base_env)
        
        # Set seed
        np.random.seed(self.seed)
        
        return env


class TqdmCallback(BaseCallback):
    """
    Custom callback for progress tracking with tqdm.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TqdmCallback, self).__init__(verbose)
        self.progress_bar = None
        self.total_timesteps = total_timesteps
        self.best_mean_reward = -np.inf
        self.last_log_time = time.time()
        
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress")
        
    def _on_step(self):
        # Update the progress bar
        self.progress_bar.update(self.num_timesteps - self.progress_bar.n)
        
        # Log metrics every 10 seconds
        if time.time() - self.last_log_time > 10:
            self.last_log_time = time.time()
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                self.progress_bar.set_postfix({
                    'reward': f'{mean_reward:.2f}',
                    'length': f'{mean_length:.1f}'
                })
                
                # Keep track of best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(MODELS_DIR, "best_model"))
                
        return True
        
    def _on_training_end(self):
        if self.progress_bar is not None:
            self.progress_bar.close()


def make_env(env_maker, rank, seed=0):
    """
    Create a function that will create and wrap an environment for parallel execution.
    """
    def _init():
        env = GymnasiumArmEnv(env_maker)
        # Set seed for this environment
        env.reset(seed=seed + rank)
        return env
    return _init


def create_policy_kwargs(hidden_size, algo):
    """Create policy network architecture parameters based on algorithm"""
    if algo.lower() == 'ppo':
        policy_kwargs = {
            "net_arch": {
                "pi": [hidden_size, hidden_size],  # Actor network
                "vf": [hidden_size, hidden_size]   # Value network for PPO
            }
        }
    else:  # SAC
        policy_kwargs = {
            "net_arch": {
                "pi": [hidden_size, hidden_size, hidden_size//2],  # Deeper actor
                "qf": [hidden_size*2, hidden_size, hidden_size]    # Wider critic
            },
            "activation_fn": torch.nn.ReLU
        }
    return policy_kwargs


def plot_training_results(model_prefix):
    """Plot training results from Stable Baselines 3 monitor files"""
    try:
        from stable_baselines3.common.monitor import load_results
        from stable_baselines3.common.results_plotter import plot_results
        
        plt.figure(figsize=(12, 6))
        
        # Plot the reward
        ax1 = plt.subplot(1, 2, 1)
        plot_results([LOGS_DIR], 1_000_000, "timesteps", "Reward over time", ax=ax1)
        ax1.set_title("Learning Curve")
        
        # Plot episode lengths
        data = load_results(LOGS_DIR)
        if len(data) > 0:
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title('Episode Length over Time')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Episode Length')
            ax2.plot(data.l.rolling(window=10).mean())
        
        plt.tight_layout()
        plt.savefig(LOGS_DIR / f"training_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.show()
            
    except Exception as e:
        print(f"Error plotting results: {e}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_random_seed(args.seed)
    
    # Configure logging
    logger = configure(str(LOGS_DIR), ["stdout", "csv", "tensorboard"])
    
    # Additional GPU optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
    
    # Define the reward weights for each component
    loss_weights = {
        'position': 1.0,    # Weight for position error
        'time': 0.1,        # Weight for time-to-target  
        'energy': 0.05,     # Weight for energy efficiency
        'progress': 0.5     # Weight for progress toward target
    }
    
    # Create environments
    print(f"Creating {args.num_envs} parallel environments...")
    env_makers = [
        EnvMaker(seed=args.seed + i, num_targets=args.num_targets, loss_weights=loss_weights)
        for i in range(args.num_envs)
    ]
    
    # Create vectorized environments
    vec_env = SubprocVecEnv([make_env(env_makers[i], i, args.seed) for i in range(args.num_envs)])
    
    # Use environment normalization for better performance
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create evaluation environment with same normalization
    eval_env = GymnasiumArmEnv(env_makers[0])
    # Wrap in a DummyVecEnv
    eval_env = DummyVecEnv([lambda: eval_env])
    # Apply normalization with same stats as training
    eval_env = VecNormalize(
        eval_env,
        training=False,  # Don't update running stats during eval
        norm_obs=vec_env.norm_obs,
        norm_reward=vec_env.norm_reward,
        clip_obs=vec_env.clip_obs,
        clip_reward=vec_env.clip_reward,
        gamma=vec_env.gamma,
        epsilon=vec_env.epsilon,
    )
    # Copy over the normalization stats
    eval_env.obs_rms = vec_env.obs_rms
    eval_env.ret_rms = vec_env.ret_rms
    
    # Get observation and action dimensions
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    print(f"Using {args.num_envs} parallel environments")
    print(f"Observation space: {obs_dim} dimensions")
    print(f"Action space: {act_dim} dimensions")
    
    # Create policy network architecture
    policy_kwargs = create_policy_kwargs(args.hidden_size, args.algo)
    
    # Create algorithms
    if args.algo.lower() == 'ppo':
        model = PPO(
            "MlpPolicy", 
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=0.01,  # For exploration
            clip_range=0.2,
            verbose=0,
            device=device
        )
        print("Using PPO algorithm")
    else:  # SAC
        model = SAC(
            "MlpPolicy", 
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            buffer_size=1_000_000,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=0,
            device=device
        )
        print("Using SAC algorithm")
    
    # If resuming from a saved model
    if args.resume:
        print(f"Loading model from {args.resume}")
        if args.algo.lower() == 'ppo':
            model = PPO.load(args.resume, env=vec_env)
        else:
            model = SAC.load(args.resume, env=vec_env)
    
    # Custom evaluation callback that saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR),  # Convert to string
        log_path=str(LOGS_DIR),                # Convert to string
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )
    
    # Progress tracking callback
    tqdm_callback = TqdmCallback(args.total_timesteps)
    
    # Start training
    print(f"Starting training for {args.total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, tqdm_callback]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = str(MODELS_DIR / "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save normalized environment stats
    vec_env_path = str(MODELS_DIR / "vec_normalize.pkl")
    vec_env.save(vec_env_path)
    print(f"Environment normalization saved to {vec_env_path}")
    
    # For final evaluation, we need to get a fresh environment but with the right normalization
    final_eval_env = GymnasiumArmEnv(env_makers[0])
    final_eval_env = DummyVecEnv([lambda: final_eval_env])
    final_eval_env = VecNormalize.load(vec_env_path, final_eval_env)
    final_eval_env.training = False  # Don't update stats during eval

    # Now evaluate
    mean_reward, std_reward = evaluate_policy(
        model, final_eval_env, n_eval_episodes=10, deterministic=True, render=True
    )
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot results
    plot_training_results("model")
    
    # Clean up
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()