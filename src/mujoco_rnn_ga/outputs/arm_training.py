import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import argparse
from tqdm import tqdm  # Import tqdm for progress bar

from dm_testing.arm_env import load
from dm_testing.dm_control_test import display_video

# Set CUDA device and optimize for WSL
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["MKL_NUM_THREADS"] = "1"  # Prevents MKL oversubscription
os.environ["OMP_NUM_THREADS"] = "1"  # Prevents OpenMP oversubscription

# Force CPU for MuJoCo
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# Limit PyTorch GPU memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class DMControlGymWrapper(gym.Env):
    """Convert dm_control environment to gym interface with precise observation shape control."""
    def __init__(self, env):
        self.env = env
        
        # Get observation shapes from environment
        time_step = env.reset()
        self._obs_keys = list(time_step.observation.keys())
        self._obs_shapes = {k: v.shape for k, v in time_step.observation.items()}
        
        # Filter out unwanted observables (e.g., hand_position)
        self._filtered_obs_keys = [k for k in self._obs_keys if k != "hand_position"]
        
        # Calculate total input size
        total_input_size = sum(np.prod(self._obs_shapes[k]) for k in self._filtered_obs_keys)
        print(f"Filtered observation keys: {self._filtered_obs_keys}")
        print(f"Total observation dimensions: {total_input_size}")
        assert total_input_size == 15, f"Expected 15 inputs, got {total_input_size}"
        
        # Create observation space
        self.observation_space = spaces.Box(
            low=np.full(total_input_size, -10.0, dtype=np.float32), 
            high=np.full(total_input_size, 10.0, dtype=np.float32), 
            shape=(total_input_size,), 
            dtype=np.float32
        )
        
        # Set action space based on dm_control spec
        action_spec = env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )
        
    def _flatten_obs(self, obs_dict):
        """Convert observation dict to flat array, excluding unwanted keys."""
        return np.concatenate([obs_dict[k].flatten() for k in self._filtered_obs_keys])
    
    def reset(self, **kwargs):
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}
    
    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = time_step.last()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Completely disable rendering during training."""
        return np.zeros((1, 1, 3), dtype=np.uint8)  # Return minimal dummy frame
        
    def close(self):
        pass

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
            self.env._random_state.seed(seed)

class CustomMlpFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor optimized for muscle control.
    Separates muscle sensor data from target position.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # Define network architecture
        self.muscle_net = torch.nn.Sequential(
            torch.nn.Linear(12, 64),  # 12 muscle sensors (4 muscles x 3 types)
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        
        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(3, 16),  # 3D target position
            torch.nn.ReLU(),
        )
        
        self.combined_net = torch.nn.Sequential(
            torch.nn.Linear(64 + 16, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations into muscle data and target position
        muscle_data = observations[:, :12]  # First 12 dimensions for muscle sensors
        target_pos = observations[:, 12:15]  # Last 3 dimensions for target position
        
        muscle_features = self.muscle_net(muscle_data)
        target_features = self.target_net(target_pos)
        
        # Combine features
        combined = torch.cat([muscle_features, target_features], dim=1)
        return self.combined_net(combined)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional metrics in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_times = []
        self.last_time = None

    def _on_training_start(self):
        self.last_time = time.time()

    def _on_step(self) -> bool:
        # Calculate steps per second
        current_time = time.time()
        step_time = current_time - self.last_time
        self.step_times.append(step_time)
        self.last_time = current_time

        if len(self.step_times) >= 100:
            mean_step_time = np.mean(self.step_times[-100:])
            fps = 1.0 / mean_step_time if mean_step_time > 0 else 0
            self.logger.record("metrics/fps", fps)

            # Log hardware utilization if available
            if torch.cuda.is_available():
                # Safely try to get GPU stats
                try:
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    self.logger.record("metrics/gpu_mem", gpu_mem)
                except:
                    pass

            # Safely log policy statistics using the replay buffer
            if hasattr(self.model, "replay_buffer"):
                try:
                    # For SAC, get actions from the replay buffer
                    if hasattr(self.model.replay_buffer, "actions") and self.model.replay_buffer.pos > 0:
                        # Get recent actions (up to buffer position)
                        buffer_pos = min(100, self.model.replay_buffer.pos)
                        if buffer_pos > 0:
                            recent_actions = self.model.replay_buffer.actions[-buffer_pos:].flatten(1)
                            self.logger.record("policy/action_mean", np.mean(recent_actions))
                            self.logger.record("policy/action_std", np.std(recent_actions))
                except Exception as e:
                    # Silently fail if we can't get action stats
                    if self.verbose > 1:
                        print(f"Error logging action stats: {e}")

        return True

def make_env(env_id: int, seed: int = 0, frameskip: int = 2, is_training: bool = True) -> callable:
    """
    Utility function for multiprocessed env.
    """
    def _init() -> gym.Env:
        dm_env = load(normalize_actions=True, frameskip=frameskip)
        if hasattr(dm_env, 'task'):
            dm_env.task._is_training = is_training
        env = DMControlGymWrapper(dm_env)
        env.seed(seed + env_id)
        env = Monitor(env)
        return env
    return _init

def make_vec_env(num_envs: int = 4, frameskip: int = 2, normalize_obs: bool = True) -> gym.Env:
    envs = [make_env(i, seed=i, frameskip=frameskip) for i in range(num_envs)]
    vec_env = SubprocVecEnv(envs)  # Use subprocesses for parallelism
    if normalize_obs:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    return vec_env

def get_model_policy_kwargs():
    """Get policy kwargs for network architecture."""
    return {
        "net_arch": {
            "pi": [128, 128],  # Policy network
            "qf": [256, 256]   # Q-function network
        },
        "features_extractor_class": CustomMlpFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "activation_fn": torch.nn.ReLU,
        "optimizer_kwargs": {"weight_decay": 1e-5},  # L2 regularization
    }

def train(
    algo: str = "sac", 
    num_envs: int = 1, 
    frameskip: int = 2, 
    timesteps: int = 500_000, 
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    resume: bool = False,
    model_path: str = None
):
    """
    Train an agent using specified algorithm.
    """
    # Create log directories
    log_dir = os.path.join("logs", algo)
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join("checkpoints", algo)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_random_seed(42)
    
    # Create vectorized environment
    env = make_vec_env(num_envs, frameskip)
    
    # Common parameters for all algorithms
    common_params = {
        "verbose": 0,  # Set verbose to 0 to avoid cluttering the output
        "tensorboard_log": os.path.join("tensorboard", algo),
        "learning_rate": learning_rate,
    }
    
    # Create policy with custom network architecture
    policy_kwargs = get_model_policy_kwargs()
    
    # If CUDA is available, use it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create the appropriate algorithm
    if resume and model_path:
        print(f"Resuming training from {model_path}")
        if algo == "sac":
            model = SAC.load(model_path, env=env)
        elif algo == "ppo":
            model = PPO.load(model_path, env=env)
        elif algo == "td3":
            model = TD3.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
    else:
        if algo == "sac":
            model = SAC(
                "MlpPolicy", 
                env, 
                buffer_size=100000,
                batch_size=512,
                use_sde=True,  # Enable State-Dependent Exploration (optional)
                device="cuda",  # Ensure GPU is used
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                target_update_interval=1,
                policy_kwargs=policy_kwargs,
                train_freq=1,
                gradient_steps=1,
                learning_starts=1000,
                **common_params
            )
        elif algo == "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=2048,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                normalize_advantage=True,
                target_kl=0.01,
                policy_kwargs=policy_kwargs,
                device=device,
                **common_params
            )
        elif algo == "td3":
            model = TD3(
                "MlpPolicy",
                env,
                buffer_size=100000,
                batch_size=batch_size,
                gamma=0.99,
                tau=0.005,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                policy_kwargs=policy_kwargs,
                device=device,
                **common_params
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1),
        save_path=checkpoint_dir,
        name_prefix=f"{algo}_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Create a separate environment for evaluation
    eval_env = make_vec_env(1, frameskip)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoint_dir, "best_model"),
        log_path=log_dir,
        eval_freq=max(5000 // num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Custom callback for additional metrics
    tensorboard_callback = TensorboardCallback()
    
    # Combine callbacks
    all_callbacks = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])
    
    # Train the model with built-in progress bar
    print(f"Starting training with {algo}, {num_envs} environments, {timesteps} timesteps...")
    start_time = time.time()
    
    env._is_training = True
    model.learn(total_timesteps=timesteps, callback=all_callbacks, progress_bar=True)
    env._is_training = False
    
    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, f"{algo}_final")
    model.save(final_model_path)
    
    # Also save normalization statistics
    if hasattr(env, "obs_rms"):
        env_stats_path = os.path.join(checkpoint_dir, f"{algo}_vecnormalize.pkl")
        env.save(env_stats_path)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return final_model_path

def record_video(model_path, output_path="video.gif", vec_normalize_path=None, episodes=1):
    """Record a video of the trained agent."""
    # Load the model
    if model_path.endswith(".zip"):
        model_name = os.path.basename(model_path).split("_")[0]
        if model_name == "sac":
            model = SAC.load(model_path)
        elif model_name == "ppo":
            model = PPO.load(model_path)
        elif model_name == "td3":
            model = TD3.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    else:
        raise ValueError(f"Invalid model path: {model_path}")
    
    # Create environment
    env = make_vec_env(1, frameskip=2, normalize_obs=True)
    
    # Load normalization stats if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Record episodes
    for episode in range(episodes):
        obs = env.reset()
        done = False
        frames = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            frame = env.render()
            frames.append(frame)
        
        # Save video
        display_video(frames, output_path)

def plot_learning_curves(algo_name="sac"):
    """Plot training progress from monitor logs."""
    log_dir = os.path.join("logs", algo_name)
    if not os.path.exists(log_dir):
        print(f"No logs found for {algo_name}")
        return
        
    # Find all monitor files
    monitor_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.monitor.csv'):
                monitor_files.append(os.path.join(root, file))
                
    if not monitor_files:
        print(f"No monitor files found in {log_dir}")
        return
        
    # Process the data
    data = []
    for monitor_file in monitor_files:
        with open(monitor_file, 'r') as f:
            # Skip header lines
            f.readline()  # Header
            f.readline()  # Empty line
            
            # Parse data
            timestamps = []
            episode_lengths = []
            episode_rewards = []
            
            for line in f:
                parts = line.split(',')
                if len(parts) >= 4:
                    r = float(parts[0])
                    l = int(parts[1])
                    t = float(parts[2])
                    
                    timestamps.append(t)
                    episode_lengths.append(l)
                    episode_rewards.append(r)
                    
            data.append({
                'file': os.path.basename(monitor_file),
                'timestamps': timestamps,
                'lengths': episode_lengths,
                'rewards': episode_rewards
            })
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot rewards
    for d in data:
        rewards = pd.Series(d['rewards']).rolling(window=10).mean()
        axes[0].plot(range(len(rewards)), rewards, label=d['file'])
        
    axes[0].set_title(f'Episode Rewards ({algo_name})')
    axes[0].set_ylabel('Reward (10-episode moving average)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # Plot episode lengths
    for d in data:
        lengths = pd.Series(d['lengths']).rolling(window=10).mean()
        axes[1].plot(range(len(lengths)), lengths)
        
    axes[1].set_title('Episode Lengths')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Length (timesteps)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join("logs", f"{algo_name}_learning_curves.png"))
    plt.show()
    
    # Compute statistics for final performance
    final_stats = {}
    for d in data:
        # Last 10% of episodes for final performance
        last_n = max(10, len(d['rewards']) // 10)
        final_rewards = d['rewards'][-last_n:]
        
        final_stats[d['file']] = {
            'mean_reward': np.mean(final_rewards),
            'std_reward': np.std(final_rewards),
            'min_reward': np.min(final_rewards),
            'max_reward': np.max(final_rewards),
            'mean_length': np.mean(d['lengths'][-last_n:])
        }
    
    # Print statistics
    print("\nFinal Performance Statistics:")
    for file, stats in final_stats.items():
        print(f"\n{file}:")
        print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"  Mean episode length: {stats['mean_length']:.1f}")

def evaluate_model(model_path, num_episodes=10, render=False, vec_normalize_path=None):
    """Evaluate trained model's performance."""
    # Extract algorithm type from model path
    if model_path.endswith(".zip"):
        model_name = os.path.basename(model_path).split("_")[0]
        if model_name == "sac":
            model = SAC.load(model_path)
        elif model_name == "ppo":
            model = PPO.load(model_path)
        elif model_name == "td3":
            model = TD3.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    else:
        raise ValueError(f"Invalid model path: {model_path}")
    
    # Create environment (single, non-vectorized for evaluation)
    env = load(normalize_actions=True, frameskip=2)
    env = DMControlGymWrapper(env)
    
    # Create a monitor env for logging
    log_dir = os.path.join("logs", "evaluation")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    # Wrap in DummyVecEnv for compatibility with trained model
    env = DummyVecEnv([lambda: env])
    
    # Load normalization stats if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Evaluation loop
    rewards = []
    episode_lengths = []
    success_rate = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if render and steps % 5 == 0:  # Render every 5 steps to avoid slowdown
                env.render()
            
            # Consider episode successful if goal reached
            if done[0] and 'is_success' in info[0] and info[0]['is_success']:
                success_rate += 1
        
        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Print evaluation statistics
    success_rate = success_rate / num_episodes
    print("\nEvaluation Results:")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    print(f"Success rate: {success_rate*100:.1f}%")
    
    return rewards, episode_lengths, success_rate

# Add linux system optimization wrapper
def optimize_system_for_training():
    """Apply system optimizations for RL training on Linux/WSL."""
    try:
        # Try to set process priority
        import os
        import psutil
        process = psutil.Process(os.getpid())
        if hasattr(process, "nice"):
            try:
                process.nice(-10)  # Higher priority
            except:
                print("Could not set process priority, may need sudo")
        
        # NUMA binding if available (for multi-socket systems)
        if torch.cuda.is_available():
            gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            try:
                import subprocess
                subprocess.call(f"numactl --cpunodebind=0 --membind=0", shell=True)
                print(f"NUMA binding applied for optimal memory access")
            except:
                pass  # Not critical if it fails
        
        # Set PyTorch settings for performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("CUDA optimizations applied")
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} with {gpu_mem:.1f} GB memory")
        
        # Set environment variables for MKL/OMP/LAPACK
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1" 
        
        # Force MuJoCo to use CPU
        os.environ["MUJOCO_GL"] = "osmesa"  # Use CPU-based Mesa OpenGL
        
        # For CUDA operations, limit GPU memory usage
        if torch.cuda.is_available():
            # Empty cache periodically
            torch.cuda.empty_cache()
            
            # Limit GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(0.5)  # Use only 50% of GPU memory
            
            # Pin memory for faster CPU->GPU transfers
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",garbage_collection_threshold:0.8"
            
            # Move more model components to CPU
            torch.set_num_threads(16)  # Use more CPU threads
            
            print("Advanced GPU optimization applied - limiting to 50% GPU memory")
            
        print("MuJoCo configured to use CPU rendering")
        print("System optimized for RL training")
    except Exception as e:
        print(f"Warning: Could not apply all system optimizations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--record", action="store_true", help="Record a video of the model")
    parser.add_argument("--plot", action="store_true", help="Plot learning curves")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm to use (sac, ppo, td3)")
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation or recording")
    parser.add_argument("--output", type=str, default="video.gif", help="Output path for video recording")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of timesteps for training")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--frameskip", type=int, default=2, help="Number of physics steps per control step")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes for video recording")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Apply system optimizations
    optimize_system_for_training()
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)
    
    if args.train:
        train(
            algo=args.algo,
            num_envs=args.num_envs,
            frameskip=args.frameskip,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    elif args.evaluate:
        # Get model's folder to find vec normalize path
        if args.model:
            model_dir = os.path.dirname(args.model)
            vec_path = os.path.join(model_dir, f"{args.algo}_vecnormalize.pkl")
            if not os.path.exists(vec_path):
                vec_path = None
                
            # Run evaluation
            evaluate_model(
                model_path=args.model,
                num_episodes=args.episodes,
                render=True,
                vec_normalize_path=vec_path
            )
        else:
            print("Error: --model path required for evaluation")
    elif args.record:
        # Get model's folder to find vec normalize path
        if args.model:
            model_dir = os.path.dirname(args.model)
            vec_path = os.path.join(model_dir, f"{args.algo}_vecnormalize.pkl")
            if not os.path.exists(vec_path):
                vec_path = None
                
            record_video(
                model_path=args.model,
                output_path=args.output,
                vec_normalize_path=vec_path,
                episodes=args.episodes
            )
        else:
            print("Error: --model path required for video recording")
    elif args.plot:
        plot_learning_curves(args.algo)
    else:
        parser.print_help()