#!/usr/bin/env python3

"""
Train an arm to reach targets using reinforcement learning with MuJoCo XLA and StableBaselines3.
This implementation achieves significant speedup through vectorized physics simulation with MJX.
"""

import os
import time
import argparse
import gc
import numpy as np
import jax
from jax import random
import torch
from pathlib import Path

# Set XLA flags for performance before importing JAX
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=4 --xla_gpu_triton_gemm_any=true'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# JAX configuration
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "default")

# Check if JAX and PyTorch can use GPU
print(f"JAX is using: {jax.devices()}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch using: {torch.cuda.get_device_name(0)}")

# Import project modules
from environment import MJXSequentialReacher
from agents import SACAgent  
from utils import get_root_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train arm reaching with MJX and SAC')
    
    # Core training parameters
    parser.add_argument('--num_envs', type=int, default=128,
                        help='Number of parallel environments')
    parser.add_argument('--total_steps', type=int, default=5_000_000,
                        help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Environment parameters
    parser.add_argument('--num_targets', type=int, default=5,
                        help='Number of targets per episode')
    parser.add_argument('--target_duration', type=float, default=3.0,
                        help='Duration for each target (seconds)')
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Use curriculum learning')
    parser.add_argument('--curriculum_level', type=int, default=0,
                        help='Initial curriculum level (0-3)')
    
    # SAC specific parameters
    parser.add_argument('--buffer_size', type=int, default=2000000,
                        help='Replay buffer size for SAC')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for SAC updates')
    parser.add_argument('--train_freq', type=int, default=1, 
                        help='How often to update the model')
    parser.add_argument('--gradient_steps', type=int, default=2,
                        help='Gradient steps per update')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network update rate for SAC')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient (if not using auto_entropy)')
    parser.add_argument('--auto_entropy', action='store_true', default=True,
                        help='Use automatic entropy tuning for SAC')
    
    # Logging and saving
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def benchmark_env_speed(env, num_steps=100):
    """Benchmark environment speed without RL overhead."""
    actions = np.random.uniform(-1, 1, (env.num_envs, env.num_actuators))
    
    # Warmup
    for _ in range(10):
        _, _, _, info = env.step(actions)
        if hasattr(env, 'use_curriculum') and env.use_curriculum:
            if "target_reached" in info:
                env.update_curriculum(np.array(info["target_reached"]))
    
    # Benchmark
    start = time.time()
    for _ in range(num_steps):
        _, _, _, info = env.step(actions)
        if hasattr(env, 'use_curriculum') and env.use_curriculum:
            if "target_reached" in info:
                env.update_curriculum(np.array(info["target_reached"]))
    end = time.time()
    
    # Calculate and report steps per second
    steps_per_second = (num_steps * env.num_envs) / (end - start)
    print(f"Environment speed: {steps_per_second:.1f} steps/second")
    return steps_per_second

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random_key = random.PRNGKey(args.seed)
    
    # Enable PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    # Optimize memory before training
    print("Optimizing memory before training...")
    torch.cuda.empty_cache()
    jax.clear_caches()
    gc.collect()
    
    # Create environment
    print(f"Creating {args.num_envs} parallel environments with MJX...")
    env = MJXSequentialReacher(
        num_envs=args.num_envs,
        num_targets=args.num_targets,
        target_duration=args.target_duration,
        use_curriculum=args.use_curriculum,
        curriculum_level=args.curriculum_level
    )
    
    # Benchmark environment speed
    print("Benchmarking environment speed...")
    env_speed = benchmark_env_speed(env)
    
    # Set up logging directory
    save_dir = Path(get_root_path()) / args.save_dir
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f"To view training progress, run: tensorboard --logdir={log_dir}")
    
    # Create agent
    agent = SACAgent(
        env=env,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        tau=args.tau,
        auto_entropy=args.auto_entropy,
        entropy_coef=args.entropy_coef,
        normalize_observations=False,  # We're normalizing in the environment
        log_dir=str(log_dir),
        device=args.device
    )

    # Start training
    print(f"Starting SAC training for {args.total_steps} timesteps...")
    start_time = time.time()
    agent.train(args.total_steps)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = save_dir / "final_sac_model"
    agent.save(str(model_path))
    print(f"Final model saved to {model_path}")
    
    # Clean up memory
    gc.collect()
    jax.clear_caches()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("\nTraining completed successfully!")

    # Print instructions for accessing TensorBoard logs
    print("\n" + "="*50)
    print("To view training metrics in TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print(f"  Then open: http://localhost:6006")
    print("="*50)

if __name__ == "__main__":
    main()