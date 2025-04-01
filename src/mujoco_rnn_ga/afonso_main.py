#!/usr/bin/env python3
# filepath: /home/animat/src/mujoco_rnn_ga/afonso_main.py

"""
Train an arm to reach specific targets using reinforcement learning.
This implementation uses PPO (Proximal Policy Optimization) with 
actor-critic architecture for efficient learning.
"""

import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Process, Pipe
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # More stable than fork on Linux

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Optimize CUDA for WSL
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
else:
    print("CUDA not available, using CPU")

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
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=2000, 
                        help='Steps per training epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for updates')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_targets', type=int, default=5, help='Targets per evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--eval_interval', type=int, default=10, 
                        help='Epochs between evaluations')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--num_envs', type=int, default=4, 
                        help='Number of parallel environments for collecting experience')
    return parser.parse_args()


# Actor-Critic Networks
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_size, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
            
    def forward(self, obs):
        """Forward pass through both actor and critic networks"""
        features = self.features(obs)
        
        # Policy (actor)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)
        
        # Value (critic)
        value = self.critic(features).squeeze(-1)
        
        return action_mean, action_std, value
    
    def act(self, obs, deterministic=False):
        """Sample action from policy"""
        with torch.no_grad():
            action_mean, action_std, _ = self.forward(obs)
            
            if deterministic:
                return torch.tanh(action_mean)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.tanh(action)  # Bound actions to [-1, 1]
            
            return action
    
    def evaluate_actions(self, obs, actions):
        """Evaluate log probability and entropy of actions"""
        action_mean, action_std, value = self.forward(obs)
        
        # Convert tanh-squashed actions back to original space
        # atanh(x) = 0.5 * log((1+x)/(1-x))
        actions_orig = torch.clamp(actions, -0.999, 0.999)  # prevent numerical issues
        actions_orig = 0.5 * torch.log((1 + actions_orig) / (1 - actions_orig))
        
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions_orig).sum(dim=-1)
        
        # Apply tanh correction to log prob
        # log_prob -= sum(log(1 - tanh(a)^2))
        log_probs -= torch.sum(torch.log(1 - actions.pow(2) + 1e-6), dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy, value


class PPOBuffer:
    """Storage for PPO data collection"""
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device=DEVICE):
        # Buffers for observations, actions, and advantages
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, done, val, logp):
        """Store one timestep of data"""
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_val=0, path_idx=None):
        """
        Calculate advantages and returns for a trajectory
        
        path_idx: If provided, only finish the trajectory at this index
                 Otherwise, finish the current trajectory
        """
        if path_idx is None:
            path_slice = slice(self.path_start_idx, self.ptr)
            self.path_start_idx = self.ptr
        else:
            # Handle specific trajectory ending
            path_slice = slice(path_idx, path_idx + 1)
        
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        # Rewards-to-go (targets for value function)
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
    def get(self):
        """Get all data from buffer and normalize advantages"""
        assert self.ptr == self.max_size  # Buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        # Convert to PyTorch tensors
        data = dict(
            obs=torch.FloatTensor(self.obs_buf).to(self.device),
            act=torch.FloatTensor(self.act_buf).to(self.device),
            ret=torch.FloatTensor(self.ret_buf).to(self.device),
            adv=torch.FloatTensor(self.adv_buf).to(self.device),
            logp=torch.FloatTensor(self.logp_buf).to(self.device),
        )
        return data
    
    @staticmethod
    def discount_cumsum(x, discount):
        """Calculate discounted cumulative sum (for rewards-to-go)"""
        n = len(x)
        y = np.zeros_like(x)
        y[n-1] = x[n-1]
        for t in range(n-2, -1, -1):
            y[t] = x[t] + discount * y[t+1]
        return y


class PPOTrainer:
    """Implements PPO algorithm for training arm reaching"""
    def __init__(
        self,
        env,
        obs_dim,
        act_dim,
        hidden_size=64,
        steps_per_epoch=2000,
        batch_size=64,
        epochs=500,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        max_ep_len=1000,
        target_kl=0.01,
        num_envs=1,  # Add number of environments
        seed=42,
        device=DEVICE
    ):
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.device = device
        self.num_envs = num_envs
        
        # Adjust steps_per_epoch to account for multiple environments
        # Each env collects steps_per_env steps, so the buffer needs to be larger
        steps_per_env = steps_per_epoch // num_envs
        self.steps_per_env = steps_per_env
        
        # Initialize actor-critic network
        self.ac = ActorCritic(obs_dim, act_dim, hidden_size).to(device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Data collection buffer - now sized for all environments
        self.buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, device)
        
        # Per-environment tracking
        self.ep_rewards = [0] * num_envs
        self.ep_lengths = [0] * num_envs
        
        # Tracking variables
        self.best_reward = -np.inf
        self.training_rewards = []
        self.eval_rewards = []
        
    def compute_loss(self, data):
        """Compute PPO policy and value losses"""
        obs, act, adv, ret, old_logp = data['obs'], data['act'], data['adv'], data['ret'], data['logp']
        
        # Policy loss
        log_probs, entropy, values = self.ac.evaluate_actions(obs, act)
        
        # PPO clipped objective
        ratio = torch.exp(log_probs - old_logp)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        
        # Value loss
        value_loss = ((values - ret)**2).mean()
        
        # Entropy loss (for exploration)
        entropy_loss = -0.01 * entropy.mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Info for logging
        approx_kl = ((old_logp - log_probs)).mean().item()
        
        return loss, approx_kl
        
    def update(self, pbar=None):
        """Update policy and value networks using PPO with progress tracking"""
        data = self.buffer.get()
        
        for i in range(self.train_iters):
            # Sample mini-batch from buffer
            idxs = torch.randperm(self.steps_per_epoch)[:self.batch_size]
            batch = {
                'obs': data['obs'][idxs],
                'act': data['act'][idxs],
                'adv': data['adv'][idxs],
                'ret': data['ret'][idxs],
                'logp': data['logp'][idxs]
            }
            
            # Update networks
            self.optimizer.zero_grad()
            loss, kl = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            if pbar is not None:
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'kl': f'{kl:.3f}'})
                pbar.update(1)
            
            # Early stopping if KL divergence is too high
            if kl > 1.5 * self.target_kl:
                if pbar is not None:
                    pbar.write(f"Early stopping at iter {i}/{self.train_iters} due to reaching max KL.")
                break
    
    def collect_rollout(self):
        """Optimized rollout collection"""
        # Initialize buffer and counters
        self.buffer.ptr, self.buffer.path_start_idx = 0, 0
        
        # Pre-allocate tensor for observations (reuse instead of creating new ones)
        obs_tensor = torch.zeros((self.num_envs, self.obs_dim), 
                               device=self.device, 
                               dtype=torch.float32)
        
        # Get initial observations from all environments
        obs = self.env.reset()
        
        # Progress bar for collection
        with tqdm(total=self.steps_per_env, desc="Collecting experience", 
                  leave=False, ncols=100) as collect_pbar:
            
            # Collect data
            for step in range(self.steps_per_env):
                # Copy observations to pre-allocated tensor (more efficient)
                obs_tensor.copy_(torch.FloatTensor(obs))
                
                # Get actions and values for all environments
                with torch.no_grad():
                    action_mean, action_std, values = self.ac(obs_tensor)
                    
                    # Sample actions directly on GPU to reduce transfers
                    dist = Normal(action_mean, action_std)
                    actions_orig = dist.sample()
                    log_probs = dist.log_prob(actions_orig).sum(dim=1)
                    
                    # Apply tanh squashing on GPU
                    actions = torch.tanh(actions_orig)
                    
                    # Only transfer final actions to CPU
                    actions_np = actions.cpu().numpy()
                    log_probs_np = log_probs.cpu().numpy()
                    values_np = values.cpu().numpy()
                
                # Step all environments (this is the potential bottleneck)
                next_obs, rewards, dones, infos = self.env.step(actions_np)
                
                # Store data in buffer efficiently (vectorized operations)
                for i in range(self.num_envs):
                    self.buffer.store(
                        obs[i], 
                        actions_np[i], 
                        rewards[i], 
                        dones[i], 
                        values_np[i], 
                        log_probs_np[i]
                    )
                    
                    # Update episode stats
                    self.ep_rewards[i] += rewards[i]
                    self.ep_lengths[i] += 1
                    
                    # Handle episode termination
                    if dones[i]:
                        # Store episode data
                        self.training_rewards.append(self.ep_rewards[i])
                        
                        # Finish trajectory
                        self.buffer.finish_path(0, i * self.steps_per_env + step)
                        
                        # Reset tracking
                        self.ep_rewards[i] = 0
                        self.ep_lengths[i] = 0
                
                # Update observation
                obs = next_obs
                
                # Update progress bar (with safe average calculation)
                current_avg = np.mean(self.ep_rewards)  # Show all rewards, not just positive
                completed_episodes = len(self.training_rewards)
                recent_avg = np.mean(self.training_rewards[-10:]) if self.training_rewards else float('nan')
                collect_pbar.set_postfix({
                    'current_avg': f'{current_avg:.2f}', 
                    'completed': completed_episodes,
                    'recent_avg': f'{recent_avg:.2f}'
                })
                collect_pbar.update(1)

    def train(self):
        """Main training loop with parallel environments"""
        start_time = time.time()
        total_steps = self.epochs * self.steps_per_epoch
        
        # Outer progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc="Training Progress", ncols=100)
        
        for epoch in epoch_pbar:
            # Collect experience in parallel
            self.collect_rollout()
            
            # Update policy using progress bar
            update_pbar = tqdm(range(self.train_iters), desc="Updating policy", leave=False, ncols=100)
            self.update(update_pbar)
            
            # Track progress in the outer progress bar
            if len(self.training_rewards) > 0:
                avg_reward = np.mean(self.training_rewards[-10:]) if len(self.training_rewards) >= 10 else np.mean(self.training_rewards)
                epoch_pbar.set_postfix({'avg_reward': f'{avg_reward:.2f}'})
            
            # Only evaluate/save periodically (less frequently than before)
            is_final_epoch = epoch == self.epochs - 1
            save_epoch = (epoch + 1) % 50 == 0 or is_final_epoch
            
            if save_epoch:
                # Evaluate policy
                eval_reward = self.evaluate(render=is_final_epoch)  # Only render on final epoch
                
                # Save checkpoint (less frequently)
                self.save_model(f"checkpoint_{epoch+1}")
                
                # Log results
                elapsed_time = time.time() - start_time
                print(f"\nEpoch {epoch+1}/{self.epochs} | "
                      f"Eval reward: {eval_reward:.2f} | "
                      f"Time: {elapsed_time:.2f}s")
                
                # Save training data (less frequently)
                self.save_training_data()
        
        # Final evaluation 
        final_reward = self.evaluate(render=True)  # Only render here
        print(f"\nTraining complete. Final evaluation reward: {final_reward:.2f}")
        
        return self.training_rewards, self.eval_rewards

    def evaluate(self, num_episodes=5, render=False):
        """Evaluate the current policy without exploration"""
        eval_rewards = []
        completed_episodes = 0
        
        # Reset all environments to start evaluation
        obs = self.env.reset()
        episode_rewards = np.zeros(self.num_envs)
        episode_steps = np.zeros(self.num_envs)
        active = np.ones(self.num_envs, dtype=bool)  # Track which environments are still active
        
        while completed_episodes < num_episodes:
            # Convert observations to tensor
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # Get deterministic actions
            with torch.no_grad():
                actions = self.ac.act(obs_tensor, deterministic=True).cpu().numpy()
            
            # Step environments
            obs, rewards, dones, _ = self.env.step(actions)
            
            # Update episode rewards and steps
            episode_rewards += rewards * active  # Only add rewards for active environments
            episode_steps += active
            
            # Render first environment if requested
            if render and active[0]:
                self.env.render()
            
            # Handle completed episodes
            for i in range(self.num_envs):
                if dones[i] and active[i]:
                    eval_rewards.append(episode_rewards[i])
                    completed_episodes += 1
                    active[i] = False  # Mark this environment as completed
                    
                    # Reset this specific environment if needed
                    if completed_episodes < num_episodes:
                        # Note: our ParallelEnv auto-resets environments that are done
                        # Reset tracking for this env
                        episode_rewards[i] = 0
                        episode_steps[i] = 0
                        active[i] = True  # Reactivate for next episode
            
            # Break if we've collected enough episodes
            if completed_episodes >= num_episodes:
                break
            
            # Safety check to prevent infinite loops
            if np.all(episode_steps > self.max_ep_len):
                # Force end any episodes that hit max length
                for i in range(self.num_envs):
                    if active[i]:
                        eval_rewards.append(episode_rewards[i])
                        completed_episodes += 1
                        active[i] = False
        
        avg_reward = np.mean(eval_rewards)
        self.eval_rewards.append(avg_reward)
        
        # Save best model
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.save_model("best_model")
        
        return avg_reward

    def save_model(self, filename):
        """Save model checkpoint to disk"""
        path = MODELS_DIR / f"{filename}.pt"
        torch.save({
            'model_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards
        }, path)
        print(f"Model saved to {path}")

    def save_training_data(self):
        """Save training data for later analysis"""
        data = {
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards
        }
        path = LOGS_DIR / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_model(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_reward = checkpoint['best_reward']
        if 'training_rewards' in checkpoint:
            self.training_rewards = checkpoint['training_rewards']
        if 'eval_rewards' in checkpoint:
            self.eval_rewards = checkpoint['eval_rewards']
        print(f"Model loaded from {path}")


class RLEnvWrapper:
    """
    Wrapper for SequentialReachingEnv to provide standard RL interface methods
    (reset, step, render) required by the PPO implementation.
    """
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


# Parallel environment implementation for efficient data collection
def worker_process(remote, parent_remote, env_fn_wrapper):
    """Worker process for parallel environments"""
    parent_remote.close()
    env = env_fn_wrapper.var()  # Create environment in the worker process
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if isinstance(observation, np.ndarray):
                    # Use 32-bit float for more efficient transfers
                    observation = observation.astype(np.float32)
                if done:
                    # Auto-reset when environment is done
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
        except Exception as e:
            remote.send((str(e), None, None, None))
            break

class EnvWorker:
    """Environment variable wrapper to be sent to worker processes"""
    def __init__(self, env_fn):
        self.env_fn = env_fn
    
    def var(self):
        """Return a new environment instance"""
        return self.env_fn()


class ParallelEnv:
    """Runs multiple environments in parallel processes"""
    def __init__(self, env_fns):
        """
        env_fns: list of functions that create environments
        """
        self.closed = False
        self.waiting = False
        
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = []
        
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            args = (work_remote, remote, EnvWorker(env_fn))
            # Set daemon=True to ensure processes are terminated when main process exits
            process = Process(target=worker_process, args=args, daemon=True)
            # Set process affinity (optionally)
            # If you have 8 cores and 4 envs, distribute them
            # process.cpu_affinity([i % os.cpu_count()])
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        self.num_envs = n_envs
        
    def step(self, actions):
        """
        Take a step in all environments
        actions: list of actions, one per environment
        """
        self.step_async(actions)
        return self.step_wait()
    
    def step_async(self, actions):
        """Send step commands to all environments asynchronously"""
        if self.waiting:
            raise ValueError("Already waiting for step results")
            
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self):
        """Wait for step results from all environments"""
        if not self.waiting:
            raise ValueError("No step command pending")
            
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # Transpose the results: [obs, rewards, dones, infos]
        obs, rews, dones, infos = zip(*results)
        return np.array(obs), np.array(rews), np.array(dones), infos
    
    def reset(self):
        """Reset all environments"""
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.array([remote.recv() for remote in self.remotes])
    
    def render(self):
        """Render only the first environment"""
        self.remotes[0].send(('render', None))
        return self.remotes[0].recv()
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        
        for remote in self.remotes:
            remote.send(('close', None))
            
        for process in self.processes:
            process.join()
            
        self.closed = True


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
        
        # Wrap with RL interface
        env = RLEnvWrapper(base_env)
        
        # Set seed
        np.random.seed(self.seed)
        
        return env


def plot_training(training_rewards, eval_rewards):
    """Plot training and evaluation rewards"""
    plt.figure(figsize=(12, 6))
    
    # Plot training rewards
    if training_rewards:
        plt.subplot(1, 2, 1)
        plt.plot(training_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Add smoothed line
        if len(training_rewards) > 10:
            smoothed = np.convolve(training_rewards, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(training_rewards)), smoothed, 'r-')
    
    # Plot evaluation rewards
    if eval_rewards:
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(eval_rewards)*10, 10), eval_rewards)
        plt.title('Evaluation Rewards')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
    
    plt.tight_layout()
    
    # Save plot to disk
    plt.savefig(LOGS_DIR / f"training_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Additional GPU optimizations
    if torch.cuda.is_available():
        # Reduce memory fragmentation
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Empty cache before starting
        torch.cuda.empty_cache()
        
        # For even better performance on WSL
        torch.set_float32_matmul_precision('high')
    
    # Increase steps per epoch but reduce updates to balance speed and learning
    # This reduces Python overhead by doing fewer, larger batches
    args.steps_per_epoch = 4000  # Collect more data per epoch
    args.batch_size = 256        # Use larger batches for GPU efficiency
    
    # Define the reward weights for each component
    loss_weights = {
        'position': 1.0,    # Weight for position error
        'time': 0.1,        # Weight for time-to-target  
        'energy': 0.05,     # Weight for energy efficiency
        'progress': 0.5     # Weight for progress toward target
    }
    
    # Create multiple environments for parallel data collection
    env_fns = []
    for i in range(args.num_envs):
        # Use a class instance that can be properly pickled
        env_maker = EnvMaker(
            seed=args.seed + i,
            num_targets=args.num_targets,
            loss_weights=loss_weights
        )
        env_fns.append(env_maker)
    
    # Create the parallel environment
    parallel_env = ParallelEnv(env_fns)
    
    # Get observation and action dimensions from a temporary environment
    temp_env = env_fns[0]()
    obs_dim = temp_env.plant.num_sensors + 3  # Plant sensors + target (x,y,z)
    act_dim = temp_env.plant.num_actuators    # Number of muscles
    temp_env.close()  # Close the temporary environment
    
    print(f"Using {args.num_envs} parallel environments")
    print(f"Observation space: {obs_dim} dimensions")
    print(f"Action space: {act_dim} dimensions")
    
    # Create trainer with parallel environment
    trainer = PPOTrainer(
        env=parallel_env,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_size,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gamma=args.gamma,
        lr=args.lr,
        seed=args.seed,
        num_envs=args.num_envs
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_model(args.resume)
    
    # Train model
    training_rewards, eval_rewards = trainer.train()
    
    # Plot results
    plot_training(training_rewards, eval_rewards)
    
    # Save final model
    trainer.save_model("final_model")
    
    # Close environment
    parallel_env.close()


if __name__ == "__main__":
    main()