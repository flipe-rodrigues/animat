"""
Shared evaluation utilities for training and testing.

Provides common functionality for evaluating controllers across different training methods.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import (
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_CALIBRATION_EPISODES,
    DEFAULT_NUM_EVAL_EPISODES,
)
from envs.reaching import ReachingEnv


def evaluate_controller(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict[str, np.ndarray],
    num_episodes: int = DEFAULT_CALIBRATION_EPISODES,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a controller.
    
    Args:
        controller: Neural network controller to evaluate
        xml_path: Path to MuJoCo XML file
        sensor_stats: Sensor normalization statistics
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        device: Device to run controller on
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation metrics:
        - success_rate: Fraction of successfully completed trials
        - mean_reward: Average total reward per episode
        - std_reward: Standard deviation of rewards
        - mean_episode_length: Average episode length
        - mean_reach_time: Average time to reach target (when successful)
        - mean_final_distance: Average distance to target at episode end
        - episode_rewards: List of all episode rewards
        - episode_lengths: List of all episode lengths
    """
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    controller.eval()
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    reach_times = []
    final_distances = []
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset()
            controller.init_hidden(1, device)
            
            episode_reward = 0.0
            reach_time = None
            
            for step in range(max_steps):
                # Get action from controller
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                # Step environment
                obs, reward, terminated, truncated, step_info = env.step(action)
                episode_reward += reward
                
                # Track when target is first reached
                if reach_time is None and step_info.get('distance_to_target', 1.0) < env.reach_threshold:
                    reach_time = step * env.dt
                
                # Check if episode is done
                if terminated or truncated:
                    if step_info.get('phase') == 'done':
                        successes += 1
                    
                    final_distances.append(step_info.get('distance_to_target', 1.0))
                    episode_lengths.append(step + 1)
                    
                    if reach_time is not None:
                        reach_times.append(reach_time)
                    
                    break
            
            episode_rewards.append(episode_reward)
            
            if verbose and (ep + 1) % 20 == 0:
                current_success_rate = successes / (ep + 1)
                print(f"  Episode {ep+1}/{num_episodes}: "
                      f"reward={episode_reward:.2f}, "
                      f"success_rate={current_success_rate:.2%}")
    
    env.close()
    
    # Compute summary statistics
    results = {
        'num_episodes': num_episodes,
        'success_rate': successes / num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_reach_time': np.mean(reach_times) if reach_times else None,
        'mean_final_distance': np.mean(final_distances),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'reach_times': reach_times,
        'final_distances': final_distances,
    }
    
    return results


def evaluate_fitness(
    flat_params: np.ndarray,
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict[str, np.ndarray],
    num_episodes: int = DEFAULT_NUM_EVAL_EPISODES,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: torch.device = torch.device('cpu'),
) -> float:
    """
    Evaluate fitness for evolutionary algorithms (CMA-ES).
    
    Args:
        flat_params: Flattened network parameters
        controller: Controller to evaluate
        xml_path: Path to MuJoCo XML file
        sensor_stats: Sensor normalization statistics
        num_episodes: Number of episodes to average over
        max_steps: Maximum steps per episode
        device: Device to run controller on
        
    Returns:
        Mean total reward over episodes (fitness score)
    """
    # Set controller parameters
    controller.set_flat_params(flat_params)
    controller.eval()
    
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, device)
            
            episode_reward = 0.0
            
            for _ in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
    
    env.close()
    
    return float(np.mean(total_rewards))


def collect_trajectory_data(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict[str, np.ndarray],
    num_episodes: int = DEFAULT_CALIBRATION_EPISODES,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: torch.device = torch.device('cpu'),
    random_actions: bool = False,
) -> tuple:
    """
    Collect trajectory data for distillation learning.
    
    Args:
        controller: Controller to collect data from (None if random_actions=True)
        xml_path: Path to MuJoCo XML file
        sensor_stats: Sensor normalization statistics
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        device: Device to run controller on
        random_actions: If True, use random actions instead of controller
        
    Returns:
        Tuple of (observations_list, actions_list) where each is a list of episode trajectories
    """
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    
    all_observations = []
    all_actions = []
    
    if not random_actions:
        controller.eval()
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            
            if not random_actions:
                controller.init_hidden(1, device)
            
            ep_observations = [obs.copy()]
            ep_actions = []
            
            for step in range(max_steps):
                if random_actions:
                    action = env.action_space.sample()
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    action, _, _ = controller.forward(obs_tensor)
                    action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                
                ep_observations.append(obs.copy())
                ep_actions.append(action.copy())
                
                if terminated or truncated:
                    break
            
            # Don't include last observation (no corresponding action)
            all_observations.append(np.array(ep_observations[:-1]))
            all_actions.append(np.array(ep_actions))
    
    env.close()
    
    return all_observations, all_actions
