"""Shared evaluation utilities for training and testing."""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple

from core.constants import DEFAULT_MAX_EPISODE_STEPS, DEFAULT_NUM_EVAL_EPISODES
from envs.reaching import ReachingEnv


def evaluate_controller(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict[str, np.ndarray],
    num_episodes: int = DEFAULT_NUM_EVAL_EPISODES,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a controller.

    Returns:
        Dictionary with success_rate, mean_reward, std_reward, mean_episode_length,
        mean_reach_time, mean_final_distance, and raw data arrays.
    """
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    controller.eval()

    episode_rewards, episode_lengths = [], []
    successes = 0
    reach_times, final_distances = [], []

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, device)

            episode_reward = 0.0
            reach_time = None

            for step in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if reach_time is None and info.get("distance_to_target", 1.0) < env.reach_threshold:
                    reach_time = step * env.dt

                if terminated or truncated:
                    if info.get("phase") == "done":
                        successes += 1
                    final_distances.append(info.get("distance_to_target", 1.0))
                    episode_lengths.append(step + 1)
                    if reach_time is not None:
                        reach_times.append(reach_time)
                    break

            episode_rewards.append(episode_reward)

            if verbose and (ep + 1) % 20 == 0:
                print(f"  Episode {ep + 1}/{num_episodes}: reward={episode_reward:.2f}, success_rate={successes / (ep + 1):.2%}")

    env.close()

    return {
        "num_episodes": num_episodes,
        "success_rate": successes / num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_reach_time": np.mean(reach_times) if reach_times else None,
        "mean_final_distance": np.mean(final_distances),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "reach_times": reach_times,
        "final_distances": final_distances,
    }


def evaluate_fitness(
    flat_params: np.ndarray,
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict[str, np.ndarray],
    num_episodes: int = DEFAULT_NUM_EVAL_EPISODES,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate fitness for evolutionary algorithms (CMA-ES)."""
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
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
    controller: Optional[torch.nn.Module],
    xml_path: str,
    sensor_stats: Dict[str, np.ndarray],
    num_episodes: int = 100,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: torch.device = torch.device("cpu"),
    random_actions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect trajectory data for distillation learning."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)

    all_observations, all_actions = [], []

    if not random_actions and controller is not None:
        controller.eval()

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            if not random_actions and controller is not None:
                controller.init_hidden(1, device)

            ep_observations, ep_actions = [obs.copy()], []

            for _ in range(max_steps):
                if random_actions:
                    action = env.action_space.sample()
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action, _, _ = controller.forward(obs_tensor)
                    action = action.squeeze(0).cpu().numpy()

                obs, _, terminated, truncated, _ = env.step(action)
                ep_observations.append(obs.copy())
                ep_actions.append(action.copy())

                if terminated or truncated:
                    break

            all_observations.append(np.array(ep_observations[:-1]))
            all_actions.append(np.array(ep_actions))

    env.close()
    return all_observations, all_actions
