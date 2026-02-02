"""
Visualization and Evaluation Utilities

Provides tools for:
- Rendering episodes to video
- Plotting training curves
- Evaluating controllers
- Analyzing network activity
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Tuple
import json
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.reaching_env import ReachingEnv
from models.controllers import RNNController, MLPController, ModelConfig


def load_controller(
    checkpoint_path: str,
    controller_type: str = 'rnn'
) -> Tuple[torch.nn.Module, ModelConfig, Dict]:
    """
    Load a trained controller from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        controller_type: 'rnn' or 'mlp'
        
    Returns:
        controller: Loaded controller
        config: Model configuration
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config_dict = checkpoint['model_config']
    config = ModelConfig(**config_dict)
    
    if controller_type == 'rnn':
        controller = RNNController(config)
    else:
        controller = MLPController(config)
    
    controller.load_state_dict(checkpoint['model_state_dict'])
    controller.eval()
    
    return controller, config, checkpoint


def evaluate_controller(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict,
    n_episodes: int = 100,
    max_steps: int = 300,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a controller.
    
    Returns:
        Dictionary with evaluation metrics
    """
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    
    device = next(controller.parameters()).device
    controller.eval()
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    reach_times = []
    hold_times = []
    final_distances = []
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            controller.init_hidden(1, device)
            
            episode_reward = 0
            reach_time = None
            
            for step in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, step_info = env.step(action)
                episode_reward += reward
                
                # Track reach time
                if reach_time is None and step_info.get('distance_to_target', 1.0) < 0.05:
                    reach_time = step
                
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
                print(f"Episode {ep+1}/{n_episodes}: reward={episode_reward:.2f}, "
                      f"success_rate={successes/(ep+1):.2%}")
    
    env.close()
    
    results = {
        'n_episodes': n_episodes,
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_reach_time': np.mean(reach_times) if reach_times else None,
        'mean_final_distance': np.mean(final_distances),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    return results


def record_episode(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict,
    max_steps: int = 300
) -> Dict[str, Any]:
    """
    Record a single episode with full trajectory data.
    
    Returns:
        Dictionary with observations, actions, rewards, and frames
    """
    env = ReachingEnv(xml_path, render_mode='rgb_array', sensor_stats=sensor_stats)
    device = next(controller.parameters()).device
    controller.eval()
    
    obs, info = env.reset()
    controller.init_hidden(1, device)
    
    trajectory = {
        'observations': [obs.copy()],
        'actions': [],
        'rewards': [],
        'infos': [info],
        'frames': [],
        'alpha': [],
        'gamma_static': [],
        'gamma_dynamic': [],
        'sensory_Ia': [],
        'sensory_II': [],
        'sensory_Ib': [],
    }
    
    with torch.no_grad():
        for step in range(max_steps):
            # Render
            frame = env.render()
            if frame is not None:
                trajectory['frames'].append(frame)
            
            # Get action
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action, _, step_info = controller.forward(obs_tensor)
            action = action.squeeze(0).cpu().numpy()
            
            # Store network outputs
            trajectory['alpha'].append(step_info['alpha'].squeeze(0).cpu().numpy())
            trajectory['gamma_static'].append(step_info['gamma_static'].squeeze(0).cpu().numpy())
            trajectory['gamma_dynamic'].append(step_info['gamma_dynamic'].squeeze(0).cpu().numpy())
            
            if 'sensory_outputs' in step_info:
                sensory = step_info['sensory_outputs']
                trajectory['sensory_Ia'].append(sensory['type_Ia'].squeeze(0).cpu().numpy())
                trajectory['sensory_II'].append(sensory['type_II'].squeeze(0).cpu().numpy())
                trajectory['sensory_Ib'].append(sensory['type_Ib'].squeeze(0).cpu().numpy())
            
            # Step environment
            obs, reward, terminated, truncated, env_info = env.step(action)
            
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)
            trajectory['infos'].append(env_info)
            
            if terminated or truncated:
                break
    
    # Convert to arrays
    for key in ['observations', 'actions', 'rewards', 'alpha', 'gamma_static', 
                'gamma_dynamic', 'sensory_Ia', 'sensory_II', 'sensory_Ib']:
        if trajectory[key]:
            trajectory[key] = np.array(trajectory[key])
    
    env.close()
    return trajectory


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
):
    """Save frames as video."""
    import cv2
    
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Video saved to {output_path}")


def plot_trajectory(
    trajectory: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = True
):
    """Plot detailed trajectory analysis."""
    fig = plt.figure(figsize=(16, 12))
    
    n_steps = len(trajectory['rewards'])
    time = np.arange(n_steps) * 0.01  # Assuming 10ms timestep
    
    # 1. Cumulative reward
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(time, np.cumsum(trajectory['rewards']))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Episode Reward')
    ax1.grid(True)
    
    # 2. Distance to target
    distances = [info.get('distance_to_target', np.nan) for info in trajectory['infos'][1:]]
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(time, distances)
    ax2.axhline(y=0.05, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance to Target')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Alpha MN activations
    ax3 = fig.add_subplot(3, 3, 3)
    alpha = trajectory['alpha']
    for i in range(alpha.shape[1]):
        ax3.plot(time, alpha[:, i], label=f'Muscle {i+1}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Activation')
    ax3.set_title('Alpha Motor Neuron Activations')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Gamma static
    ax4 = fig.add_subplot(3, 3, 4)
    gamma_s = trajectory['gamma_static']
    for i in range(gamma_s.shape[1]):
        ax4.plot(time, gamma_s[:, i], label=f'Muscle {i+1}')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Gain')
    ax4.set_title('Gamma Static (Length Modulation)')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Gamma dynamic
    ax5 = fig.add_subplot(3, 3, 5)
    gamma_d = trajectory['gamma_dynamic']
    for i in range(gamma_d.shape[1]):
        ax5.plot(time, gamma_d[:, i], label=f'Muscle {i+1}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Gain')
    ax5.set_title('Gamma Dynamic (Velocity Modulation)')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Sensory Ia (length + velocity)
    if len(trajectory['sensory_Ia']) > 0:
        ax6 = fig.add_subplot(3, 3, 6)
        Ia = trajectory['sensory_Ia']
        for i in range(Ia.shape[1]):
            ax6.plot(time, Ia[:, i], label=f'Muscle {i+1}')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Activity')
        ax6.set_title('Type Ia Sensory Neurons')
        ax6.legend()
        ax6.grid(True)
    
    # 7. Sensory II (length)
    if len(trajectory['sensory_II']) > 0:
        ax7 = fig.add_subplot(3, 3, 7)
        II = trajectory['sensory_II']
        for i in range(II.shape[1]):
            ax7.plot(time, II[:, i], label=f'Muscle {i+1}')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Activity')
        ax7.set_title('Type II Sensory Neurons')
        ax7.legend()
        ax7.grid(True)
    
    # 8. Sensory Ib (force)
    if len(trajectory['sensory_Ib']) > 0:
        ax8 = fig.add_subplot(3, 3, 8)
        Ib = trajectory['sensory_Ib']
        for i in range(Ib.shape[1]):
            ax8.plot(time, Ib[:, i], label=f'Muscle {i+1}')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Activity')
        ax8.set_title('Type Ib Sensory Neurons (GTO)')
        ax8.legend()
        ax8.grid(True)
    
    # 9. Phase indicators
    ax9 = fig.add_subplot(3, 3, 9)
    phases = [info.get('phase', 'unknown') for info in trajectory['infos'][1:]]
    phase_map = {'pre_delay': 0, 'reach': 1, 'hold': 2, 'post_delay': 3, 'done': 4}
    phase_numeric = [phase_map.get(p, -1) for p in phases]
    ax9.plot(time, phase_numeric, drawstyle='steps-post')
    ax9.set_yticks([0, 1, 2, 3, 4])
    ax9.set_yticklabels(['pre_delay', 'reach', 'hold', 'post_delay', 'done'])
    ax9.set_xlabel('Time (s)')
    ax9.set_title('Trial Phase')
    ax9.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    history_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    """Plot training curves from history file."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Best fitness
    axes[0].plot(history['best_fitness_history'], label='Best')
    axes[0].plot(history['mean_fitness_history'], label='Mean', alpha=0.7)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness')
    axes[0].set_title('Fitness History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Sigma
    axes[1].plot(history['sigma_history'])
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Sigma')
    axes[1].set_title('CMA-ES Step Size')
    axes[1].grid(True)
    
    # Fitness distribution over time
    best = np.array(history['best_fitness_history'])
    mean = np.array(history['mean_fitness_history'])
    axes[2].fill_between(range(len(best)), mean, best, alpha=0.3)
    axes[2].plot(best, label='Best')
    axes[2].plot(mean, label='Mean')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Fitness')
    axes[2].set_title('Fitness Range')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_controllers(
    controller_paths: List[str],
    labels: List[str],
    xml_path: str,
    sensor_stats: Dict,
    n_episodes: int = 50
) -> Dict[str, Any]:
    """Compare multiple controllers."""
    results = {}
    
    for path, label in zip(controller_paths, labels):
        print(f"\nEvaluating {label}...")
        
        # Detect controller type
        controller_type = 'rnn' if 'rnn' in path.lower() else 'mlp'
        controller, config, _ = load_controller(path, controller_type)
        
        eval_results = evaluate_controller(
            controller, xml_path, sensor_stats,
            n_episodes=n_episodes, verbose=False
        )
        
        results[label] = eval_results
        print(f"  Success rate: {eval_results['success_rate']:.2%}")
        print(f"  Mean reward: {eval_results['mean_reward']:.2f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    labels_list = list(results.keys())
    success_rates = [results[l]['success_rate'] for l in labels_list]
    mean_rewards = [results[l]['mean_reward'] for l in labels_list]
    
    x = np.arange(len(labels_list))
    
    axes[0].bar(x, success_rates)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_list)
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylim(0, 1)
    
    axes[1].bar(x, mean_rewards)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_list)
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Mean Reward Comparison')
    
    plt.tight_layout()
    plt.show()
    
    return results


##############################################################################
# Network Weight Inspection
##############################################################################

def get_weight_summary(controller: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get summary statistics for all weights in the network.
    
    Returns:
        Dict mapping layer name to stats (shape, mean, std, min, max, sparsity)
    """
    summary = {}
    
    for name, param in controller.named_parameters():
        data = param.data.cpu().numpy()
        summary[name] = {
            'shape': data.shape,
            'numel': data.size,
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'abs_mean': float(np.mean(np.abs(data))),
            'sparsity': float(np.mean(np.abs(data) < 0.01)),  # fraction near zero
            'norm': float(np.linalg.norm(data)),
        }
    
    return summary


def print_weight_summary(controller: torch.nn.Module):
    """Print a formatted summary of network weights."""
    summary = get_weight_summary(controller)
    
    print(f"\n{'='*80}")
    print("Network Weight Summary")
    print(f"{'='*80}")
    print(f"{'Layer':<45} {'Shape':<20} {'Mean':>8} {'Std':>8} {'Sparsity':>8}")
    print(f"{'-'*80}")
    
    total_params = 0
    for name, stats in summary.items():
        shape_str = str(stats['shape'])
        print(f"{name:<45} {shape_str:<20} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['sparsity']:>8.2%}")
        total_params += stats['numel']
    
    print(f"{'-'*80}")
    print(f"Total parameters: {total_params:,}")


def plot_weight_distributions(
    controller: torch.nn.Module,
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Plot histograms of weight distributions for all layers.
    """
    params = list(controller.named_parameters())
    n_params = len(params)
    
    # Determine grid size
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, (name, param) in enumerate(params):
        ax = axes[idx]
        data = param.data.cpu().numpy().flatten()
        
        ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"{name}\n({param.shape})", fontsize=8)
        ax.set_xlabel('Weight', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Weight Distributions by Layer', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Weight distributions saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_weight_matrices(
    controller: torch.nn.Module,
    layer_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    cmap: str = 'RdBu_r'
):
    """
    Plot weight matrices as heatmaps for specified layers.
    
    Args:
        controller: The neural network
        layer_names: List of layer names to plot (None = all 2D weights)
        output_path: Path to save figure
        show: Whether to display
        cmap: Colormap for heatmap
    """
    # Collect 2D weight matrices
    matrices = []
    for name, param in controller.named_parameters():
        if param.dim() == 2:  # Only plot 2D weight matrices
            if layer_names is None or any(ln in name for ln in layer_names):
                matrices.append((name, param.data.cpu().numpy()))
    
    if not matrices:
        print("No 2D weight matrices found matching criteria")
        return
    
    n_matrices = len(matrices)
    n_cols = min(3, n_matrices)
    n_rows = (n_matrices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_matrices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, weights) in enumerate(matrices):
        ax = axes[idx]
        
        # Symmetric color limits for diverging colormap
        vmax = np.abs(weights).max()
        vmin = -vmax
        
        im = ax.imshow(weights, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\n{weights.shape}", fontsize=9)
        ax.set_xlabel('Input', fontsize=8)
        ax.set_ylabel('Output', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_matrices, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Weight Matrices', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Weight matrices saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_reflex_connections(
    controller: torch.nn.Module,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot the direct reflex connections (Ia->Alpha, II->Alpha) as heatmaps.
    
    These show the strength of monosynaptic stretch reflex pathways.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Find reflex weight matrices
    Ia_weights = None
    II_weights = None
    
    for name, param in controller.named_parameters():
        if 'Ia_to_alpha' in name and 'weight' in name:
            Ia_weights = param.data.cpu().numpy()
        elif 'II_to_alpha' in name and 'weight' in name:
            II_weights = param.data.cpu().numpy()
    
    if Ia_weights is None or II_weights is None:
        print("Reflex connections not found in this controller")
        return
    
    # Plot Ia -> Alpha
    vmax = max(np.abs(Ia_weights).max(), np.abs(II_weights).max())
    
    im1 = axes[0].imshow(Ia_weights, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Type Ia → Alpha MN\n(length + velocity reflex)', fontsize=10)
    axes[0].set_xlabel('Ia Sensory Neuron (muscle)')
    axes[0].set_ylabel('Alpha MN (muscle)')
    plt.colorbar(im1, ax=axes[0])
    
    # Add diagonal reference
    n = Ia_weights.shape[0]
    for i in range(n):
        axes[0].add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='green', linewidth=2))
    
    # Plot II -> Alpha
    im2 = axes[1].imshow(II_weights, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Type II → Alpha MN\n(length reflex)', fontsize=10)
    axes[1].set_xlabel('II Sensory Neuron (muscle)')
    axes[1].set_ylabel('Alpha MN (muscle)')
    plt.colorbar(im2, ax=axes[1])
    
    for i in range(n):
        axes[1].add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='green', linewidth=2))
    
    plt.suptitle('Stretch Reflex Connections\n(green boxes = same-muscle diagonal)', fontsize=11)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reflex connections saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_sensory_weights(
    controller: torch.nn.Module,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot the sensory neuron weights (Ia, II, Ib) for each muscle.
    
    Ia = velocity input (1 weight per muscle)
    II = length input (1 weight per muscle)
    Ib = force input (1 weight per muscle)
    """
    # Extract sensory weights
    Ia_weights = []
    II_weights = []
    Ib_weights = []
    
    for name, param in controller.named_parameters():
        if 'proprioceptive.type_Ia' in name and 'weight' in name:
            Ia_weights.append(param.data.cpu().numpy().flatten()[0])
        elif 'proprioceptive.type_II' in name and 'weight' in name:
            II_weights.append(param.data.cpu().numpy().flatten()[0])
        elif 'proprioceptive.type_Ib' in name and 'weight' in name:
            Ib_weights.append(param.data.cpu().numpy().flatten()[0])
    
    if not Ia_weights:
        print("Sensory weights not found")
        return
    
    n_muscles = len(Ia_weights)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x = np.arange(n_muscles)
    bar_width = 0.6
    
    # Type Ia (velocity)
    axes[0].bar(x, Ia_weights, bar_width, color='orange', edgecolor='black')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'M{i+1}' for i in range(n_muscles)])
    axes[0].set_xlabel('Muscle')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Type Ia (Velocity)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Type II (length)
    axes[1].bar(x, II_weights, bar_width, color='blue', edgecolor='black')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'M{i+1}' for i in range(n_muscles)])
    axes[1].set_xlabel('Muscle')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Type II (Length)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Type Ib (force)
    axes[2].bar(x, Ib_weights, bar_width, color='red', edgecolor='black')
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'M{i+1}' for i in range(n_muscles)])
    axes[2].set_xlabel('Muscle')
    axes[2].set_ylabel('Weight')
    axes[2].set_title('Type Ib (Force/GTO)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Sensory Neuron Input Weights', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Sensory weights saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rnn_weights(
    controller: torch.nn.Module,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot RNN recurrent weights and input weights.
    """
    rnn_weights = {}
    
    for name, param in controller.named_parameters():
        if 'rnn.' in name:
            rnn_weights[name] = param.data.cpu().numpy()
    
    if not rnn_weights:
        print("RNN weights not found")
        return
    
    n_weights = len(rnn_weights)
    n_cols = min(2, n_weights)
    n_rows = (n_weights + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_weights == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, weights) in enumerate(rnn_weights.items()):
        ax = axes[idx]
        
        if weights.ndim == 2:
            vmax = np.abs(weights).max()
            im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax)
        else:
            ax.bar(range(len(weights)), weights)
        
        ax.set_title(f"{name}\n{weights.shape}", fontsize=9)
    
    for idx in range(n_weights, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('RNN Weights', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"RNN weights saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_weights(
    controller: torch.nn.Module,
    output_dir: str = ".",
    show: bool = False
):
    """
    Generate all weight visualization plots and save to directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating weight visualizations...")
    
    print_weight_summary(controller)
    
    plot_weight_distributions(controller, 
                             output_path=str(output_dir / "weight_distributions.png"),
                             show=show)
    
    plot_weight_matrices(controller,
                        output_path=str(output_dir / "weight_matrices.png"),
                        show=show)
    
    plot_reflex_connections(controller,
                           output_path=str(output_dir / "reflex_connections.png"),
                           show=show)
    
    plot_sensory_weights(controller,
                        output_path=str(output_dir / "sensory_weights.png"),
                        show=show)
    
    plot_rnn_weights(controller,
                    output_path=str(output_dir / "rnn_weights.png"),
                    show=show)
    
    print(f"\nAll visualizations saved to {output_dir}")


def plot_episode_summary(
    trajectory: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 14)
):
    """
    Plot comprehensive episode summary with sensor data, kinematics, and rewards.
    
    Args:
        trajectory: Dict from record_episode() containing observations, actions, 
                   rewards, infos, and network outputs
        output_path: Path to save figure
        show: Whether to display
        figsize: Figure size
    """
    n_steps = len(trajectory['rewards'])
    time = np.arange(n_steps) * 0.01  # Assuming 10ms timestep
    
    # Extract data from trajectory
    obs = trajectory['observations']
    infos = trajectory['infos'][1:] if len(trajectory['infos']) > n_steps else trajectory['infos']
    rewards = trajectory['rewards']
    
    # Determine number of muscles from alpha shape
    if 'alpha' in trajectory and len(trajectory['alpha']) > 0:
        n_muscles = trajectory['alpha'].shape[1]
    else:
        n_muscles = 4  # Default
    
    # Parse observations to get sensor data
    # Layout: [lengths, velocities, forces, target_grid, phase, time]
    obs_array = np.array(obs[:n_steps])
    
    lengths = obs_array[:, :n_muscles]
    velocities = obs_array[:, n_muscles:2*n_muscles]
    forces = obs_array[:, 2*n_muscles:3*n_muscles]
    
    # Target grid starts after proprioceptive
    proprio_dim = 3 * n_muscles
    # Try to infer target grid size
    remaining = obs_array.shape[1] - proprio_dim - 4  # -4 for phase(3) + time(1)
    target_grid_size = int(np.sqrt(remaining)) if remaining > 0 else 5
    target_dim = target_grid_size ** 2
    
    target_encoding = obs_array[:, proprio_dim:proprio_dim + target_dim]
    
    # Extract kinematics from infos
    hand_positions = []
    target_positions = []
    distances = []
    phases = []
    
    for info in infos:
        hand_positions.append(info.get('hand_position', [0, 0, 0]))
        target_positions.append(info.get('target_position', [0, 0, 0]))
        distances.append(info.get('distance_to_target', 0))
        phases.append(info.get('phase', 'unknown'))
    
    hand_positions = np.array(hand_positions)
    target_positions = np.array(target_positions)
    distances = np.array(distances)
    
    # Compute hand velocity
    if len(hand_positions) > 1:
        hand_velocity = np.diff(hand_positions, axis=0) / 0.01  # velocity in m/s
        hand_speed = np.linalg.norm(hand_velocity, axis=1)
        hand_speed = np.concatenate([[0], hand_speed])  # Pad to match length
    else:
        hand_speed = np.zeros(n_steps)
    
    # Compute reward components (approximate based on typical reward function)
    distance_reward = -distances
    reach_bonus = (distances < 0.05).astype(float) * 0.5
    
    # Energy penalty from alpha
    if 'alpha' in trajectory and len(trajectory['alpha']) > 0:
        alpha = trajectory['alpha']
        energy_penalty = -0.01 * np.sum(alpha ** 2, axis=1)
    else:
        energy_penalty = np.zeros(n_steps)
    
    # Phase encoding for shading
    phase_map = {'pre_delay': 0, 'reach': 1, 'hold': 2, 'post_delay': 3, 'done': 4}
    phase_colors = {
        'pre_delay': '#FFE4B5',  # Light orange
        'reach': '#E6F3FF',       # Light blue
        'hold': '#E6FFE6',        # Light green
        'post_delay': '#FFE6FF',  # Light pink
        'done': '#F0F0F0'         # Light gray
    }
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)
    
    def add_phase_shading(ax, phases, time):
        """Add colored background for each phase."""
        if len(phases) == 0:
            return
        current_phase = phases[0]
        start_idx = 0
        
        for i, phase in enumerate(phases):
            if phase != current_phase or i == len(phases) - 1:
                end_idx = i if phase != current_phase else i + 1
                if current_phase in phase_colors:
                    ax.axvspan(time[start_idx], time[min(end_idx, len(time)-1)], 
                              alpha=0.3, color=phase_colors[current_phase], zorder=0)
                current_phase = phase
                start_idx = i
    
    # ===== Row 1: Muscle Lengths (Type II input) =====
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(n_muscles):
        ax1.plot(time, lengths[:, i], label=f'Muscle {i+1}', linewidth=1.5)
    add_phase_shading(ax1, phases, time)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Length (norm)')
    ax1.set_title('Muscle Lengths (Type II Sensory Input)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ===== Row 1: Muscle Velocities (Type Ia input) =====
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(n_muscles):
        ax2.plot(time, velocities[:, i], label=f'Muscle {i+1}', linewidth=1.5)
    add_phase_shading(ax2, phases, time)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (norm)')
    ax2.set_title('Muscle Velocities (Type Ia Sensory Input)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ===== Row 2: Muscle Forces (Type Ib input) =====
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(n_muscles):
        ax3.plot(time, forces[:, i], label=f'Muscle {i+1}', linewidth=1.5)
    add_phase_shading(ax3, phases, time)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (norm)')
    ax3.set_title('Muscle Forces (Type Ib / GTO Input)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ===== Row 2: Target Grid Encoding =====
    ax4 = fig.add_subplot(gs[1, 1])
    # Show as heatmap over time (subsample if needed)
    n_show = min(100, n_steps)
    step_indices = np.linspace(0, n_steps-1, n_show, dtype=int)
    target_subset = target_encoding[step_indices, :]
    
    im = ax4.imshow(target_subset.T, aspect='auto', cmap='viridis',
                    extent=[time[0], time[-1], 0, target_dim])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Target Grid Unit')
    ax4.set_title('Target Grid Encoding (Gaussian-tuned)', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Activation')
    
    # ===== Row 3: Hand Kinematics (velocity + distance) =====
    ax5 = fig.add_subplot(gs[2, :])
    
    # Distance to target
    ln1 = ax5.plot(time, distances, 'b-', linewidth=2, label='Distance to Target')
    ax5.axhline(y=0.05, color='b', linestyle='--', alpha=0.5, label='Reach Threshold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Distance (m)', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    
    # Hand speed on secondary axis
    ax5b = ax5.twinx()
    ln2 = ax5b.plot(time, hand_speed, 'r-', linewidth=2, alpha=0.7, label='Hand Speed')
    ax5b.set_ylabel('Speed (m/s)', color='r')
    ax5b.tick_params(axis='y', labelcolor='r')
    
    add_phase_shading(ax5, phases, time)
    
    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='upper right')
    ax5.set_title('Hand Kinematics: Distance to Target & Speed', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ===== Row 4: Reward Components =====
    ax6 = fig.add_subplot(gs[3, :])
    
    ax6.plot(time, distance_reward, 'b-', linewidth=1.5, label='Distance Reward', alpha=0.8)
    ax6.plot(time, reach_bonus, 'g-', linewidth=1.5, label='Reach Bonus', alpha=0.8)
    ax6.plot(time, energy_penalty, 'r-', linewidth=1.5, label='Energy Penalty', alpha=0.8)
    ax6.plot(time, rewards, 'k-', linewidth=2, label='Total Reward')
    
    add_phase_shading(ax6, phases, time)
    ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Reward')
    ax6.set_title('Reward Components Over Time', fontweight='bold')
    ax6.legend(loc='upper right', ncol=4)
    ax6.grid(True, alpha=0.3)
    
    # ===== Row 5: Cumulative Reward + Phase Timeline =====
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.plot(time, np.cumsum(rewards), 'k-', linewidth=2)
    add_phase_shading(ax7, phases, time)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Cumulative Reward')
    ax7.set_title('Cumulative Reward', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Final success indicator
    final_phase = phases[-1] if phases else 'unknown'
    success = final_phase == 'done'
    result_text = "SUCCESS" if success else f"Phase: {final_phase}"
    result_color = 'green' if success else 'orange'
    ax7.text(0.98, 0.95, result_text, transform=ax7.transAxes, fontsize=12,
             fontweight='bold', color=result_color, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===== Row 5: Phase Timeline =====
    ax8 = fig.add_subplot(gs[4, 1])
    phase_numeric = [phase_map.get(p, -1) for p in phases]
    ax8.plot(time, phase_numeric, 'k-', linewidth=2, drawstyle='steps-post')
    ax8.fill_between(time, phase_numeric, step='post', alpha=0.3)
    ax8.set_yticks(list(phase_map.values()))
    ax8.set_yticklabels(list(phase_map.keys()))
    ax8.set_xlabel('Time (s)')
    ax8.set_title('Trial Phase Timeline', fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Add phase color legend
    legend_patches = [patches.Patch(color=color, label=phase, alpha=0.5) 
                     for phase, color in phase_colors.items()]
    ax8.legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    plt.suptitle('Episode Summary', fontsize=14, fontweight='bold', y=1.0)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Episode summary saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def inspect_checkpoint(
    checkpoint_path: str,
    xml_path: str,
    sensor_stats: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    n_episodes: int = 1,
    max_steps: int = 300,
    show: bool = True
):
    """
    Comprehensive checkpoint inspection: weights + episode summary.
    
    Args:
        checkpoint_path: Path to controller checkpoint
        xml_path: Path to MuJoCo XML for running episode
        sensor_stats: Sensor normalization stats (loads from checkpoint dir if None)
        output_dir: Directory to save all plots
        n_episodes: Number of episodes to record and analyze
        max_steps: Max steps per episode
        show: Whether to display plots
    
    Usage:
        inspect_checkpoint(
            "outputs/best_controller.pt",
            "arm.xml",
            output_dir="inspection/"
        )
    """
    import pickle
    
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Load controller
    controller, config, checkpoint = load_controller(checkpoint_path)
    
    print(f"{'='*60}")
    print(f"Checkpoint Inspection: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"Controller type: RNN")
    print(f"Parameters: {sum(p.numel() for p in controller.parameters()):,}")
    
    if 'fitness' in checkpoint:
        print(f"Training fitness: {checkpoint['fitness']:.3f}")
    if 'generation' in checkpoint:
        print(f"Training generation: {checkpoint['generation']}")
    
    # Load sensor stats
    if sensor_stats is None:
        stats_path = Path(checkpoint_path).parent / 'sensor_stats.pkl'
        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                sensor_stats = pickle.load(f)
            print(f"Loaded sensor stats from {stats_path}")
        else:
            print("Warning: No sensor stats found, using defaults")
            sensor_stats = {}
    
    # 1. Weight summary
    print("\n" + "="*60)
    print_weight_summary(controller)
    
    # 2. Weight plots
    if output_path:
        print(f"\nGenerating weight plots...")
        plot_all_weights(controller, str(output_path), show=False)
    
    # 3. Record episode(s) and plot summaries
    print(f"\nRecording {n_episodes} episode(s)...")
    
    for ep in range(n_episodes):
        trajectory = record_episode(
            controller=controller,
            xml_path=xml_path,
            sensor_stats=sensor_stats,
            max_steps=max_steps
        )
        
        total_reward = sum(trajectory['rewards'])
        final_phase = trajectory['infos'][-1].get('phase', 'unknown')
        
        print(f"  Episode {ep+1}: {len(trajectory['rewards'])} steps, "
              f"reward={total_reward:.2f}, final_phase={final_phase}")
        
        # Plot episode summary
        summary_path = str(output_path / f'episode_summary_{ep+1}.png') if output_path else None
        plot_episode_summary(trajectory, output_path=summary_path, show=show and ep == 0)
        
        # Plot detailed trajectory
        traj_path = str(output_path / f'trajectory_{ep+1}.png') if output_path else None
        plot_trajectory(trajectory, output_path=traj_path, show=False)
    
    print(f"\n{'='*60}")
    if output_path:
        print(f"All plots saved to: {output_path}")
    print("Inspection complete!")


# Need to import patches for the legend
import matplotlib.patches as patches


if __name__ == '__main__':
    print("Visualization utilities loaded.")
    print("\nAvailable functions:")
    print("\n  Evaluation & Recording:")
    print("    - load_controller(path, type)")
    print("    - evaluate_controller(controller, xml_path, sensor_stats)")
    print("    - record_episode(controller, xml_path, sensor_stats)")
    print("    - save_video(frames, output_path)")
    
    print("\n  Trajectory & Training Plots:")
    print("    - plot_trajectory(trajectory)")
    print("    - plot_training_curves(history_path)")
    print("    - compare_controllers(paths, labels, xml_path, sensor_stats)")
    
    print("\n  Weight Inspection:")
    print("    - get_weight_summary(controller)")
    print("    - print_weight_summary(controller)")
    print("    - plot_weight_distributions(controller)")
    print("    - plot_weight_matrices(controller, layer_names)")
    print("    - plot_reflex_connections(controller)")
    print("    - plot_sensory_weights(controller)")
    print("    - plot_rnn_weights(controller)")
    print("    - plot_all_weights(controller, output_dir)")
    print("    - inspect_controller(checkpoint_path, output_dir)")
