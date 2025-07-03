import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tianshou.data import Batch
import pandas as pd
import seaborn as sns
from shimmy_wrapper import create_env

def test_rnn_dynamics(policy, save_dir="rnn_analysis"):
    """
    Advanced test of RNN temporal dynamics using multiple input patterns.
    Tests how the RNN responds to different temporal input patterns.
    
    Args:
        policy: Policy with RNN to test
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    steps = 100
    
    # Define different input patterns
    input_patterns = {
        "impulse": create_impulse_input(steps, obs_dim),
        "sinusoidal": create_sinusoidal_input(steps, obs_dim),
        "ramps": create_ramp_input(steps, obs_dim),
        "multi_dim": create_multidim_input(steps, obs_dim),
        "sequence": create_sequence_input(steps, obs_dim),
        "realistic": create_realistic_input(env, steps)
    }
    
    results = {}
    
    # Process each input pattern
    for pattern_name, obs_seq in input_patterns.items():
        print(f"Processing {pattern_name} pattern...")
        state = None
        outputs = []
        hiddens = []
        
        for t in range(steps):
            batch = Batch(obs=obs_seq[t:t+1], info={})
            with torch.no_grad():
                result = policy.forward(batch, state=state)
                action = result.act
                state = result.state
                outputs.append(action)
                
                # Store hidden state
                if state is not None:
                    if isinstance(state, dict) and 'hidden' in state:
                        hiddens.append(state['hidden'].detach().cpu())
                    elif isinstance(state, torch.Tensor):
                        hiddens.append(state.detach().cpu())
        
        # Store results
        actions = torch.cat(outputs, dim=0)
        hidden_array = torch.cat(hiddens, dim=0)
        if len(hidden_array.shape) == 3:  # [steps, batch=1, hidden_size]
            hidden_array = hidden_array.squeeze(1)
        
        results[pattern_name] = {
            "input": obs_seq,
            "actions": actions,
            "hidden": hidden_array
        }
        
        # Calculate neuron activation metrics
        activation_metrics = {
            "mean": hidden_array.mean().item(),
            "std": hidden_array.std().item(),
            "min": hidden_array.min().item(),
            "max": hidden_array.max().item(),
            "saturation_pct": ((hidden_array.abs() > 0.9).float().mean() * 100).item(),
            "activation_range": hidden_array.max().item() - hidden_array.min().item()
        }
        
        print(f"  {pattern_name} activation stats:")
        print(f"    Mean: {activation_metrics['mean']:.4f}")
        print(f"    Std: {activation_metrics['std']:.4f}")
        print(f"    Min: {activation_metrics['min']:.4f}")
        print(f"    Max: {activation_metrics['max']:.4f}")
        print(f"    Saturation: {activation_metrics['saturation_pct']:.1f}%")
        print(f"    Range: {activation_metrics['activation_range']:.4f}")
    
    # Create comparative visualization
    visualize_rnn_responses(results, save_dir)
    
    # Create activation distribution comparison
    compare_activation_distributions(results, save_dir)
    
    return results


def create_impulse_input(steps, obs_dim):
    """Traditional impulse at the beginning."""
    obs_seq = torch.zeros((steps, obs_dim))
    obs_seq[0, 0] = 1.0
    return obs_seq


def create_sinusoidal_input(steps, obs_dim):
    """Sinusoidal inputs of different frequencies."""
    obs_seq = torch.zeros((steps, obs_dim))
    frequencies = [0.05, 0.1, 0.2]
    for i, freq in enumerate(frequencies):
        if i < min(3, obs_dim):
            obs_seq[:, i] = 0.5 * torch.sin(torch.linspace(0, freq*2*np.pi*steps/20, steps))
    return obs_seq


def create_ramp_input(steps, obs_dim):
    """Gradual ramps up and down."""
    obs_seq = torch.zeros((steps, obs_dim))
    half_steps = steps // 2
    obs_seq[:half_steps, 0] = torch.linspace(0, 1, half_steps)  # Ramp up
    obs_seq[half_steps:, 1] = torch.linspace(1, 0, steps-half_steps)  # Ramp down
    return obs_seq


def create_multidim_input(steps, obs_dim):
    """Multiple dimensions active at different times."""
    obs_seq = torch.zeros((steps, obs_dim))
    segment = steps // 5
    for i in range(min(5, obs_dim)):
        start = i * segment
        end = min((i+1) * segment, steps)
        obs_seq[start:end, i] = 0.8
    return obs_seq


def create_sequence_input(steps, obs_dim):
    """Sequence of pulses in different dimensions."""
    obs_seq = torch.zeros((steps, obs_dim))
    pulse_width = 3
    pulse_interval = steps // min(8, obs_dim)
    for i in range(min(8, obs_dim)):
        pulse_start = i * pulse_interval
        obs_seq[pulse_start:pulse_start+pulse_width, i % min(5, obs_dim)] = 1.0
    return obs_seq


def create_realistic_input(env, steps):
    """Record a realistic trajectory from the environment."""
    obs_seq = torch.zeros((steps, env.observation_space.shape[0]))
    obs, _ = env.reset()
    
    # Use random actions to generate a trajectory
    for i in range(steps):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        obs_seq[i] = torch.tensor(obs, dtype=torch.float32)
    
    # Normalize
    obs_mean = obs_seq.mean(dim=0, keepdim=True)
    obs_std = obs_seq.std(dim=0, keepdim=True) + 1e-8
    obs_seq = (obs_seq - obs_mean) / obs_std
    
    return obs_seq


def visualize_rnn_responses(results, save_dir):
    """Create comparative visualization of RNN responses to different inputs."""
    n_patterns = len(results)
    fig = plt.figure(figsize=(20, 5 * n_patterns))
    
    for i, (pattern_name, data) in enumerate(results.items()):
        # First row for each pattern: Input
        plt.subplot(n_patterns, 3, i*3 + 1)
        plt.title(f"{pattern_name.capitalize()} Input")
        # Show first 5 dimensions only for clarity
        for j in range(min(5, data["input"].shape[1])):
            plt.plot(data["input"][:, j].numpy(), label=f"Dim {j}")
        plt.xlabel("Timestep")
        plt.ylabel("Input Value")
        plt.legend()
        
        # Second row: Actions
        plt.subplot(n_patterns, 3, i*3 + 2)
        plt.title(f"Actions ({pattern_name})")
        for j in range(data["actions"].shape[1]):
            plt.plot(data["actions"][:, j].numpy(), label=f"Action {j}")
        plt.xlabel("Timestep")
        plt.ylabel("Action Value")
        plt.legend()
        
        # Third row: Hidden states
        plt.subplot(n_patterns, 3, i*3 + 3)
        plt.title(f"Hidden States ({pattern_name})")
        plt.imshow(data["hidden"].numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(label="Activation")
        plt.xlabel("Hidden Unit")
        plt.ylabel("Timestep")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rnn_responses_comparison.png"))
    plt.close()
    
    # Also create individual detailed plots
    for pattern_name, data in results.items():
        plt.figure(figsize=(15, 12))
        
        # Input 
        plt.subplot(3, 1, 1)
        plt.title(f"{pattern_name.capitalize()} Input")
        for j in range(min(5, data["input"].shape[1])):
            plt.plot(data["input"][:, j].numpy(), label=f"Dim {j}")
        plt.xlabel("Timestep")
        plt.ylabel("Input Value")
        plt.legend()
        
        # Actions
        plt.subplot(3, 1, 2)
        plt.title("Output Actions")
        for j in range(data["actions"].shape[1]):
            plt.plot(data["actions"][:, j].numpy(), label=f"Action {j}")
        plt.xlabel("Timestep")
        plt.ylabel("Action Value")
        plt.legend()
        
        # Hidden states
        plt.subplot(3, 1, 3)
        plt.title("Hidden State Evolution")
        plt.imshow(data["hidden"].numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(label="Activation")
        plt.xlabel("Hidden Unit")
        plt.ylabel("Timestep")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{pattern_name}_detailed.png"))
        plt.close()


def compare_activation_distributions(results, save_dir):
    """Compare activation value distributions across input patterns."""
    plt.figure(figsize=(15, 8))
    
    # Create dataset for seaborn
    all_data = []
    for pattern_name, data in results.items():
        # Flatten hidden activations to 1D
        hidden_flat = data["hidden"].numpy().flatten()
        # Sample to avoid too many points
        sample_size = min(10000, len(hidden_flat))
        sample_idx = np.random.choice(len(hidden_flat), sample_size, replace=False)
        hidden_sample = hidden_flat[sample_idx]
        
        # Add to dataframe
        pattern_data = pd.DataFrame({
            'activation': hidden_sample,
            'pattern': [pattern_name] * sample_size
        })
        all_data.append(pattern_data)
    
    activation_df = pd.concat(all_data, ignore_index=True)
    
    # Create distribution plot
    plt.subplot(2, 1, 1)
    sns.histplot(data=activation_df, x='activation', hue='pattern', kde=True, bins=50)
    plt.title("Hidden Activation Distributions")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    
    # Create violin plot
    plt.subplot(2, 1, 2)
    sns.violinplot(data=activation_df, x='pattern', y='activation')
    plt.title("Activation Distribution by Input Pattern")
    plt.xlabel("Input Pattern")
    plt.ylabel("Activation Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_distributions.png"))
    plt.close()
    
    # Calculate saturation levels
    saturation_data = []
    for pattern_name, data in results.items():
        hidden = data["hidden"].numpy()
        # Calculate saturation levels
        highly_positive = (hidden > 0.9).mean() * 100
        highly_negative = (hidden < -0.9).mean() * 100
        moderate = ((hidden >= -0.5) & (hidden <= 0.5)).mean() * 100
        
        saturation_data.append({
            'pattern': pattern_name,
            'high_positive': highly_positive,
            'high_negative': highly_negative,
            'moderate': moderate
        })
    
    saturation_df = pd.DataFrame(saturation_data)
    
    # Plot saturation levels
    plt.figure(figsize=(12, 6))
    saturation_melted = pd.melt(
        saturation_df, 
        id_vars=['pattern'], 
        value_vars=['high_positive', 'high_negative', 'moderate'],
        var_name='activation_type',
        value_name='percentage'
    )
    
    sns.barplot(data=saturation_melted, x='pattern', y='percentage', hue='activation_type')
    plt.title("Activation Saturation by Pattern")
    plt.xlabel("Input Pattern")
    plt.ylabel("Percentage of Activations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_saturation.png"))
    plt.close()


if __name__ == "__main__":
    import torch
    from debug_training import make_ppo_policy
    
    # Create environment and policy
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    policy = make_ppo_policy(
        obs_dim=obs_dim, 
        action_dim=action_dim,
        action_space=env.action_space,
        hidden_size=128,
        num_layers=1,
        tau_mem=50.0,
        debug=True
    )
    
    # Load a trained policy if available
    try:
        policy.load_state_dict(torch.load("log/latest/policy_final.pth"))
        print("Loaded trained policy")
    except:
        print("Using untrained policy")
    
    # Run the test
    results = test_rnn_dynamics(policy, "rnn_analysis_results")
    print("Analysis complete - check the rnn_analysis_results directory")