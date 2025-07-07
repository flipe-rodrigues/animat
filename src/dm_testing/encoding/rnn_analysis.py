import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from policy import NumpyStyleRNNPolicy

# Set up matplotlib
plt.style.use('seaborn-v0_8')
np.random.seed(42)
torch.manual_seed(42)

# Initialize the RNN policy
obs_dim = 29  # 12 muscle + 16 grid + 1 z
action_dim = 4
hidden_size = 25
alpha = 0.1

policy = NumpyStyleRNNPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_size=hidden_size,
    alpha=alpha
)

print(f"RNN Policy Configuration:")
print(f"  Observation dimension: {obs_dim}")
print(f"  Action dimension: {action_dim}")
print(f"  Hidden size: {hidden_size}")
print(f"  Integration rate (α): {alpha}")
print(f"  Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

# %%
"""
TEST 1: MEMORY DYNAMICS - How alpha affects memory retention
"""
def test_memory_dynamics():
    """Test how the leaky integration affects memory over time"""
    
    # Create test with different alpha values
    alphas = [0.01, 0.1, 0.3, 0.5, 1.0]
    sequence_length = 50
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, alpha_test in enumerate(alphas):
        if i >= len(axes):
            break
            
        # Create policy with test alpha
        test_policy = NumpyStyleRNNPolicy(obs_dim, action_dim, hidden_size, alpha=alpha_test)
        
        # Create input sequence: strong signal for first 10 steps, then zeros
        obs_sequence = torch.zeros(1, sequence_length, obs_dim)
        obs_sequence[0, :10, :] = 1.0  # Strong input for first 10 steps
        
        # Run forward pass
        with torch.no_grad():
            outputs, (final_h, final_o) = test_policy(obs_sequence)
        
        # Plot hidden state magnitude over time
        hidden_magnitude = torch.norm(final_h.squeeze(), dim=-1)
        output_magnitude = torch.norm(outputs.squeeze(), dim=-1)
        
        axes[i].plot(hidden_magnitude.numpy(), label='Hidden state', linewidth=2)
        axes[i].plot(output_magnitude.numpy(), label='Output state', linewidth=2)
        axes[i].axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Input stops')
        axes[i].set_title(f'α = {alpha_test}')
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('State magnitude')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(alphas) < len(axes):
        axes[-1].remove()
    
    plt.suptitle('Memory Dynamics: Effect of Integration Rate (α)', fontsize=16)
    plt.tight_layout()
    plt.savefig('rnn_memory_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()

test_memory_dynamics()

# %%
"""
TEST 2: RESPONSE TO DIFFERENT INPUT PATTERNS
"""
def test_input_responses():
    """Test how RNN responds to different input patterns"""
    
    sequence_length = 100
    patterns = {
        'Constant': torch.ones(1, sequence_length, obs_dim) * 0.5,
        'Step': torch.cat([torch.zeros(1, 50, obs_dim), torch.ones(1, 50, obs_dim)], dim=1),
        'Pulse': torch.zeros(1, sequence_length, obs_dim),
        'Sine': torch.zeros(1, sequence_length, obs_dim),
        'Noise': torch.randn(1, sequence_length, obs_dim) * 0.1
    }
    
    # Create specific patterns
    patterns['Pulse'][0, 45:55, :] = 1.0  # Pulse from step 45-55
    for t in range(sequence_length):
        patterns['Sine'][0, t, :] = 0.5 + 0.3 * np.sin(2 * np.pi * t / 20)
    
    fig, axes = plt.subplots(len(patterns), 2, figsize=(12, 12))
    
    for i, (pattern_name, input_seq) in enumerate(patterns.items()):
        with torch.no_grad():
            outputs, (final_h, final_o) = policy(input_seq)
        
        # Plot input pattern
        axes[i, 0].plot(input_seq.squeeze().mean(dim=-1).numpy(), 'b-', linewidth=2)
        axes[i, 0].set_title(f'{pattern_name} Input')
        axes[i, 0].set_ylabel('Input magnitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot output response
        axes[i, 1].plot(outputs.squeeze().numpy())
        axes[i, 1].set_title(f'{pattern_name} Output')
        axes[i, 1].set_ylabel('Action values')
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == len(patterns) - 1:
            axes[i, 0].set_xlabel('Time step')
            axes[i, 1].set_xlabel('Time step')
    
    plt.suptitle('RNN Response to Different Input Patterns', fontsize=16)
    plt.tight_layout()
    plt.savefig('rnn_input_responses.png', dpi=150, bbox_inches='tight')
    plt.show()

test_input_responses()

# %%
"""
TEST 3: WEIGHT ANALYSIS
"""
def analyze_weights():
    """Analyze the structure and properties of learned weights"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Get weight matrices
    W_in = policy.W_in.data.numpy()    # (64, 29)
    W_h = policy.W_h.data.numpy()      # (64, 64)
    W_out = policy.W_out.data.numpy()  # (4, 64)
    b_h = policy.b_h.data.numpy()      # (64,)
    b_out = policy.b_out.data.numpy()  # (4,)
    
    # Plot 1: Input weights heatmap
    sns.heatmap(W_in, cmap='coolwarm', center=0, ax=axes[0, 0], cbar=True)
    axes[0, 0].set_title('Input Weights (W_in)')
    axes[0, 0].set_xlabel('Input dimension')
    axes[0, 0].set_ylabel('Hidden unit')
    
    # Plot 2: Recurrent weights heatmap
    sns.heatmap(W_h, cmap='coolwarm', center=0, ax=axes[0, 1], cbar=True)
    axes[0, 1].set_title('Recurrent Weights (W_h)')
    axes[0, 1].set_xlabel('Hidden unit (from)')
    axes[0, 1].set_ylabel('Hidden unit (to)')
    
    # Plot 3: Output weights heatmap
    sns.heatmap(W_out, cmap='coolwarm', center=0, ax=axes[0, 2], cbar=True)
    axes[0, 2].set_title('Output Weights (W_out)')
    axes[0, 2].set_xlabel('Hidden unit')
    axes[0, 2].set_ylabel('Action dimension')
    
    # Plot 4: Weight distributions
    axes[1, 0].hist(W_in.flatten(), bins=50, alpha=0.7, color='blue', label='W_in')
    axes[1, 0].hist(W_h.flatten(), bins=50, alpha=0.7, color='green', label='W_h')
    axes[1, 0].hist(W_out.flatten(), bins=50, alpha=0.7, color='red', label='W_out')
    axes[1, 0].set_xlabel('Weight value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Weight Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Bias values
    axes[1, 1].bar(range(len(b_h)), b_h, alpha=0.7, color='blue', label='Hidden bias')
    axes[1, 1].set_xlabel('Hidden unit')
    axes[1, 1].set_ylabel('Bias value')
    axes[1, 1].set_title('Hidden Biases')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Output bias values
    axes[1, 2].bar(range(len(b_out)), b_out, alpha=0.7, color='red')
    axes[1, 2].set_xlabel('Action dimension')
    axes[1, 2].set_ylabel('Bias value')
    axes[1, 2].set_title('Output Biases')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('RNN Weight Structure Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('rnn_weight_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print weight statistics
    print(f"\nWeight Statistics:")
    print(f"  W_in: mean={W_in.mean():.4f}, std={W_in.std():.4f}, range=[{W_in.min():.4f}, {W_in.max():.4f}]")
    print(f"  W_h:  mean={W_h.mean():.4f}, std={W_h.std():.4f}, range=[{W_h.min():.4f}, {W_h.max():.4f}]")
    print(f"  W_out: mean={W_out.mean():.4f}, std={W_out.std():.4f}, range=[{W_out.min():.4f}, {W_out.max():.4f}]")

analyze_weights()

# %%
"""
TEST 4: STATE EVOLUTION ANALYSIS
"""
def analyze_state_evolution():
    """Analyze how hidden and output states evolve over time"""
    
    # Create a test sequence with changing inputs
    sequence_length = 100
    obs_sequence = torch.zeros(1, sequence_length, obs_dim)
    
    # Different phases of input
    obs_sequence[0, 0:20, :5] = 1.0      # Muscle sensors active
    obs_sequence[0, 20:40, 12:20] = 1.0  # Grid encoding active
    obs_sequence[0, 40:60, 28] = 1.0     # Target Z active
    obs_sequence[0, 60:80, :] = 0.5      # All moderate
    # 80-100: all zeros (rest)
    
    # Run through the network, collecting all intermediate states
    hidden_states = []
    output_states = []
    actions = []
    
    h = torch.zeros(1, hidden_size)
    o = torch.zeros(1, action_dim)
    
    with torch.no_grad():
        for t in range(sequence_length):
            x_t = obs_sequence[0, t:t+1, :]
            
            # Manual forward pass to collect states
            pre_h = x_t @ policy.W_in.t() + h @ policy.W_h.t() + policy.b_h
            h = (1-policy.alpha)*h + policy.alpha*policy.activation(pre_h)
            
            pre_o = h @ policy.W_out.t() + policy.b_out
            o = (1-policy.alpha)*o + policy.alpha*torch.sigmoid(pre_o)
            
            hidden_states.append(h.clone())
            output_states.append(o.clone())
            actions.append(o.clone())
    
    hidden_states = torch.stack(hidden_states).squeeze()  # [T, hidden_size]
    output_states = torch.stack(output_states).squeeze()  # [T, action_dim]
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot 1: Input phases
    input_means = obs_sequence.squeeze().mean(dim=-1).numpy()
    axes[0, 0].plot(input_means, linewidth=2)
    axes[0, 0].axvspan(0, 20, alpha=0.3, color='red', label='Muscle phase')
    axes[0, 0].axvspan(20, 40, alpha=0.3, color='green', label='Grid phase')
    axes[0, 0].axvspan(40, 60, alpha=0.3, color='blue', label='Target Z phase')
    axes[0, 0].axvspan(60, 80, alpha=0.3, color='orange', label='Mixed phase')
    axes[0, 0].axvspan(80, 100, alpha=0.3, color='gray', label='Rest phase')
    axes[0, 0].set_title('Input Phases')
    axes[0, 0].set_ylabel('Input magnitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hidden state evolution (PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    hidden_pca = pca.fit_transform(hidden_states.numpy())
    
    axes[0, 1].plot(hidden_pca[:, 0], label='PC1', linewidth=2)
    axes[0, 1].plot(hidden_pca[:, 1], label='PC2', linewidth=2)
    axes[0, 1].plot(hidden_pca[:, 2], label='PC3', linewidth=2)
    axes[0, 1].set_title('Hidden State Evolution (PCA)')
    axes[0, 1].set_ylabel('PC value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Hidden state magnitude
    hidden_magnitude = torch.norm(hidden_states, dim=1).numpy()
    axes[1, 0].plot(hidden_magnitude, linewidth=2, color='purple')
    axes[1, 0].set_title('Hidden State Magnitude')
    axes[1, 0].set_ylabel('||h||')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Output actions
    for i in range(action_dim):
        axes[1, 1].plot(output_states[:, i].numpy(), label=f'Action {i}', linewidth=2)
    axes[1, 1].set_title('Output Actions')
    axes[1, 1].set_ylabel('Action value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: State change rates
    hidden_changes = torch.diff(hidden_states, dim=0).norm(dim=1).numpy()
    output_changes = torch.diff(output_states, dim=0).norm(dim=1).numpy()
    
    axes[2, 0].plot(hidden_changes, label='Hidden Δ', linewidth=2)
    axes[2, 0].plot(output_changes, label='Output Δ', linewidth=2)
    axes[2, 0].set_title('State Change Rates')
    axes[2, 0].set_ylabel('||Δstate||')
    axes[2, 0].set_xlabel('Time step')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Hidden state heatmap
    im = axes[2, 1].imshow(hidden_states.T.numpy(), aspect='auto', cmap='coolwarm', interpolation='nearest')
    axes[2, 1].set_title('Hidden State Heatmap')
    axes[2, 1].set_ylabel('Hidden unit')
    axes[2, 1].set_xlabel('Time step')
    plt.colorbar(im, ax=axes[2, 1])
    
    plt.suptitle('RNN State Evolution Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('rnn_state_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nState Evolution Statistics:")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_[:3]}")
    print(f"  Hidden state range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
    print(f"  Output state range: [{output_states.min():.4f}, {output_states.max():.4f}]")
    print(f"  Average hidden change: {hidden_changes.mean():.6f}")
    print(f"  Average output change: {output_changes.mean():.6f}")

analyze_state_evolution()

# %%
"""
TEST 5: STABILITY ANALYSIS
"""
def test_stability():
    """Test the stability properties of the RNN"""
    
    # Test 1: Fixed point analysis
    fixed_input = torch.ones(1, 1, obs_dim) * 0.5
    
    states_over_time = []
    h = torch.zeros(1, hidden_size)
    o = torch.zeros(1, action_dim)
    
    with torch.no_grad():
        for _ in range(200):  # Run for many steps with same input
            x_t = fixed_input.squeeze(1)
            pre_h = x_t @ policy.W_in.t() + h @ policy.W_h.t() + policy.b_h
            h = (1-policy.alpha)*h + policy.alpha*policy.activation(pre_h)
            
            pre_o = h @ policy.W_out.t() + policy.b_out
            o = (1-policy.alpha)*o + policy.alpha*torch.sigmoid(pre_o)
            
            states_over_time.append(torch.cat([h, o], dim=1).clone())
    
    states_tensor = torch.stack(states_over_time).squeeze()
    
    # Plot convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # State magnitude over time
    state_magnitudes = torch.norm(states_tensor, dim=1).numpy()
    axes[0].plot(state_magnitudes, linewidth=2)
    axes[0].set_title('State Magnitude Convergence')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('||state||')
    axes[0].grid(True, alpha=0.3)
    
    # State change over time
    state_changes = torch.diff(states_tensor, dim=0).norm(dim=1).numpy()
    axes[1].semilogy(state_changes, linewidth=2)
    axes[1].set_title('State Change Rate (Log Scale)')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('||Δstate||')
    axes[1].grid(True, alpha=0.3)
    
    # Final state analysis
    final_h = h.squeeze().numpy()
    final_o = o.squeeze().numpy()
    
    # Show final action values
    axes[2].bar(range(action_dim), final_o, alpha=0.7, color='red')
    axes[2].set_title('Converged Actions')
    axes[2].set_xlabel('Action dimension')
    axes[2].set_ylabel('Action value')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('RNN Stability Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('rnn_stability.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print convergence info
    convergence_threshold = 1e-6
    converged_at = np.where(state_changes < convergence_threshold)[0]
    if len(converged_at) > 0:
        print(f"\nConverged after {converged_at[0]} steps (threshold: {convergence_threshold})")
    else:
        print(f"\nDid not converge within 200 steps (final change: {state_changes[-1]:.2e})")
    
    print(f"Final state magnitude: {state_magnitudes[-1]:.6f}")
    print(f"Final actions: {final_o}")

test_stability()

print("\n✅ RNN analysis complete!")
print("📁 Generated files:")
print("   - rnn_memory_dynamics.png")
print("   - rnn_input_responses.png") 
print("   - rnn_weight_analysis.png")
print("   - rnn_state_evolution.png")
print("   - rnn_stability.png")