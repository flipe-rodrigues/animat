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

# Test both alpha values
obs_dim = 29
action_dim = 4
hidden_size = 64

# Create policies with different alpha values
policy_alpha_01 = NumpyStyleRNNPolicy(obs_dim, action_dim, hidden_size, alpha=0.1)
policy_alpha_10 = NumpyStyleRNNPolicy(obs_dim, action_dim, hidden_size, alpha=1.0)

# Copy weights to make fair comparison
with torch.no_grad():
    policy_alpha_10.W_in.copy_(policy_alpha_01.W_in)
    policy_alpha_10.W_h.copy_(policy_alpha_01.W_h)
    policy_alpha_10.W_out.copy_(policy_alpha_01.W_out)
    policy_alpha_10.b_h.copy_(policy_alpha_01.b_h)
    policy_alpha_10.b_out.copy_(policy_alpha_01.b_out)

print("Testing α=1.0 vs α=0.1 with identical weights...")

# %%
"""
TEST: ALPHA = 1.0 vs ALPHA = 0.1 COMPARISON
"""
def compare_alpha_effects():
    """Compare behavior between α=0.1 and α=1.0"""
    
    # Test different input patterns
    sequence_length = 50
    
    # Create test inputs
    test_patterns = {
        'Step Input': torch.cat([torch.zeros(1, 25, obs_dim), torch.ones(1, 25, obs_dim) * 0.8], dim=1),
        'Pulse Input': torch.zeros(1, sequence_length, obs_dim),
        'Noisy Input': torch.randn(1, sequence_length, obs_dim) * 0.1 + 0.3,
        'Oscillating': torch.zeros(1, sequence_length, obs_dim)
    }
    
    # Create pulse
    test_patterns['Pulse Input'][0, 20:25, :] = 1.0
    
    # Create oscillation
    for t in range(sequence_length):
        test_patterns['Oscillating'][0, t, :] = 0.5 + 0.4 * np.sin(2 * np.pi * t / 10)
    
    fig, axes = plt.subplots(len(test_patterns), 3, figsize=(18, 12))
    
    for i, (pattern_name, input_seq) in enumerate(test_patterns.items()):
        
        # Run both policies
        with torch.no_grad():
            outputs_01, (h_01, o_01) = policy_alpha_01(input_seq)
            outputs_10, (h_10, o_10) = policy_alpha_10(input_seq)
        
        # Plot input
        axes[i, 0].plot(input_seq.squeeze().mean(dim=-1).numpy(), 'k-', linewidth=2, label='Input')
        axes[i, 0].set_title(f'{pattern_name}')
        axes[i, 0].set_ylabel('Input magnitude')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()
        
        # Plot outputs comparison
        for j in range(action_dim):
            axes[i, 1].plot(outputs_01.squeeze()[:, j].numpy(), '--', alpha=0.7, label=f'α=0.1, Act{j}')
            axes[i, 1].plot(outputs_10.squeeze()[:, j].numpy(), '-', linewidth=2, label=f'α=1.0, Act{j}')
        
        axes[i, 1].set_title('Output Actions Comparison')
        axes[i, 1].set_ylabel('Action value')
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:  # Only show legend for first row
            axes[i, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot hidden state magnitudes
        h_mag_01 = torch.norm(h_01.squeeze(), dim=-1).numpy()
        h_mag_10 = torch.norm(h_10.squeeze(), dim=-1).numpy()
        
        axes[i, 2].plot(h_mag_01, '--', linewidth=2, label='α=0.1 (smooth)', color='blue')
        axes[i, 2].plot(h_mag_10, '-', linewidth=2, label='α=1.0 (direct)', color='red')
        axes[i, 2].set_title('Hidden State Magnitude')
        axes[i, 2].set_ylabel('||h||')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].legend()
        
        if i == len(test_patterns) - 1:
            axes[i, 0].set_xlabel('Time step')
            axes[i, 1].set_xlabel('Time step')
            axes[i, 2].set_xlabel('Time step')
    
    plt.suptitle('α=1.0 vs α=0.1: Behavior Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('alpha_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_alpha_effects()

# %%
"""
TEST: MEMORY RETENTION WITH ALPHA = 1.0
"""
def test_memory_loss_alpha_1():
    """Test what happens to memory when α=1.0"""
    
    sequence_length = 30
    
    # Create input with early strong signal, then silence
    input_seq = torch.zeros(1, sequence_length, obs_dim)
    input_seq[0, 5:10, :] = 1.0  # Strong signal from step 5-10
    
    # Manually step through to track states
    states_01 = []
    states_10 = []
    
    # Initialize states
    h_01 = torch.zeros(1, hidden_size)
    o_01 = torch.zeros(1, action_dim)
    h_10 = torch.zeros(1, hidden_size)
    o_10 = torch.zeros(1, action_dim)
    
    with torch.no_grad():
        for t in range(sequence_length):
            x_t = input_seq[0, t:t+1, :]
            
            # α=0.1 update
            pre_h_01 = x_t @ policy_alpha_01.W_in.t() + h_01 @ policy_alpha_01.W_h.t() + policy_alpha_01.b_h
            h_01 = (1-0.1)*h_01 + 0.1*torch.sigmoid(pre_h_01)
            
            pre_o_01 = h_01 @ policy_alpha_01.W_out.t() + policy_alpha_01.b_out
            o_01 = (1-0.1)*o_01 + 0.1*torch.sigmoid(pre_o_01)
            
            # α=1.0 update
            pre_h_10 = x_t @ policy_alpha_10.W_in.t() + h_10 @ policy_alpha_10.W_h.t() + policy_alpha_10.b_h
            h_10 = (1-1.0)*h_10 + 1.0*torch.sigmoid(pre_h_10)  # = 0*h_10 + 1.0*new_h
            
            pre_o_10 = h_10 @ policy_alpha_10.W_out.t() + policy_alpha_10.b_out
            o_10 = (1-1.0)*o_10 + 1.0*torch.sigmoid(pre_o_10)  # = 0*o_10 + 1.0*new_o
            
            states_01.append({
                'h_mag': torch.norm(h_01).item(),
                'o_mag': torch.norm(o_01).item(),
                'h': h_01.clone(),
                'o': o_01.clone()
            })
            
            states_10.append({
                'h_mag': torch.norm(h_10).item(),
                'o_mag': torch.norm(o_10).item(),
                'h': h_10.clone(),
                'o': o_10.clone()
            })
    
    # Plot memory retention analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Hidden state magnitude
    h_mags_01 = [s['h_mag'] for s in states_01]
    h_mags_10 = [s['h_mag'] for s in states_10]
    
    axes[0, 0].plot(h_mags_01, 'b-', linewidth=2, label='α=0.1 (retains memory)')
    axes[0, 0].plot(h_mags_10, 'r-', linewidth=2, label='α=1.0 (no memory)')
    axes[0, 0].axvspan(5, 10, alpha=0.3, color='gray', label='Input active')
    axes[0, 0].axvline(x=10, color='black', linestyle='--', alpha=0.7, label='Input stops')
    axes[0, 0].set_title('Hidden State Magnitude')
    axes[0, 0].set_ylabel('||h||')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Output state magnitude
    o_mags_01 = [s['o_mag'] for s in states_01]
    o_mags_10 = [s['o_mag'] for s in states_10]
    
    axes[0, 1].plot(o_mags_01, 'b-', linewidth=2, label='α=0.1 (smooth decay)')
    axes[0, 1].plot(o_mags_10, 'r-', linewidth=2, label='α=1.0 (instant change)')
    axes[0, 1].axvspan(5, 10, alpha=0.3, color='gray')
    axes[0, 1].axvline(x=10, color='black', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Output State Magnitude')
    axes[0, 1].set_ylabel('||o||')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory decay analysis (post input)
    post_input_steps = range(10, sequence_length)
    h_decay_01 = [h_mags_01[i] / h_mags_01[10] for i in post_input_steps]  # Normalized to value at step 10
    h_decay_10 = [h_mags_10[i] / max(h_mags_10[10], 1e-8) for i in post_input_steps]  # Avoid div by zero
    
    axes[1, 0].plot(post_input_steps, h_decay_01, 'b-', linewidth=2, label='α=0.1')
    axes[1, 0].plot(post_input_steps, h_decay_10, 'r-', linewidth=2, label='α=1.0')
    axes[1, 0].set_title('Memory Decay (Normalized)')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Relative magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # State change rates
    h_changes_01 = [abs(h_mags_01[i] - h_mags_01[i-1]) for i in range(1, len(h_mags_01))]
    h_changes_10 = [abs(h_mags_10[i] - h_mags_10[i-1]) for i in range(1, len(h_mags_10))]
    
    axes[1, 1].plot(h_changes_01, 'b-', linewidth=2, label='α=0.1 (gradual)')
    axes[1, 1].plot(h_changes_10, 'r-', linewidth=2, label='α=1.0 (abrupt)')
    axes[1, 1].axvspan(5, 10, alpha=0.3, color='gray')
    axes[1, 1].set_title('State Change Rate')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('|Δ||h|||')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Memory Analysis: α=1.0 vs α=0.1', fontsize=16)
    plt.tight_layout()
    plt.savefig('memory_analysis_alpha_1.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    print("\n🔍 Memory Analysis Results:")
    print(f"📊 Hidden state after input stops (step 10):")
    print(f"   α=0.1: {h_mags_01[10]:.4f}")
    print(f"   α=1.0: {h_mags_10[10]:.4f}")
    
    print(f"📊 Hidden state 10 steps later (step 20):")
    print(f"   α=0.1: {h_mags_01[20]:.4f} (retains {h_mags_01[20]/h_mags_01[10]*100:.1f}%)")
    print(f"   α=1.0: {h_mags_10[20]:.4f} (retains {h_mags_10[20]/max(h_mags_10[10], 1e-8)*100:.1f}%)")
    
    print(f"📊 Maximum state change in one step:")
    print(f"   α=0.1: {max(h_changes_01):.4f}")
    print(f"   α=1.0: {max(h_changes_10):.4f}")

test_memory_loss_alpha_1()

print("\n✅ Alpha comparison analysis complete!")
print("📁 Generated files:")
print("   - alpha_comparison.png")
print("   - memory_analysis_alpha_1.png")
print("\n🧠 KEY FINDINGS:")
print("   α=1.0 → NO MEMORY: h ← 0·h + 1·new_h = new_h")
print("   α=0.1 → MEMORY: h ← 0.9·h + 0.1·new_h (90% retention)")
print("   α=1.0 makes the RNN effectively feedforward!")