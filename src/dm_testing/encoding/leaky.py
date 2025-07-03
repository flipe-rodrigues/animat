# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from policy import NumpyStyleRNNPolicy

# Set up matplotlib for clean publication-style plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'legend.fontsize': 22,
    'lines.linewidth': 2.5,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})
np.random.seed(42)
torch.manual_seed(42)

# Create policies with different alpha values
obs_dim = 29
action_dim = 4
hidden_size = 64

policy_alpha_01 = NumpyStyleRNNPolicy(obs_dim, action_dim, hidden_size, alpha=0.1)
policy_alpha_10 = NumpyStyleRNNPolicy(obs_dim, action_dim, hidden_size, alpha=1.0)

# Copy weights for fair comparison
with torch.no_grad():
    policy_alpha_10.W_in.copy_(policy_alpha_01.W_in)
    policy_alpha_10.W_h.copy_(policy_alpha_01.W_h)
    policy_alpha_10.W_out.copy_(policy_alpha_01.W_out)
    policy_alpha_10.b_h.copy_(policy_alpha_01.b_h)
    policy_alpha_10.b_out.copy_(policy_alpha_01.b_out)

def plot_oscillatory_response():
    """Clean oscillatory response comparison for report"""
    
    sequence_length = 60
    
    # Create oscillatory input
    osc_input = torch.zeros(1, sequence_length, obs_dim)
    for t in range(sequence_length):
        osc_input[0, t, :] = 0.5 + 0.4 * np.sin(2 * np.pi * t / 12)
    
    # Run both policies
    with torch.no_grad():
        outputs_01, _ = policy_alpha_01(osc_input)
        outputs_10, _ = policy_alpha_10(osc_input)
    
    # Create clean plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot input and responses
    input_trace = osc_input.squeeze().mean(dim=-1).numpy()
    mean_action_01 = outputs_01.squeeze().mean(dim=-1).numpy()
    mean_action_10 = outputs_10.squeeze().mean(dim=-1).numpy()
    
    ax.plot(input_trace, 'k-', linewidth=2, alpha=0.6, label='Input Signal')
    ax.plot(mean_action_01, 'b-', linewidth=3, label='α = 0.1 (Leaky Integration)')
    ax.plot(mean_action_10, 'r-', linewidth=3, label='α = 1.0 (Direct Update)')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Signal Amplitude')
    ax.set_title('Response to Oscillatory Input')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, sequence_length-1)
    
    plt.tight_layout()
    plt.savefig('oscillatory_response_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_oscillatory_response()

print("✅ Clean oscillatory response plot created!")
print("📁 File: oscillatory_response_comparison.png")
