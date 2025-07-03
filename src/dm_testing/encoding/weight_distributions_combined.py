import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from rnn_adapter import RNNAdapter, config

# Setup output directory
config.setup_output_dir()

# Load model using config
best_rnn = RNNAdapter()

# Extract weights
weights_input = best_rnn.W_in      # (64, 38)
weights_hidden = best_rnn.W_h      # (64, 64)  
weights_output = best_rnn.W_out    # (4, 64)

print(f"✅ Extracted model weights:")
print(f"  Input weights shape: {weights_input.shape}")
print(f"  Hidden weights shape: {weights_hidden.shape}")
print(f"  Output weights shape: {weights_output.shape}")

# Calculate total absolute weights for each hidden unit
total_abs_input_weights = np.sum(np.abs(weights_input), axis=1)    # Sum over 38 inputs → (64,)
total_abs_output_weights = np.sum(np.abs(weights_output), axis=0)  # Sum over 4 muscles → (64,)
total_abs_hidden_weights = np.sum(np.abs(weights_hidden), axis=1)  # Sum over 64 hidden → (64,)

print(f"\n📊 Weight Statistics:")
print(f"  Input weights  - Mean: {np.mean(total_abs_input_weights):.3f}, Std: {np.std(total_abs_input_weights):.3f}")
print(f"  Output weights - Mean: {np.mean(total_abs_output_weights):.3f}, Std: {np.std(total_abs_output_weights):.3f}")
print(f"  Hidden weights - Mean: {np.mean(total_abs_hidden_weights):.3f}, Std: {np.std(total_abs_hidden_weights):.3f}")

# PLOT 1: LINE PLOT
plt.figure(figsize=(12, 8))

# Create density plots for each weight type
x_input = np.linspace(total_abs_input_weights.min(), total_abs_input_weights.max(), 100)
x_output = np.linspace(total_abs_output_weights.min(), total_abs_output_weights.max(), 100)
x_hidden = np.linspace(total_abs_hidden_weights.min(), total_abs_hidden_weights.max(), 100)

# Calculate kernel density estimates
kde_input = stats.gaussian_kde(total_abs_input_weights)
kde_output = stats.gaussian_kde(total_abs_output_weights)
kde_hidden = stats.gaussian_kde(total_abs_hidden_weights)

# Plot the density curves
plt.plot(x_input, kde_input(x_input), linewidth=3, label='Input Weights', color='green', alpha=0.8)
plt.plot(x_output, kde_output(x_output), linewidth=3, label='Output Weights', color='blue', alpha=0.8)
plt.plot(x_hidden, kde_hidden(x_hidden), linewidth=3, label='Hidden Weights', color='orange', alpha=0.8)

# Styling
plt.xlabel('Total Absolute Weight Magnitude', fontsize=16, fontweight='bold')
plt.ylabel('Density', fontsize=16, fontweight='bold')
plt.title('Distribution of Total Absolute Weights for Each Hidden Unit', fontsize=18, fontweight='bold', pad=20)
plt.legend(fontsize=14, loc='upper right')
plt.grid(True, alpha=0.3)

# Make tick labels bigger
plt.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# Save line plot
plt.savefig(config.OUTPUT_DIR + '/weight_distributions_lines.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Line plot saved to: {config.OUTPUT_DIR}/weight_distributions_lines.png")

# PLOT 2: BAR PLOT
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Input weights histogram
axes[0].hist(total_abs_input_weights, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[0].set_title('Input Weights Distribution', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Total Absolute Weight Magnitude', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='both', which='major', labelsize=12)

# Output weights histogram
axes[1].hist(total_abs_output_weights, bins=20, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_title('Output Weights Distribution', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Total Absolute Weight Magnitude', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# Hidden weights histogram
axes[2].hist(total_abs_hidden_weights, bins=20, color='orange', alpha=0.7, edgecolor='black')
axes[2].set_title('Hidden Weights Distribution', fontsize=16, fontweight='bold')
axes[2].set_xlabel('Total Absolute Weight Magnitude', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Frequency', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='both', which='major', labelsize=12)

# Overall title
fig.suptitle('Weight Distributions by Type', fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()

# Save bar plot
plt.savefig(config.OUTPUT_DIR + '/weight_distributions_bars.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Bar plot saved to: {config.OUTPUT_DIR}/weight_distributions_bars.png")

# Print summary statistics
print(f"\n📈 Distribution Analysis:")
print(f"  Input weights span:  {total_abs_input_weights.min():.3f} to {total_abs_input_weights.max():.3f}")
print(f"  Output weights span: {total_abs_output_weights.min():.3f} to {total_abs_output_weights.max():.3f}")
print(f"  Hidden weights span: {total_abs_hidden_weights.min():.3f} to {total_abs_hidden_weights.max():.3f}")