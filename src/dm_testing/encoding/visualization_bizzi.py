# %%
"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plants import SequentialReacher
from environments import SequentialReachingEnv
from utils import *
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from policy import NumpyStyleRNNPolicy
from shimmy_wrapper import ModalitySpecificEncoder
from rnn_adapter import RNNAdapter, config

reacher = SequentialReacher(plant_xml_file="arm.xml")
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 3, "min": 1, "max": 6},
    num_targets=10,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)

# Setup output directory
config.setup_output_dir()

# Load model using config
best_rnn = RNNAdapter()

# Add debug prints to verify dimensions
print(f"Input weights shape: {best_rnn.W_in.shape}")  # Should be (64, 38) now
print(f"Hidden weights shape: {best_rnn.W_h.shape}")  # Should be (64, 64)
print(f"Output weights shape: {best_rnn.W_out.shape}")  # Should be (4, 64)
print(f"Hidden biases shape: {best_rnn.b_h.shape}")   # Should be (64,)
print(f"Output biases shape: {best_rnn.b_out.shape}")  # Should be (4,)

# %%
"""
WEIGHT VISUALIZATION
"""

# Extract weights CORRECTLY - NO TRANSPOSE!
weights_input = best_rnn.W_in      # (64, 38) ✅
weights_hidden = best_rnn.W_h      # (64, 64) ✅  
weights_output = best_rnn.W_out    # (4, 64) ✅ - DON'T TRANSPOSE!
bias_hidden = best_rnn.b_h         # (64,) ✅
bias_output = best_rnn.b_out       # (4,) ✅

# Combine all weights and biases into a single figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot input weights
sns.heatmap(weights_input, cmap="viridis", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title("Input Weights")
axes[0, 0].set_xlabel("Input Features")
axes[0, 0].set_ylabel("Hidden Units")

# Plot hidden weights
sns.heatmap(weights_hidden, cmap="viridis", cbar=True, ax=axes[0, 1])
axes[0, 1].set_title("Hidden Weights")
axes[0, 1].set_xlabel("Hidden Units")
axes[0, 1].set_ylabel("Hidden Units")

# Plot output weights
sns.heatmap(weights_output, cmap="viridis", cbar=True, ax=axes[0, 2])
axes[0, 2].set_title("Output Weights")
axes[0, 2].set_xlabel("Hidden Units")
axes[0, 2].set_ylabel("Output Units")

# Plot hidden biases
sns.heatmap(
    bias_hidden.reshape(1, -1), cmap="viridis", cbar=True, annot=False, ax=axes[1, 1]
)
axes[1, 1].set_title("Hidden Biases")
axes[1, 1].set_xlabel("Hidden Units")
axes[1, 1].set_yticks([])

# Plot output biases
sns.heatmap(
    bias_output.reshape(1, -1), cmap="viridis", cbar=True, annot=False, ax=axes[1, 2]
)
axes[1, 2].set_title("Output Biases")
axes[1, 2].set_xlabel("Output Units")
axes[1, 2].set_yticks([])

# Remove the last unused subplot
axes[1, 0].axis("off")

# Adjust layout
plt.tight_layout()

# Save plot
config.save_plot("weight_heatmaps.png")
plt.show()

# %%
"""
INPUT SPACE ANALYSIS
"""
num_targets = 1000

# Sample target positions from the reacher
sampled_targets = reacher.sample_targets(num_targets)

# Extract x and y coordinates from the sampled targets
x_coords = sampled_targets[:, 0]
y_coords = sampled_targets[:, 1]

# Plot the x and y coordinates
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c="red", s=25, alpha=0.7)
plt.title("Sampled Targets: X vs Y Coordinates")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

# Save plot
config.save_plot("sampled_targets.png")
plt.show()

# Extract encoded target weights for 4×4 grid encoding
# Input structure: [muscle(12), grid_xy(16), target_z(1)] = 29D total

input_weights_muscles = weights_input[:, :12]           # 12D muscle sensors
input_weights_encoded_target = weights_input[:, 12:28]  # 16D encoded target (4×4 grid)
input_weights_target_z = weights_input[:, 28:29]        # 1D direct Z coordinate

# To properly analyze target influence, we need to encode the sampled targets first
with torch.no_grad():
    # Convert to torch tensor
    target_tensor = torch.tensor(sampled_targets, dtype=torch.float32)
    
    # Create encoder instance (using grid_size=4 for 4×4 grid)
    encoder = ModalitySpecificEncoder(grid_size=4)  # 4×4 = 16D
    
    # Encode XY coordinates (2D -> 16D grid encoding)
    target_xy = target_tensor[:, :2]
    encoded_target_xy = encoder.target_encoder(target_xy)  # [num_targets, 16]
    
    # Add Z coordinate directly
    target_z = target_tensor[:, 2:3]

# OPTION 1: Use only the grid-encoded XY for projection (16D)
projections_encoded_xy = np.dot(encoded_target_xy.numpy(), input_weights_encoded_target.T)

# OPTION 2: Use the full target encoding including Z (17D)
# Combine grid-encoded XY + direct Z: [16D + 1D] = 17D
input_weights_full_target = weights_input[:, 12:29]  # 17D: grid_xy(16) + target_z(1)
full_encoded_target = torch.cat([encoded_target_xy, target_z], dim=1)  # [1000, 17]
projections_encoded_full = np.dot(full_encoded_target.numpy(), input_weights_full_target.T)

# Use the full target encoding for the rest of the analysis
projections_encoded = projections_encoded_full

# Perform PCA on the encoded projections
pca = PCA(n_components=3)
pca_projections_encoded = pca.fit_transform(projections_encoded)

# Select specific hidden units for visualization (adjust indices as needed)
selected_units = [6, 19, 24] if weights_input.shape[0] > 24 else [0, 1, 2]
selected_weights_encoded = input_weights_full_target[selected_units, :]  # Use full 17D weights
selected_projections_encoded = np.dot(full_encoded_target.numpy(), selected_weights_encoded.T)

# Note: Plotly figures are saved separately as HTML
# Visualize the encoded target projections in 3D using Plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=selected_projections_encoded[:, 0],
            y=selected_projections_encoded[:, 1] if selected_projections_encoded.shape[1] > 1 else np.zeros(len(selected_projections_encoded)),
            z=selected_projections_encoded[:, 2] if selected_projections_encoded.shape[1] > 2 else np.zeros(len(selected_projections_encoded)),
            mode="markers",
            marker=dict(
                size=2,
                color="black",
                opacity=0.8,
            ),
        )
    ]
)

fig.update_layout(
    title="Encoded Target Projections Through RNN Input Weights (3D)",
    scene=dict(
        xaxis_title=f"Unit {selected_units[0]}", 
        yaxis_title=f"Unit {selected_units[1]}", 
        zaxis_title=f"Unit {selected_units[2]}"
    ),
)

# Save plotly figure
fig.write_html(os.path.join(config.OUTPUT_DIR, "encoded_target_projections_3d.html"))
print(f"💾 Saved plot: {os.path.join(config.OUTPUT_DIR, 'encoded_target_projections_3d.html')}")
fig.show()

# Visualize the PCA projections of encoded targets in 3D
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=pca_projections_encoded[:, 0],
            y=pca_projections_encoded[:, 1],
            z=pca_projections_encoded[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=pca_projections_encoded[:, 0],  # Color by the first principal component
                colorscale="Viridis",
                opacity=0.8,
            ),
        )
    ]
)

fig.update_layout(
    title="PCA Projections of Encoded Targets Through RNN Input Weights (3D)",
    scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
)

# Save plotly figure
fig.write_html(os.path.join(config.OUTPUT_DIR, "pca_projections_3d.html"))
print(f"💾 Saved plot: {os.path.join(config.OUTPUT_DIR, 'pca_projections_3d.html')}")
fig.show()

# %%
"""
WEIGHT DISTRIBUTION ANALYSIS
"""

# Calculate the total absolute output weights for each hidden unit
total_abs_output_weights = np.sum(np.abs(weights_output), axis=0)  # Sum over 4 muscles → (64,)

# Plot the distribution of total absolute output weights
plt.figure(figsize=(10, 6))
sns.histplot(total_abs_output_weights, kde=True, bins=20, color="blue")
plt.title("Distribution of Total Absolute Output Weights for Each Hidden Unit")
plt.xlabel("Total Absolute Output Weights")
plt.ylabel("Frequency")
plt.grid(True)

# Save plot
config.save_plot("output_weights_distribution.png")
plt.show()

# Calculate the total absolute sensory weights for each hidden unit
total_abs_input_weights = np.sum(np.abs(weights_input), axis=1)    # Sum over 38 inputs → (64,)

# Plot the distribution of total absolute sensory weights
plt.figure(figsize=(10, 6))
sns.histplot(total_abs_input_weights, kde=True, bins=20, color="green")
plt.title("Distribution of Total Absolute Sensory Weights for Each Hidden Unit")
plt.xlabel("Total Absolute Sensory Weights")
plt.ylabel("Frequency")
plt.grid(True)

# Save plot
config.save_plot("input_weights_distribution.png")
plt.show()

total_abs_hidden_weights = np.sum(np.abs(weights_hidden), axis=1)  # Sum over 64 hidden → (64,)

# Plot the distribution of total absolute hidden weights
plt.figure(figsize=(10, 6))
sns.histplot(total_abs_hidden_weights, kde=True, bins=20, color="orange")
plt.title("Distribution of Total Absolute Hidden Weights for Each Hidden Unit")
plt.xlabel("Total Absolute Hidden Weights")
plt.ylabel("Frequency")
plt.grid(True)

# Save plot
config.save_plot("hidden_weights_distribution.png")
plt.show()

# %%
"""
JOINT DISTRIBUTION PLOTS
"""

# Create a 2D joint distribution plot of total sensory and total output weights
plt.figure(figsize=(10, 8))
joint_plot = sns.jointplot(
    x=total_abs_input_weights,
    y=total_abs_output_weights,
    kind="scatter",
    cmap="viridis",
    marginal_kws=dict(bins=20, fill=True),
)

# Overlay the index of each neuron as text
for i, (x, y) in enumerate(zip(total_abs_input_weights, total_abs_output_weights)):
    joint_plot.ax_joint.text(
        x, y + 0.1, str(i), fontsize=8, color="black", ha="center", va="center"
    )

plt.suptitle("2D Joint Distribution of Total Input and Output Weights", y=1.02)
joint_plot.set_axis_labels(
    "Total Absolute Input Weights", "Total Absolute Output Weights"
)

# Save plot
config.save_plot("joint_input_output_weights.png")
plt.show()

# Create other joint plots and save them
plt.figure(figsize=(10, 8))
joint_plot = sns.jointplot(
    x=total_abs_input_weights,
    y=total_abs_hidden_weights,
    kind="scatter",
    cmap="viridis",
    marginal_kws=dict(bins=20, fill=True),
)

for i, (x, y) in enumerate(zip(total_abs_input_weights, total_abs_hidden_weights)):
    joint_plot.ax_joint.text(
        x, y + 0.1, str(i), fontsize=8, color="black", ha="center", va="center"
    )

plt.suptitle("2D Joint Distribution of Total Input and Hidden Weights", y=1.02)
joint_plot.set_axis_labels(
    "Total Absolute Input Weights", "Total Absolute Hidden Weights"
)

# Save plot
config.save_plot("joint_input_hidden_weights.png")
plt.show()

plt.figure(figsize=(10, 8))
joint_plot = sns.jointplot(
    x=total_abs_hidden_weights,
    y=total_abs_output_weights,
    kind="scatter",
    cmap="viridis",
    marginal_kws=dict(bins=20, fill=True),
)

for i, (x, y) in enumerate(zip(total_abs_hidden_weights, total_abs_output_weights)):
    joint_plot.ax_joint.text(
        x, y + 0.1, str(i), fontsize=8, color="black", ha="center", va="center"
    )

plt.suptitle("2D Joint Distribution of Total Hidden and Output Weights", y=1.02)
joint_plot.set_axis_labels(
    "Total Absolute Hidden Weights", "Total Absolute Output Weights"
)

# Save plot
config.save_plot("joint_hidden_output_weights.png")
plt.show()

# %%
"""
NEURON PATTERN ANALYSIS
"""

print("\n" + "="*80)
print("ANALYZING NEURON WEIGHT PATTERNS")
print("="*80)

# Calculate percentiles for each weight type
input_25th = np.percentile(total_abs_input_weights, 25)
input_75th = np.percentile(total_abs_input_weights, 75)

output_25th = np.percentile(total_abs_output_weights, 25)
output_75th = np.percentile(total_abs_output_weights, 75)

hidden_25th = np.percentile(total_abs_hidden_weights, 25)
hidden_75th = np.percentile(total_abs_hidden_weights, 75)

print(f"Weight Percentiles:")
print(f"  Input weights  - 25th: {input_25th:.3f}, 75th: {input_75th:.3f}")
print(f"  Output weights - 25th: {output_25th:.3f}, 75th: {output_75th:.3f}")
print(f"  Hidden weights - 25th: {hidden_25th:.3f}, 75th: {hidden_75th:.3f}")

# Find neurons with specific patterns
def find_neurons_by_pattern(input_weights, output_weights, hidden_weights, 
                           input_condition, output_condition, hidden_condition, pattern_name):
    """Find neurons matching specific weight patterns"""
    
    input_mask = input_condition(input_weights)
    output_mask = output_condition(output_weights)
    hidden_mask = hidden_condition(hidden_weights)
    
    # Combine all conditions
    pattern_mask = input_mask & output_mask & hidden_mask
    matching_neurons = np.where(pattern_mask)[0]
    
    print(f"\n{pattern_name}:")
    print(f"  Found {len(matching_neurons)} neurons: {matching_neurons}")
    
    if len(matching_neurons) > 0:
        for neuron_idx in matching_neurons:
            print(f"    Neuron {neuron_idx:2d}: Input={input_weights[neuron_idx]:.3f}, "
                  f"Output={output_weights[neuron_idx]:.3f}, Hidden={hidden_weights[neuron_idx]:.3f}")
    
    return matching_neurons

# Find all neuron patterns
internal_processors = find_neurons_by_pattern(
    total_abs_input_weights, total_abs_output_weights, total_abs_hidden_weights,
    lambda x: x < input_25th,     
    lambda x: x < output_25th,    
    lambda x: x > hidden_75th,    
    "🔄 INTERNAL PROCESSORS (Low Input + Low Output + High Hidden)"
)

input_specialists = find_neurons_by_pattern(
    total_abs_input_weights, total_abs_output_weights, total_abs_hidden_weights,
    lambda x: x > input_75th,     
    lambda x: x < output_25th,    
    lambda x: x < hidden_25th,    
    "📥 INPUT SPECIALISTS (High Input + Low Output + Low Hidden)"
)

output_specialists = find_neurons_by_pattern(
    total_abs_input_weights, total_abs_output_weights, total_abs_hidden_weights,
    lambda x: x < input_25th,     
    lambda x: x > output_75th,    
    lambda x: x < hidden_25th,    
    "📤 OUTPUT SPECIALISTS (Low Input + High Output + Low Hidden)"
)

hub_neurons = find_neurons_by_pattern(
    total_abs_input_weights, total_abs_output_weights, total_abs_hidden_weights,
    lambda x: x > input_75th,     
    lambda x: x > output_75th,    
    lambda x: x > hidden_75th,    
    "🌟 HUB NEURONS (High Input + High Output + High Hidden)"
)

silent_neurons = find_neurons_by_pattern(
    total_abs_input_weights, total_abs_output_weights, total_abs_hidden_weights,
    lambda x: x < input_25th,     
    lambda x: x < output_25th,    
    lambda x: x < hidden_25th,    
    "💤 SILENT NEURONS (Low Input + Low Output + Low Hidden)"
)

# Create enhanced joint plots with highlighted special neurons
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Input vs Output (highlight internal processors)
axes[0].scatter(total_abs_input_weights, total_abs_output_weights, 
               c='lightgray', alpha=0.6, s=50)

if len(internal_processors) > 0:
    axes[0].scatter(total_abs_input_weights[internal_processors], 
                   total_abs_output_weights[internal_processors],
                   c='red', s=100, alpha=0.8, label=f'Internal Processors ({len(internal_processors)})')
    
    for neuron_idx in internal_processors:
        axes[0].annotate(f'{neuron_idx}', 
                        (total_abs_input_weights[neuron_idx], total_abs_output_weights[neuron_idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, color='red', weight='bold')

if len(hub_neurons) > 0:
    axes[0].scatter(total_abs_input_weights[hub_neurons], 
                   total_abs_output_weights[hub_neurons],
                   c='gold', s=100, alpha=0.8, label=f'Hub Neurons ({len(hub_neurons)})')

axes[0].set_xlabel('Total Absolute Input Weights')
axes[0].set_ylabel('Total Absolute Output Weights')
axes[0].set_title('Input vs Output Weights')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Input vs Hidden (highlight internal processors)
axes[1].scatter(total_abs_input_weights, total_abs_hidden_weights, 
               c='lightgray', alpha=0.6, s=50)

if len(internal_processors) > 0:
    axes[1].scatter(total_abs_input_weights[internal_processors], 
                   total_abs_hidden_weights[internal_processors],
                   c='red', s=100, alpha=0.8, label=f'Internal Processors ({len(internal_processors)})')
    
    for neuron_idx in internal_processors:
        axes[1].annotate(f'{neuron_idx}', 
                        (total_abs_input_weights[neuron_idx], total_abs_hidden_weights[neuron_idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, color='red', weight='bold')

if len(hub_neurons) > 0:
    axes[1].scatter(total_abs_input_weights[hub_neurons], 
                   total_abs_hidden_weights[hub_neurons],
                   c='gold', s=100, alpha=0.8, label=f'Hub Neurons ({len(hub_neurons)})')

axes[1].set_xlabel('Total Absolute Input Weights')
axes[1].set_ylabel('Total Absolute Hidden Weights')
axes[1].set_title('Input vs Hidden Weights')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: Hidden vs Output (highlight internal processors)
axes[2].scatter(total_abs_hidden_weights, total_abs_output_weights, 
               c='lightgray', alpha=0.6, s=50)

if len(internal_processors) > 0:
    axes[2].scatter(total_abs_hidden_weights[internal_processors], 
                   total_abs_output_weights[internal_processors],
                   c='red', s=100, alpha=0.8, label=f'Internal Processors ({len(internal_processors)})')
    
    for neuron_idx in internal_processors:
        axes[2].annotate(f'{neuron_idx}', 
                        (total_abs_hidden_weights[neuron_idx], total_abs_output_weights[neuron_idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, color='red', weight='bold')

if len(hub_neurons) > 0:
    axes[2].scatter(total_abs_hidden_weights[hub_neurons], 
                   total_abs_output_weights[hub_neurons],
                   c='gold', s=100, alpha=0.8, label=f'Hub Neurons ({len(hub_neurons)})')

axes[2].set_xlabel('Total Absolute Hidden Weights')
axes[2].set_ylabel('Total Absolute Output Weights')
axes[2].set_title('Hidden vs Output Weights')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.suptitle('Neuron Weight Patterns Analysis', fontsize=16)
plt.tight_layout()

# Save plot
config.save_plot("neuron_patterns_analysis.png")
plt.show()

print(f"\n🎯 SUMMARY:")
print(f"  Internal Processors: {len(internal_processors)} neurons")
print(f"  Hub Neurons: {len(hub_neurons)} neurons")
print(f"  Input Specialists: {len(input_specialists)} neurons")
print(f"  Output Specialists: {len(output_specialists)} neurons")
print(f"  Silent Neurons: {len(silent_neurons)} neurons")

print(f"\n✅ All plots saved to: {config.OUTPUT_DIR}")



# %%
