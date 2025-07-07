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
from rnn_adapter import RNNAdapter, config

# %%
"""
.##........#######.....###....########.
.##.......##.....##...##.##...##.....##
.##.......##.....##..##...##..##.....##
.##.......##.....##.##.....##.##.....##
.##.......##.....##.#########.##.....##
.##.......##.....##.##.....##.##.....##
.########..#######..##.....##.########.
"""

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
print(f"Input weights shape: {best_rnn.W_in.shape}")  # Should be (64, 93)
print(f"Hidden weights shape: {best_rnn.W_h.shape}")  # Should be (64, 64)
print(f"Output weights shape: {best_rnn.W_out.shape}")  # Should be (4, 64)
print(f"Hidden biases shape: {best_rnn.b_h.shape}")   # Should be (64,)
print(f"Output biases shape: {best_rnn.b_out.shape}")  # Should be (4,)

# %%
"""
..######..########.####.##.....##.##.....##.##..........###....########.########
.##....##....##.....##..###...###.##.....##.##.........##.##......##....##......
.##..........##.....##..####.####.##.....##.##........##...##.....##....##......
..######.....##.....##..##.###.##.##.....##.##.......##.....##....##....######..
.......##....##.....##..##.....##.##.....##.##.......#########....##....##......
.##....##....##.....##..##.....##.##.....##.##.......##.....##....##....##......
..######.....##....####.##.....##..#######..########.##.....##....##....########
"""

print("Analyzing convergence for ALL neurons - no plotting...")
print(f"Starting analysis with {config.HIDDEN_SIZE} hidden units...")

# Zero out encoded target influence for stimulation
print("Zeroing out encoded target influence for stimulation...")
best_rnn.policy.W_in.data[:, 12:] = 0  # Zero 81D encoded target

# Initialize lists to store convergence metrics for all units
convergence_scores = []
unit_indices = []
stimulation_results = []

# ANALYZE ALL UNITS (no plotting)
for unit_idx in range(config.HIDDEN_SIZE):
    print(f"Analyzing unit {unit_idx}/{config.HIDDEN_SIZE}...")

    # Stimulate the unit
    force_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
    )

    # Process data
    position_vecs = np.nan_to_num(np.array(force_data["position"]))
    force_vecs = np.nan_to_num(np.array(force_data["force"]))
    time = np.linspace(0, reacher.data.time, len(force_vecs))

    # Calculate average forces in different time windows
    time_window = 0.1  # 100 ms
    average_positions = []
    rest_average_forces = []
    stim_average_forces = []

    # Iterate second by second
    for t in range(1, int(reacher.data.time) + 1):
        # Rest period (before stimulation in each cycle)
        start_time = t - 0.5 - time_window
        stop_time = t - 0.5
        indices = (time > start_time) & (time <= stop_time)
        avg_force_vec = np.mean(force_vecs[indices], axis=0)
        rest_average_forces.append(avg_force_vec)

        # Stimulation period
        start_time = t - time_window
        stop_time = t
        indices = (time > start_time) & (time <= stop_time)
        avg_position_vec = np.mean(position_vecs[indices], axis=0)
        avg_force_vec = np.mean(force_vecs[indices], axis=0)
        average_positions.append(avg_position_vec)
        stim_average_forces.append(avg_force_vec)

    # Convert to numpy arrays
    average_positions = np.array(average_positions)
    rest_average_forces = np.array(rest_average_forces)
    stim_average_forces = np.array(stim_average_forces)

    # Calculate difference between stimulated and rest force fields
    force_difference = stim_average_forces - rest_average_forces
    
    # Extract x and y components
    x_positions = average_positions[:, 0]
    y_positions = average_positions[:, 1]
    x_force_diff = force_difference[:, 0]
    y_force_diff = force_difference[:, 1]

    # CONVERGENCE ANALYSIS
    # Method 1: Calculate convergence point using force differences
    force_diff_magnitudes = np.sqrt(x_force_diff**2 + y_force_diff**2)
    
    # Only consider positions where there's significant force difference
    significant_force_mask = force_diff_magnitudes > np.percentile(force_diff_magnitudes, 50)
    
    if np.sum(significant_force_mask) > 3:  # Need at least 3 points for meaningful analysis
        # Weighted convergence point
        weights = force_diff_magnitudes[significant_force_mask]
        convergence_x = np.average(x_positions[significant_force_mask], weights=weights)
        convergence_y = np.average(y_positions[significant_force_mask], weights=weights)
        
        # Method 2: Calculate convergence strength - how well do forces point toward convergence point
        convergence_vectors = np.column_stack([
            convergence_x - x_positions[significant_force_mask],
            convergence_y - y_positions[significant_force_mask]
        ])
        force_vectors = np.column_stack([
            x_force_diff[significant_force_mask],
            y_force_diff[significant_force_mask]
        ])
        
        # Normalize vectors
        convergence_vectors_norm = convergence_vectors / (np.linalg.norm(convergence_vectors, axis=1, keepdims=True) + 1e-8)
        force_vectors_norm = force_vectors / (np.linalg.norm(force_vectors, axis=1, keepdims=True) + 1e-8)
        
        # Calculate alignment (dot product)
        alignment = np.sum(convergence_vectors_norm * force_vectors_norm, axis=1)
        
        # Convergence score: mean alignment weighted by force magnitude
        convergence_score = np.average(alignment, weights=weights)
        
        # Alternative convergence metrics
        force_consistency = np.std(force_diff_magnitudes[significant_force_mask])  # Lower = more consistent
        spatial_spread = np.std(np.column_stack([x_positions[significant_force_mask], y_positions[significant_force_mask]]), axis=0).mean()
        
        # Combined convergence score (higher = better convergence)
        combined_score = convergence_score * np.mean(weights) / (force_consistency + 1e-8)
        
    else:
        convergence_score = 0
        combined_score = 0
        convergence_x = np.mean(x_positions)
        convergence_y = np.mean(y_positions)

    # Store results
    convergence_scores.append(combined_score)
    unit_indices.append(unit_idx)
    
    # Store detailed results for top units
    stimulation_results.append({
        'unit_idx': unit_idx,
        'convergence_score': combined_score,
        'raw_alignment': convergence_score if 'convergence_score' in locals() else 0,
        'convergence_point': (convergence_x, convergence_y),
        'mean_force_magnitude': np.mean(force_diff_magnitudes),
        'max_force_magnitude': np.max(force_diff_magnitudes),
        'positions': average_positions,
        'rest_forces': rest_average_forces,
        'stim_forces': stim_average_forces,
        'force_differences': force_difference
    })

# Convert to numpy arrays
convergence_scores = np.array(convergence_scores)
unit_indices = np.array(unit_indices)

# Find top units with highest convergence
top_unit_indices = np.argsort(convergence_scores)[-config.TOP_N_UNITS:][::-1]  # Highest first

print("\n" + "="*60)
print("CONVERGENCE ANALYSIS RESULTS")
print("="*60)
print(f"Analyzed {len(unit_indices)} units total")
print(f"Mean convergence score: {np.mean(convergence_scores):.4f}")
print(f"Std convergence score: {np.std(convergence_scores):.4f}")
print(f"Top {config.TOP_N_UNITS} units by convergence score:")

for i, idx in enumerate(top_unit_indices):
    unit_idx = unit_indices[idx]
    score = convergence_scores[idx]
    result = stimulation_results[idx]
    print(f"  #{i+1}: Unit {unit_idx:2d} - Score: {score:.4f} - Conv.Point: ({result['convergence_point'][0]:.3f}, {result['convergence_point'][1]:.3f})")

print("\n" + "="*60)
print("PLOTTING TOP CONVERGENT UNITS")
print("="*60)

# NOW PLOT ONLY THE TOP UNITS
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for plot_idx, result_idx in enumerate(top_unit_indices):
    if plot_idx >= len(axes):
        break
        
    ax = axes[plot_idx]
    result = stimulation_results[result_idx]
    unit_idx = result['unit_idx']
    
    # Extract data
    x_positions = result['positions'][:, 0]
    y_positions = result['positions'][:, 1]
    x_force_diff = result['force_differences'][:, 0]
    y_force_diff = result['force_differences'][:, 1]
    
    # Plot force difference field (stimulated - rest)
    quiver = ax.quiver(
        x_positions,
        y_positions,
        x_force_diff,
        y_force_diff,
        angles="xy",
        scale_units="xy",
        scale=500,
        linewidth=1,
        color="red",
        alpha=0.8,
    )
    
    # Mark convergence point
    conv_x, conv_y = result['convergence_point']
    ax.plot(conv_x, conv_y, 'ko', markersize=10, markerfacecolor='yellow', 
            markeredgecolor='black', markeredgewidth=2, label='Convergence')
    
    # Formatting
    ax.set_title(f"Unit {unit_idx} - Conv: {result['convergence_score']:.3f}\n"
                f"Force: {result['mean_force_magnitude']:.3f}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0])
    ax.set_ylim(reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1])
    ax.legend()

plt.suptitle(f"Top {config.TOP_N_UNITS} Units by Convergence Score (Force Difference Fields)", fontsize=16)
plt.tight_layout()

# Save plot
config.save_plot("top_convergent_units.png")
plt.show()

# Summary statistics
print(f"\nConvergence score distribution:")
print(f"  Min: {np.min(convergence_scores):.4f}")
print(f"  25th percentile: {np.percentile(convergence_scores, 25):.4f}")
print(f"  Median: {np.percentile(convergence_scores, 50):.4f}")
print(f"  75th percentile: {np.percentile(convergence_scores, 75):.4f}")
print(f"  Max: {np.max(convergence_scores):.4f}")

print(f"\nUnits with convergence score > 75th percentile:")
high_convergence_units = unit_indices[convergence_scores > np.percentile(convergence_scores, 75)]
print(f"  Units: {high_convergence_units}")
print(f"  Count: {len(high_convergence_units)}/{len(unit_indices)}")

print(f"\n✅ All plots saved to: {config.OUTPUT_DIR}")
print("Analysis complete!")

# %%
