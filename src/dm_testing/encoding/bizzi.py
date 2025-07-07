#%%
import os
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

# %%
"""
EXTRACT MODEL WEIGHTS AND CALCULATE STATISTICS
"""

# Extract weights and biases from the RNN
weights_input = best_rnn.W_in                    # (hidden_size, 29) - FIXED from 93
weights_hidden = best_rnn.W_h                    # (hidden_size, hidden_size)  
weights_output = best_rnn.W_out                  # (4, hidden_size) - DON'T transpose!
bias_hidden = best_rnn.b_h                       # (hidden_size,)
bias_output = best_rnn.b_out                     # (4,)

print(f"✅ Extracted model weights:")
print(f"  Input weights shape: {weights_input.shape}")  # Should show (64, 29)
print(f"  Hidden weights shape: {weights_hidden.shape}")
print(f"  Output weights shape: {weights_output.shape}")
print(f"  Hidden bias shape: {bias_hidden.shape}")
print(f"  Output bias shape: {bias_output.shape}")

# Calculate weight statistics for neuron analysis - FIXED!
total_abs_output_weights = np.sum(np.abs(weights_output), axis=0)  # Sum over output dimensions -> (64,)
total_abs_input_weights = np.sum(np.abs(weights_input), axis=1)    # Sum over input dimensions -> (64,)
total_abs_hidden_weights = np.sum(np.abs(weights_hidden), axis=1)  # Sum over hidden dimensions -> (64,)

print(f"✅ Calculated weight statistics:")
print(f"  Total abs input weights shape: {total_abs_input_weights.shape}")
print(f"  Total abs output weights shape: {total_abs_output_weights.shape}")
print(f"  Total abs hidden weights shape: {total_abs_hidden_weights.shape}")
print(f"  Total abs input weights: min={np.min(total_abs_input_weights):.3f}, max={np.max(total_abs_input_weights):.3f}")
print(f"  Total abs output weights: min={np.min(total_abs_output_weights):.3f}, max={np.max(total_abs_output_weights):.3f}")
print(f"  Total abs hidden weights: min={np.min(total_abs_hidden_weights):.3f}, max={np.max(total_abs_hidden_weights):.3f}")

print("✅ Ready for stimulation analysis!")

# Zero out encoded target influence for stimulation
print("Zeroing out encoded target influence for stimulation...")
best_rnn.policy.W_in.data[:, 12:] = 0  # Zero 81D encoded target (12: covers 12:93 for target + Z)

def calculate_convergence_point_intersection(x_pos, y_pos, x_forces, y_forces, threshold=1e-4):
    """Find where force vectors intersect - true convergence point"""
    valid_mask = np.sqrt(x_forces**2 + y_forces**2) > threshold
    
    if np.sum(valid_mask) < 2:
        return None, None
    
    x_pos_valid = x_pos[valid_mask]
    y_pos_valid = y_pos[valid_mask] 
    x_forces_valid = x_forces[valid_mask]
    y_forces_valid = y_forces[valid_mask]
    
    # Find intersections of force vector lines
    intersections_x = []
    intersections_y = []
    
    for i in range(len(x_pos_valid)):
        for j in range(i+1, len(x_pos_valid)):
            # Line 1: from (x1,y1) in direction (fx1,fy1)
            x1, y1, fx1, fy1 = x_pos_valid[i], y_pos_valid[i], x_forces_valid[i], y_forces_valid[i]
            # Line 2: from (x2,y2) in direction (fx2,fy2)  
            x2, y2, fx2, fy2 = x_pos_valid[j], y_pos_valid[j], x_forces_valid[j], y_forces_valid[j]
            
            # Solve for intersection: (x1,y1) + t1*(fx1,fy1) = (x2,y2) + t2*(fx2,fy2)
            denom = fx1*fy2 - fx2*fy1
            if abs(denom) > 1e-6:  # Lines not parallel
                t1 = ((x2-x1)*fy2 - (y2-y1)*fx2) / denom
                intersect_x = x1 + t1*fx1
                intersect_y = y1 + t1*fy1
                intersections_x.append(intersect_x)
                intersections_y.append(intersect_y)
    
    if len(intersections_x) == 0:
        return None, None
        
    # Return median intersection point (robust to outliers)
    return np.median(intersections_x), np.median(intersections_y)

# STIMULATE SPECIFIC UNITS from config with enhanced debugging
stimulation_data = {}  # Store data for plotting

for unit_idx in config.STIMULATION_UNITS:
    print(f"\n🔍 Debugging Unit {unit_idx}:")
    
    # CHECK OUTPUT WEIGHTS - This is the key diagnostic!
    unit_output_weights = weights_output[:, unit_idx]  # All 4 muscle connections for this unit
    print(f"  Output weights to muscles: {unit_output_weights}")
    print(f"  Output weights magnitude: {np.abs(unit_output_weights)}")
    print(f"  Max output weight: {np.max(np.abs(unit_output_weights)):.6f}")
    print(f"  Total output strength: {total_abs_output_weights[unit_idx]:.6f}")
    
    # CHECK INPUT WEIGHTS (muscle sensors only - first 12 dims)
    unit_input_weights = weights_input[unit_idx, :12]  # Only muscle sensor connections
    print(f"  Input weights from muscle sensors: {unit_input_weights[:4]}...")  # Show first 4
    print(f"  Max input weight: {np.max(np.abs(unit_input_weights)):.6f}")
    print(f"  Total input strength: {total_abs_input_weights[unit_idx]:.6f}")
    
    # CHECK HIDDEN RECURRENT WEIGHTS
    unit_hidden_out = weights_hidden[unit_idx, :]  # How this unit affects others
    unit_hidden_in = weights_hidden[:, unit_idx]   # How others affect this unit
    print(f"  Max outgoing hidden weight: {np.max(np.abs(unit_hidden_out)):.6f}")
    print(f"  Max incoming hidden weight: {np.max(np.abs(unit_hidden_in)):.6f}")
    
    # CHECK BIAS
    print(f"  Hidden bias: {bias_hidden[unit_idx]:.6f}")
    
    # PREDICTION: Will this unit produce forces?
    max_output = np.max(np.abs(unit_output_weights))
    if max_output < 1e-4:
        print(f"  ⚠️  WARNING: Very weak output connections! Max = {max_output:.6f}")
        print(f"      This unit likely won't produce measurable forces.")
    elif max_output < 1e-2:
        print(f"  ⚠️  CAUTION: Weak output connections. Max = {max_output:.6f}")
        print(f"      Forces may be very small.")
    else:
        print(f"  ✅ Strong output connections. Max = {max_output:.6f}")
        print(f"      Should produce measurable forces.")

    print(f"\n🚀 Stimulating unit {unit_idx}...")

    # CAPTURE HIDDEN STATE BEFORE AND DURING STIMULATION
    pre_stim_hidden = best_rnn.h.copy()
    
    # Use the existing env and best_rnn (RNNAdapter)
    force_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
    )
    
    post_stim_hidden = best_rnn.h.copy()
    
    # CHECK IF STIMULATION ACTUALLY CHANGED THE HIDDEN STATE
    hidden_change = np.abs(post_stim_hidden[unit_idx] - pre_stim_hidden[unit_idx])
    print(f"  Hidden state change for unit {unit_idx}: {hidden_change:.6f}")
    
    if hidden_change < 1e-6:
        print(f"  ⚠️  WARNING: Stimulation didn't change hidden state!")
    
    # ANALYZE THE FORCE DATA
    position_vecs = np.nan_to_num(np.array(force_data["position"]))
    force_vecs = np.nan_to_num(np.array(force_data["force"]))
    
    # Check raw force statistics
    force_magnitudes_raw = np.sqrt(np.sum(force_vecs**2, axis=1))
    print(f"  Raw force data:")
    print(f"    Shape: {force_vecs.shape}")
    print(f"    Non-zero samples: {np.sum(force_magnitudes_raw > 1e-6)}/{len(force_magnitudes_raw)}")
    print(f"    Force range: [{np.min(force_magnitudes_raw):.6f}, {np.max(force_magnitudes_raw):.6f}]")
    print(f"    Mean force: {np.mean(force_magnitudes_raw):.6f}")
    
    # Check for NaN values
    nan_count = np.sum(np.isnan(force_vecs))
    if nan_count > 0:
        print(f"  ⚠️  WARNING: {nan_count} NaN values in force data!")
    
    # CHECK IF CONSTRAINT WAS ACTIVE
    if hasattr(force_data, 'eq_active'):
        active_count = np.sum(force_data['eq_active'])
        print(f"  Constraint active samples: {active_count}/{len(force_data['eq_active'])}")
        if active_count == 0:
            print(f"  ⚠️  ERROR: Constraint was never active! Forces will be zero.")
    
    # Process data (existing code)
    time = np.linspace(0, reacher.data.time, len(force_vecs))

    # Define the time window for averaging (100 ms)
    time_window = 0.1

    # Initialize lists to store average force vectors
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

    # FINAL FORCE DIFFERENCE ANALYSIS
    rest_forces_array = np.array(rest_average_forces)
    stim_forces_array = np.array(stim_average_forces)
    force_diff = stim_forces_array - rest_forces_array
    force_diff_magnitudes = np.sqrt(np.sum(force_diff**2, axis=1))
    
    print(f"  Final force difference analysis:")
    print(f"    Max difference magnitude: {np.max(force_diff_magnitudes):.6f}")
    print(f"    Mean difference magnitude: {np.mean(force_diff_magnitudes):.6f}")
    print(f"    Samples with difference > 1e-6: {np.sum(force_diff_magnitudes > 1e-6)}/{len(force_diff_magnitudes)}")
    
    if np.max(force_diff_magnitudes) < 1e-6:
        print(f"  🔴 ZERO FORCES CONFIRMED for unit {unit_idx}")
        if max_output < 1e-4:
            print(f"      CAUSE: Weak output weights (max = {max_output:.6f})")
        else:
            print(f"      CAUSE: Unknown - output weights seem strong enough")
    else:
        print(f"  ✅ NON-ZERO FORCES detected for unit {unit_idx}")

    # Store data for plotting
    stimulation_data[unit_idx] = {
        'positions': np.array(average_positions),
        'rest_forces': rest_forces_array,
        'stim_forces': stim_forces_array,
        'force_diff_magnitudes': force_diff_magnitudes,
        'max_output_weight': max_output
    }

# SUMMARY ANALYSIS ACROSS ALL STIMULATED UNITS
print(f"\n📊 SUMMARY ANALYSIS:")
for unit_idx in config.STIMULATION_UNITS:
    data = stimulation_data[unit_idx]
    max_force_diff = np.max(data['force_diff_magnitudes'])
    max_output = data['max_output_weight']
    
    print(f"Unit {unit_idx}: Max force diff = {max_force_diff:.6f}, Max output weight = {max_output:.6f}")
    
    if max_force_diff < 1e-6 and max_output < 1e-4:
        print(f"  → WEAK OUTPUT WEIGHTS explain zero forces")
    elif max_force_diff < 1e-6 and max_output >= 1e-4:
        print(f"  → ZERO FORCES despite strong weights - investigate further!")
    else:
        print(f"  → Normal force production")

# NOW PLOT THE RESULTS
for unit_idx in config.STIMULATION_UNITS:
    data = stimulation_data[unit_idx]
    
    # Extract data
    x_positions = data['positions'][:, 0]
    y_positions = data['positions'][:, 1]
    x_stim_forces = data['stim_forces'][:, 0]
    y_stim_forces = data['stim_forces'][:, 1]
    x_rest_forces = data['rest_forces'][:, 0]
    y_rest_forces = data['rest_forces'][:, 1]
    
    # Calculate difference
    x_force_diff = x_stim_forces - x_rest_forces
    y_force_diff = y_stim_forces - y_rest_forces
    
    # Enhanced unit characteristics - NOW CORRECTLY INDEXED
    output_strength = total_abs_output_weights[unit_idx]  # Now correctly (64,) array
    input_strength = total_abs_input_weights[unit_idx]
    hidden_strength = total_abs_hidden_weights[unit_idx]
    
    # PLOT 1: Stimulated vs Rest Forces - MUCH LARGER ARROWS
    plt.figure(figsize=(10, 10))
    
    # Plot stimulated forces in red - BIGGER ARROWS
    plt.quiver(
        x_positions, y_positions, x_stim_forces, y_stim_forces,
        angles="xy", scale_units="xy", scale=500,  
        linewidth=3, color="red", alpha=0.8, label="Stimulated", 
        width=0.003,  
        headwidth=2, headlength=3  
    )
    
    # Plot rest forces in black - BIGGER ARROWS
    plt.quiver(
        x_positions, y_positions, x_rest_forces, y_rest_forces,
        angles="xy", scale_units="xy", scale=500,  
        linewidth=2, color="black", alpha=0.6, label="Rest", 
        width=0.002,  
        headwidth=2, headlength=3  
    )
    
    plt.title(f"Unit {unit_idx} - Stimulated vs Rest Forces\n"
              f"Output:{output_strength:.2f}, Input:{input_strength:.2f}, Hidden:{hidden_strength:.2f}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim(reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0])
    plt.ylim(reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1])
    plt.tight_layout()
    
    # Save plot
    config.save_plot(f"bizzi_unit_{unit_idx}_stimulated_vs_rest.png")
    plt.show()
    
    # PLOT 2: Force Difference Field with ADAPTIVE SCALING
    plt.figure(figsize=(10, 10))
    
    # Calculate force magnitudes for color coding
    force_magnitudes = np.sqrt(x_force_diff**2 + y_force_diff**2)
    max_force = np.max(force_magnitudes)
    
    # ADAPTIVE SCALING: Calculate scale to make largest arrow ~10% of plot width
    if max_force > 0:
        plot_width = np.max(x_positions) - np.min(x_positions)
        target_arrow_size = plot_width * 0.1  # Want largest arrow to be 10% of plot width
        adaptive_scale = max_force / target_arrow_size
        print(f"  Adaptive scaling for unit {unit_idx}:")
        print(f"    Max force: {max_force:.2e}")
        print(f"    Plot width: {plot_width:.3f}")
        print(f"    Target arrow size: {target_arrow_size:.3f}")
        print(f"    Adaptive scale: {adaptive_scale:.2e}")
    else:
        adaptive_scale = 1
        print(f"  No forces detected for unit {unit_idx}, using default scale")
    
    # Plot force difference field with adaptive scaling
    quiver = plt.quiver(
        x_positions, y_positions, x_force_diff, y_force_diff,
        force_magnitudes,  # Color by magnitude
        angles="xy", scale_units="xy", scale=adaptive_scale,
        linewidth=2.5, alpha=0.9, cmap='plasma', 
        width=0.008, headwidth=4, headlength=5
    )
    
    # Add colorbar
    cbar = plt.colorbar(quiver, shrink=0.8)
    cbar.set_label('Force Magnitude', fontsize=12)
    
    # Calculate proper convergence point using intersection method
    convergence_x, convergence_y = calculate_convergence_point_intersection(
        x_positions, y_positions, x_force_diff, y_force_diff
    )
    
    if convergence_x is not None and convergence_y is not None:
        plt.plot(convergence_x, convergence_y, 'o', markersize=15,  
                markerfacecolor='cyan', markeredgecolor='black', 
                markeredgewidth=3, label='Convergence Point')
        plt.legend(fontsize=12)
        print(f"Convergence point for unit {unit_idx}: ({convergence_x:.3f}, {convergence_y:.3f})")
    else:
        print(f"No convergence point found for unit {unit_idx}")
    
    plt.title(f"Unit {unit_idx} - Force Difference Field (Adaptive Scale: {adaptive_scale:.2e})\n"
              f"Mean Diff: {np.mean(force_magnitudes):.3f}, Max Diff: {np.max(force_magnitudes):.3f}",
              fontsize=14)
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim(reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0])
    plt.ylim(reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1])
    
    # Add statistics text box
    if convergence_x is not None and convergence_y is not None:
        textstr = f'Points: {len(x_positions)}\nMean Force: {np.mean(force_magnitudes):.3f}\nMax Force: {np.max(force_magnitudes):.3f}\nConvergence: ({convergence_x:.3f}, {convergence_y:.3f})'
    else:
        textstr = f'Points: {len(x_positions)}\nMean Force: {np.mean(force_magnitudes):.3f}\nMax Force: {np.max(force_magnitudes):.3f}\nNo Convergence Found'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    config.save_plot(f"bizzi_unit_{unit_idx}_force_difference_field.png")
    plt.show()
    
    # PLOT 3: Force Magnitude Distribution 
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.hist(force_magnitudes, bins=20, alpha=0.7, color='purple')
    plt.title(f"Unit {unit_idx} - Force Magnitude Distribution")
    plt.xlabel("Force Magnitude")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(force_magnitudes, 'o-', alpha=0.7, markersize=4)
    plt.title("Force Magnitudes Over Time/Position")
    plt.xlabel("Sample Index")
    plt.ylabel("Force Magnitude")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    config.save_plot(f"bizzi_unit_{unit_idx}_force_magnitude_distribution.png")
    plt.show()
    
    # Print force statistics to check if forces exist
    print(f"\n📊 Unit {unit_idx} Force Statistics:")
    print(f"  Force magnitudes: min={np.min(force_magnitudes):.6f}, max={np.max(force_magnitudes):.6f}")
    print(f"  Mean force magnitude: {np.mean(force_magnitudes):.6f}")
    print(f"  Non-zero forces: {np.sum(force_magnitudes > 1e-6)}/{len(force_magnitudes)}")
    print(f"  X forces range: [{np.min(x_force_diff):.6f}, {np.max(x_force_diff):.6f}]")
    print(f"  Y forces range: [{np.min(y_force_diff):.6f}, {np.max(y_force_diff):.6f}]")

print(f"\n✅ All plots saved to: {config.OUTPUT_DIR}")



