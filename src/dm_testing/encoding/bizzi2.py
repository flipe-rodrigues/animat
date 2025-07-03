#%%
import os
import matplotlib.pyplot as plt
from plants import SequentialReacher
from environments import SequentialReachingEnv
from utils import *
import numpy as np
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

# Zero out encoded target influence for stimulation
print("Zeroing out encoded target influence for stimulation...")
best_rnn.policy.W_in.data[:, 12:] = 0  

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

# Process each stimulation unit
for unit_idx in config.STIMULATION_UNITS:
    print(f"\nProcessing unit {unit_idx}...")
    
    # Stimulate the unit
    force_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
    )
    
    # Extract data
    position_vecs = np.nan_to_num(np.array(force_data["position"]))
    force_vecs = np.nan_to_num(np.array(force_data["force"]))
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

        # During rest collection
        if t - 0.5 - time_window < reacher.data.time <= t - 0.5:
            # This should show zero muscle activation during rest
            print(f"Rest period muscle activation: {reacher.data.ctrl}")
    # Convert to arrays
    positions = np.array(average_positions)
    rest_forces = np.array(rest_average_forces)
    stim_forces = np.array(stim_average_forces)
    
    # Extract components
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    x_stim_forces = stim_forces[:, 0]
    y_stim_forces = stim_forces[:, 1]
    x_rest_forces = rest_forces[:, 0]
    y_rest_forces = rest_forces[:, 1]
    
    # Calculate difference
    x_force_diff = x_stim_forces - x_rest_forces
    y_force_diff = y_stim_forces - y_rest_forces
    force_magnitudes = np.sqrt(x_force_diff**2 + y_force_diff**2)
    
    # Calculate zoom area for better visualization
    x_center = np.mean(x_positions)
    y_center = np.mean(y_positions)
    x_range = np.max(x_positions) - np.min(x_positions)
    y_range = np.max(y_positions) - np.min(y_positions)
    
    # Zoom to 70% of the data range for better detail
    zoom_factor = 0.7
    x_zoom_range = x_range * zoom_factor
    y_zoom_range = y_range * zoom_factor
    
    x_zoom_min = x_center - x_zoom_range / 2
    x_zoom_max = x_center + x_zoom_range / 2
    y_zoom_min = y_center - y_zoom_range / 2
    y_zoom_max = y_center + y_zoom_range / 2
    
    # PLOT 1: Stimulated vs Rest Forces (ZOOMED)
    plt.figure(figsize=(10, 10))
    
    plt.quiver(
        x_positions, y_positions, x_stim_forces, y_stim_forces,
        angles="xy", scale_units="xy", scale=500,  
        linewidth=3, color="red", alpha=0.8, label="Stimulated", 
        width=0.003, headwidth=2, headlength=3  
    )
    
    plt.quiver(
        x_positions, y_positions, x_rest_forces, y_rest_forces,
        angles="xy", scale_units="xy", scale=500,  
        linewidth=2, color="black", alpha=0.6, label="Rest", 
        width=0.002, headwidth=2, headlength=3  
    )
    
    plt.title(f"Unit {unit_idx} - Stimulated vs Rest Forces (Zoomed)", fontsize=16, fontweight='bold')
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    
    # Apply zoom limits
    plt.xlim(x_zoom_min, x_zoom_max)
    plt.ylim(y_zoom_min, y_zoom_max)
    
    plt.tight_layout()
    
    config.save_plot(f"bizzi_unit_{unit_idx}_stimulated_vs_rest_zoomed.png")
    plt.show()
    
    # PLOT 2: Force Difference Field (WITH GRID)
    plt.figure(figsize=(10, 10))
    
    # Filter out weak forces
    force_threshold = 1e-6
    strong_force_mask = force_magnitudes > force_threshold
    
    if np.sum(strong_force_mask) > 0:
        # Filter arrays to only include strong forces
        x_pos_filtered = x_positions[strong_force_mask]
        y_pos_filtered = y_positions[strong_force_mask]
        x_diff_filtered = x_force_diff[strong_force_mask]
        y_diff_filtered = y_force_diff[strong_force_mask]
        magnitudes_filtered = force_magnitudes[strong_force_mask]
        
        max_force = np.max(magnitudes_filtered)
        
        # Calculate adaptive scale
        if max_force > 0:
            plot_width = np.max(x_pos_filtered) - np.min(x_pos_filtered)
            target_arrow_size = plot_width * 0.1
            adaptive_scale = max_force / target_arrow_size
        else:
            adaptive_scale = 1
        
        # Plot force difference field
        quiver = plt.quiver(
            x_pos_filtered, y_pos_filtered, x_diff_filtered, y_diff_filtered,
            magnitudes_filtered,  # Color by magnitude
            angles="xy", scale_units="xy", scale=adaptive_scale,
            linewidth=2.5, alpha=0.9, cmap='plasma', 
            width=0.008, headwidth=4, headlength=5
        )
        
        # Add colorbar
        cbar = plt.colorbar(quiver, shrink=0.8)
        cbar.set_label('Force Magnitude', fontsize=12)
        
        # Calculate convergence point
        convergence_x, convergence_y = calculate_convergence_point_intersection(
            x_pos_filtered, y_pos_filtered, x_diff_filtered, y_diff_filtered
        )
        
        if convergence_x is not None and convergence_y is not None:
            plt.plot(convergence_x, convergence_y, 'o', markersize=15,  
                    markerfacecolor='cyan', markeredgecolor='black', 
                    markeredgewidth=3, label='Convergence Point')
            plt.legend(fontsize=12)
    
    plt.title(f"Unit {unit_idx} - Force Difference Field", fontsize=16, fontweight='bold')
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.grid(True, alpha=0.3)  # ADDED GRID
    plt.axis("equal")
    plt.xlim(reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0])
    plt.ylim(reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1])
    plt.tight_layout()
    
    config.save_plot(f"bizzi_unit_{unit_idx}_force_difference_field.png")
    plt.show()

print(f"\n✅ All plots saved to: {config.OUTPUT_DIR}")