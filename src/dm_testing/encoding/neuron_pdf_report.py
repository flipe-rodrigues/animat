import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
import numpy as np
from plants import SequentialReacher
from environments import SequentialReachingEnv
from rnn_adapter import RNNAdapter, config

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

def evaluate_convergence_like_stimulation_bizzi(x_positions, y_positions, x_forces, y_forces):
    """
    Evaluate convergence quality using the same method as stimulation_bizzi.py
    but keep intersection-based convergence point calculation
    
    Returns:
        dict: Contains convergence analysis results
    """
    # Calculate force magnitudes
    force_magnitudes = np.sqrt(x_forces**2 + y_forces**2)
    
    # Only consider positions where there's significant force (above median)
    significant_force_mask = force_magnitudes > np.percentile(force_magnitudes, 50)
    
    if np.sum(significant_force_mask) <= 3:  # Need at least 3 points for meaningful analysis
        return {
            'is_convergent': False,
            'convergence_score': 0,
            'combined_score': 0,
            'convergence_point': (np.mean(x_positions), np.mean(y_positions)),
            'mean_force_magnitude': np.mean(force_magnitudes),
            'max_force_magnitude': np.max(force_magnitudes),
            'reason': 'insufficient_significant_forces'
        }
    
    # Method 1: Calculate intersection-based convergence point (keep original method)
    convergence_x, convergence_y = calculate_convergence_point_intersection(
        x_positions, y_positions, x_forces, y_forces
    )
    
    if convergence_x is None or convergence_y is None:
        return {
            'is_convergent': False,
            'convergence_score': 0,
            'combined_score': 0,
            'convergence_point': (np.mean(x_positions), np.mean(y_positions)),
            'mean_force_magnitude': np.mean(force_magnitudes),
            'max_force_magnitude': np.max(force_magnitudes),
            'reason': 'no_intersections_found'
        }
    
    # Method 2: Evaluate convergence quality using stimulation_bizzi approach
    # Calculate how well forces point toward the intersection-based convergence point
    weights = force_magnitudes[significant_force_mask]
    
    convergence_vectors = np.column_stack([
        convergence_x - x_positions[significant_force_mask],
        convergence_y - y_positions[significant_force_mask]
    ])
    force_vectors = np.column_stack([
        x_forces[significant_force_mask],
        y_forces[significant_force_mask]
    ])
    
    # Normalize vectors
    convergence_vectors_norm = convergence_vectors / (np.linalg.norm(convergence_vectors, axis=1, keepdims=True) + 1e-8)
    force_vectors_norm = force_vectors / (np.linalg.norm(force_vectors, axis=1, keepdims=True) + 1e-8)
    
    # Calculate alignment (dot product) - how well forces point toward convergence
    alignment = np.sum(convergence_vectors_norm * force_vectors_norm, axis=1)
    
    # Convergence score: mean alignment weighted by force magnitude (stimulation_bizzi method)
    convergence_score = np.average(alignment, weights=weights)
    
    # Additional metrics from stimulation_bizzi
    force_consistency = np.std(force_magnitudes[significant_force_mask])  # Lower = more consistent
    spatial_spread = np.std(np.column_stack([x_positions[significant_force_mask], y_positions[significant_force_mask]]), axis=0).mean()
    
    # Combined convergence score (higher = better convergence)
    combined_score = convergence_score * np.mean(weights) / (force_consistency + 1e-8)
    
    # Determine if convergent based on stimulation_bizzi thresholds
    is_convergent = (convergence_score > 0.3 and  # At least 30% alignment
                    combined_score > 0.01 and     # Reasonable combined score
                    np.mean(weights) > 1e-6)       # Sufficient force magnitude
    
    return {
        'is_convergent': is_convergent,
        'convergence_score': convergence_score,
        'raw_alignment': convergence_score,
        'combined_score': combined_score,
        'convergence_point': (convergence_x, convergence_y),  # Intersection-based!
        'mean_force_magnitude': np.mean(force_magnitudes),
        'max_force_magnitude': np.max(force_magnitudes),
        'force_consistency': force_consistency,
        'spatial_spread': spatial_spread,
        'significant_force_count': np.sum(significant_force_mask),
        'weights_mean': np.mean(weights)
    }

def generate_neuron_page(unit_idx, env, best_rnn, weights_input, weights_hidden, weights_output, 
                        bias_hidden, total_abs_input_weights, total_abs_output_weights, 
                        total_abs_hidden_weights, reacher, pdf_pages):
    """Generate a single page with 4 plots for one neuron"""
    
    print(f"Processing unit {unit_idx}...")
    
    # Stimulate the unit to get force data
    force_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
    )
    
    # Process the data
    position_vecs = np.nan_to_num(np.array(force_data["position"]))
    force_vecs = np.nan_to_num(np.array(force_data["force"]))
    time = np.linspace(0, reacher.data.time, len(force_vecs))
    time_window = 0.1
    
    # Calculate averaged forces
    average_positions = []
    rest_average_forces = []
    stim_average_forces = []
    
    for t in range(1, int(reacher.data.time) + 1):
        # Rest period
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
    
    # Convert to arrays
    positions = np.array(average_positions)
    rest_forces = np.array(rest_average_forces)
    stim_forces = np.array(stim_average_forces)
    
    # Extract x,y components
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    x_rest_forces = rest_forces[:, 0]
    y_rest_forces = rest_forces[:, 1]
    x_stim_forces = stim_forces[:, 0]
    y_stim_forces = stim_forces[:, 1]
    x_force_diff = x_stim_forces - x_rest_forces
    y_force_diff = y_stim_forces - y_rest_forces
    
    # CALCULATE UNIFIED SCALE FOR PLOTS 1 & 2 ONLY
    # Find the maximum force magnitude for rest and stimulated forces
    rest_magnitudes = np.sqrt(x_rest_forces**2 + y_rest_forces**2)
    stim_magnitudes = np.sqrt(x_stim_forces**2 + y_stim_forces**2)
    diff_magnitudes = np.sqrt(x_force_diff**2 + y_force_diff**2)
    
    max_rest_force = np.max(rest_magnitudes)
    max_stim_force = np.max(stim_magnitudes)
    max_diff_force = np.max(diff_magnitudes)
    
    # Use the maximum force between rest and stimulated for unified scaling (plots 1 & 2)
    unified_max_force = max(max_rest_force, max_stim_force)
    
    if unified_max_force > 0:
        plot_width = np.max(x_positions) - np.min(x_positions)
        target_arrow_size = plot_width * 0.1
        unified_scale = unified_max_force / target_arrow_size
    else:
        unified_scale = 200  # Fallback scale
    
    # Calculate adaptive scale for plot 3 (force differences)
    if max_diff_force > 0:
        adaptive_scale = max_diff_force / target_arrow_size
    else:
        adaptive_scale = 1
    
    # Calculate scale ratio for plot 3 title - FIX THE RATIO CALCULATION
    # The ratio should show how much SMALLER the adaptive scale is compared to unified
    # OR how much LARGER the arrows appear in plot 3
    if unified_scale > 0 and adaptive_scale > 0:
        # This shows how much larger the unified scale is compared to adaptive scale
        # Which means arrows in plot 3 appear this many times LARGER than in plots 1&2
        scale_ratio = unified_scale / adaptive_scale
    else:
        scale_ratio = 1
    
    print(f"  Unified scale (plots 1&2): {unified_scale:.1f}, Adaptive scale (plot 3): {adaptive_scale:.1f}, Ratio: {scale_ratio:.1f}x")
    
    # CALCULATE CONVERGENCE ANALYSIS FOR ALL THREE FORCE TYPES
    rest_convergence = evaluate_convergence_like_stimulation_bizzi(
        x_positions, y_positions, x_rest_forces, y_rest_forces
    )
    stim_convergence = evaluate_convergence_like_stimulation_bizzi(
        x_positions, y_positions, x_stim_forces, y_stim_forces
    )
    diff_convergence = evaluate_convergence_like_stimulation_bizzi(
        x_positions, y_positions, x_force_diff, y_force_diff
    )
    
    # Create figure with 4 subplots and BIG TITLE
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # ADD BIG TITLE FOR THE ENTIRE FIGURE
    fig.suptitle(f'Unit {unit_idx} Analysis', fontsize=18, weight='bold', y=1.1)
    
    # Common plotting parameters
    xlim = [reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0]]
    ylim = [reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1]]
    
    # Plot 1: Rest Forces - USING UNIFIED SCALE
    ax1 = axes[0]
    if max_rest_force > 1e-8:
        quiver1 = ax1.quiver(
            x_positions, y_positions, x_rest_forces, y_rest_forces,
            angles="xy", scale_units="xy", scale=unified_scale,  # UNIFIED SCALE
            linewidth=2, color="black", alpha=0.8,
            width=0.004, headwidth=3, headlength=4
        )
        
        # Add convergence point if convergent
        if rest_convergence['is_convergent']:
            conv_x, conv_y = rest_convergence['convergence_point']
            ax1.plot(conv_x, conv_y, 'o', markersize=8,
                    markerfacecolor='cyan', markeredgecolor='black',
                    markeredgewidth=2, label='Convergence Point')
    
    # Create title with convergence info
    rest_conv_info = ""
    if rest_convergence['is_convergent']:
        rest_conv_info = f"\n✓ Conv: {rest_convergence['convergence_score']:.2f}"
    else:
        rest_conv_info = f"\n✗ Score: {rest_convergence['convergence_score']:.2f}"
    
    ax1.set_title(f"Rest Forces\nMax: {max_rest_force:.2e}{rest_conv_info}", fontsize=11, weight='bold')
    ax1.set_xlabel("X Position (m)", fontsize=10)
    ax1.set_ylabel("Y Position (m)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect('equal')
    
    # Plot 2: Stimulated Forces - USING UNIFIED SCALE
    ax2 = axes[1]
    if max_stim_force > 1e-8:
        quiver2 = ax2.quiver(
            x_positions, y_positions, x_stim_forces, y_stim_forces,
            angles="xy", scale_units="xy", scale=unified_scale,  # UNIFIED SCALE
            linewidth=2, color="red", alpha=0.8,
            width=0.004, headwidth=3, headlength=4
        )
        
        # Add convergence point if convergent
        if stim_convergence['is_convergent']:
            conv_x, conv_y = stim_convergence['convergence_point']
            ax2.plot(conv_x, conv_y, 'o', markersize=8,
                    markerfacecolor='cyan', markeredgecolor='black',
                    markeredgewidth=2, label='Convergence Point')
    
    # Create title with convergence info
    stim_conv_info = ""
    if stim_convergence['is_convergent']:
        stim_conv_info = f"\n✓ Conv: {stim_convergence['convergence_score']:.2f}"
    else:
        stim_conv_info = f"\n✗ Score: {stim_convergence['convergence_score']:.2f}"
    
    ax2.set_title(f"Stimulated Forces\nMax: {max_stim_force:.2e}{stim_conv_info}", fontsize=11, weight='bold')
    ax2.set_xlabel("X Position (m)", fontsize=10)
    ax2.set_ylabel("Y Position (m)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal')
    
    # Plot 3: Force Differences - USING ADAPTIVE SCALE
    ax3 = axes[2]
    diff_conv_info = ""  # Will hold convergence status for title
    
    if max_diff_force > 1e-8:  # Only plot if there are meaningful forces
        quiver3 = ax3.quiver(
            x_positions, y_positions, x_force_diff, y_force_diff,
            diff_magnitudes,  # Color by magnitude
            angles="xy", scale_units="xy", scale=adaptive_scale,  # ADAPTIVE SCALE
            linewidth=2, alpha=0.9, cmap='plasma',
            width=0.006, headwidth=4, headlength=5
        )
        
        # ADD COLORBAR FOR FORCE MAGNITUDES - FIX THE SIZE ISSUE
        # Create the colorbar with specific positioning to maintain plot size
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(quiver3, cax=cax)
        cbar.set_label('Force Magnitude', fontsize=8)
        
        # Add intersection-based convergence point ONLY if field passes stimulation_bizzi convergence test
        if diff_convergence['is_convergent']:
            conv_x, conv_y = diff_convergence['convergence_point']  # This is intersection-based
            ax3.plot(conv_x, conv_y, 'o', markersize=10,
                    markerfacecolor='cyan', markeredgecolor='black',
                    markeredgewidth=2, label='Convergence Point')
            
            # Add convergence quality info to title
            conv_score = diff_convergence['convergence_score']
            diff_conv_info = f"\n✓ Conv: {conv_score:.2f}"
        else:
            # Show why it's not convergent
            conv_score = diff_convergence['convergence_score']
            diff_conv_info = f"\n✗ Score: {conv_score:.2f}"
    else:
        # If no forces, just show empty plot with message
        ax3.text(0.5, 0.5, 'No Forces\nDetected', transform=ax3.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        diff_conv_info = "\nNo forces detected"
    
    # Add scale ratio information to title
    if scale_ratio > 1.1:
        scale_info = f"Arrows {scale_ratio:.1f}x larger"
    elif scale_ratio < 0.9:
        scale_info = f"Arrows {1/scale_ratio:.1f}x smaller"
    else:
        scale_info = f"Same scale"
            
    ax3.set_title(f"Force Differences\nMax: {max_diff_force:.2e}\n{scale_info}{diff_conv_info}", 
                  fontsize=10, weight='bold')
    ax3.set_xlabel("X Position (m)", fontsize=10)
    ax3.set_ylabel("Y Position (m)", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect('equal')
    
    # Plot 4: Weight Distribution Plot - SUBTITLE ONLY
    ax4 = axes[3]
    create_weight_distribution_plot(ax4, unit_idx, total_abs_input_weights, 
                                   total_abs_output_weights, total_abs_hidden_weights)
    
    
    # Adjust subplot spacing to make room for the big title
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # More room for bigger titles
    
    # Save to PDF
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Print convergence summary for this unit
    print(f"  Unit {unit_idx} Convergence:")
    print(f"    Rest: {'✓' if rest_convergence['is_convergent'] else '✗'} {rest_convergence['convergence_score']:.3f}")
    print(f"    Stim: {'✓' if stim_convergence['is_convergent'] else '✗'} {stim_convergence['convergence_score']:.3f}")
    print(f"    Diff: {'✓' if diff_convergence['is_convergent'] else '✗'} {diff_convergence['convergence_score']:.3f}")

def create_weight_distribution_plot(ax, unit_idx, total_abs_input_weights, total_abs_output_weights, total_abs_hidden_weights):
    """Create weight distribution plot with vertical lines showing current neuron's position"""
    ax.clear()
    
    # Set up the plot with 3 vertical sections - MORE COMPACT
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.5)  # Reduced from 3 to 2.5
    
    # Get current neuron's values
    current_input = total_abs_input_weights[unit_idx]
    current_output = total_abs_output_weights[unit_idx] 
    current_hidden = total_abs_hidden_weights[unit_idx]
    
    # Add title - REMOVED UNIT NUMBER since it's now in the big title
    ax.text(0.5, 2.4, 'Weight Distributions', ha='center', va='center', 
            fontsize=12, weight='bold')  # Smaller font since it's now a subtitle
    
    # 1. INPUT WEIGHTS (top section: y = 1.8 to 2.1) - MOVED DOWN
    x_input = np.linspace(np.min(total_abs_input_weights), np.max(total_abs_input_weights), 100)
    kde_input = gaussian_kde(total_abs_input_weights)
    y_input_kde = kde_input(x_input)
    
    # Normalize x to 0-1 range for plotting
    x_input_norm = (x_input - np.min(total_abs_input_weights)) / (np.max(total_abs_input_weights) - np.min(total_abs_input_weights))
    # Normalize y and shift to top section - MORE COMPACT
    y_input_norm = 1.95 + 0.15 * (y_input_kde / np.max(y_input_kde))  # Reduced height
    
    ax.plot(x_input_norm, y_input_norm, color='green', linewidth=3, alpha=0.8, label='Input Weights')
    
    # Add vertical line for current neuron
    current_input_norm = (current_input - np.min(total_abs_input_weights)) / (np.max(total_abs_input_weights) - np.min(total_abs_input_weights))
    ax.axvline(current_input_norm, ymin=1.85/2.5, ymax=2.15/2.5, color='green', linestyle='--', linewidth=3, alpha=0.9)
    
    # Add text annotation
    ax.text(current_input_norm, 2.15, f'{current_input:.2f}', 
            ha='center', va='bottom', fontsize=9, color='green', weight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='lightgreen', alpha=0.8))
    
    # Add label
    ax.text(0.5, 1.75, 'Input Weights', ha='center', va='center', fontsize=11, weight='bold', color='green')
    
    # 2. HIDDEN WEIGHTS (middle section: y = 1.0 to 1.3) - MORE COMPACT
    x_hidden = np.linspace(np.min(total_abs_hidden_weights), np.max(total_abs_hidden_weights), 100)
    kde_hidden = gaussian_kde(total_abs_hidden_weights)
    y_hidden_kde = kde_hidden(x_hidden)
    
    # Normalize x to 0-1 range for plotting
    x_hidden_norm = (x_hidden - np.min(total_abs_hidden_weights)) / (np.max(total_abs_hidden_weights) - np.min(total_abs_hidden_weights))
    # Normalize y and shift to middle section - MORE COMPACT
    y_hidden_norm = 1.15 + 0.15 * (y_hidden_kde / np.max(y_hidden_kde))  # Reduced height
    
    ax.plot(x_hidden_norm, y_hidden_norm, color='orange', linewidth=3, alpha=0.8, label='Hidden Weights')
    
    # Add vertical line for current neuron
    current_hidden_norm = (current_hidden - np.min(total_abs_hidden_weights)) / (np.max(total_abs_hidden_weights) - np.min(total_abs_hidden_weights))
    ax.axvline(current_hidden_norm, ymin=1.05/2.5, ymax=1.35/2.5, color='orange', linestyle='--', linewidth=3, alpha=0.9)
    
    # Add text annotation
    ax.text(current_hidden_norm, 1.35, f'{current_hidden:.2f}', 
            ha='center', va='bottom', fontsize=9, color='orange', weight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow', alpha=0.8))
    
    # Add label
    ax.text(0.5, 0.95, 'Hidden Weights', ha='center', va='center', fontsize=11, weight='bold', color='orange')
    
    # 3. OUTPUT WEIGHTS (bottom section: y = 0.2 to 0.5) - MORE COMPACT
    x_output = np.linspace(np.min(total_abs_output_weights), np.max(total_abs_output_weights), 100)
    kde_output = gaussian_kde(total_abs_output_weights)
    y_output_kde = kde_output(x_output)
    
    # Normalize x to 0-1 range for plotting
    x_output_norm = (x_output - np.min(total_abs_output_weights)) / (np.max(total_abs_output_weights) - np.min(total_abs_output_weights))
    # Normalize y and shift to bottom section - MORE COMPACT
    y_output_norm = 0.35 + 0.15 * (y_output_kde / np.max(y_output_kde))  # Reduced height
    
    ax.plot(x_output_norm, y_output_norm, color='blue', linewidth=3, alpha=0.8, label='Output Weights')
    
    # Add vertical line for current neuron
    current_output_norm = (current_output - np.min(total_abs_output_weights)) / (np.max(total_abs_output_weights) - np.min(total_abs_output_weights))
    ax.axvline(current_output_norm, ymin=0.25/2.5, ymax=0.55/2.5, color='blue', linestyle='--', linewidth=3, alpha=0.9)
    
    # Add text annotation
    ax.text(current_output_norm, 0.55, f'{current_output:.2f}', 
            ha='center', va='bottom', fontsize=9, color='blue', weight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='lightblue', alpha=0.8))
    
    # Add label
    ax.text(0.5, 0.15, 'Output Weights', ha='center', va='center', fontsize=11, weight='bold', color='blue')
    
    # Add range information on the right - ADJUSTED POSITIONS
    ax.text(1.02, 1.95, f'[{np.min(total_abs_input_weights):.1f}, {np.max(total_abs_input_weights):.1f}]', 
            ha='left', va='center', fontsize=7, color='green', transform=ax.transData)
    ax.text(1.02, 1.15, f'[{np.min(total_abs_hidden_weights):.1f}, {np.max(total_abs_hidden_weights):.1f}]', 
            ha='left', va='center', fontsize=7, color='orange', transform=ax.transData)
    ax.text(1.02, 0.35, f'[{np.min(total_abs_output_weights):.1f}, {np.max(total_abs_output_weights):.1f}]', 
            ha='left', va='center', fontsize=7, color='blue', transform=ax.transData)
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def main():
    """Main function to generate the neuron PDF report"""
    
    # Initialize environment and model
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
    
    # Load model
    best_rnn = RNNAdapter()
    
    # Extract weights
    weights_input = best_rnn.W_in
    weights_hidden = best_rnn.W_h
    weights_output = best_rnn.W_out
    bias_hidden = best_rnn.b_h
    
    # Calculate weight statistics (needed for the distribution plot)
    total_abs_output_weights = np.sum(np.abs(weights_output), axis=0)  # Sum over output dimensions -> (64,)
    total_abs_input_weights = np.sum(np.abs(weights_input), axis=1)    # Sum over input dimensions -> (64,)
    total_abs_hidden_weights = np.sum(np.abs(weights_hidden), axis=1)  # Sum over hidden dimensions -> (64,)
    
    # Zero out encoded target influence for stimulation
    print("Zeroing out encoded target influence for stimulation...")
    best_rnn.policy.W_in.data[:, 12:] = 0
    
    # Setup output directory and PDF file
    config.setup_output_dir()
    pdf_filename = os.path.join(config.OUTPUT_DIR, "neuron_analysis_report_complete.pdf")
    
    print(f"Generating PDF report: {pdf_filename}")
    
    # NOW ITERATE THROUGH SELECTED NEURONS
    all_units = list(range(config.HIDDEN_SIZE))  # All output neurons
    print(f"Processing {len(all_units)} neurons: {all_units}")
    
    with PdfPages(pdf_filename) as pdf_pages:
        for unit_idx in all_units:
            generate_neuron_page(unit_idx, env, best_rnn, weights_input, 
                                weights_hidden, weights_output, bias_hidden,
                                total_abs_input_weights, total_abs_output_weights,
                                total_abs_hidden_weights, reacher, pdf_pages)
    
    print(f"✅ PDF report generated: {pdf_filename}")
    print(f"📄 Contains {len(all_units)} neuron pages")

if __name__ == "__main__":
    main()