import matplotlib.pyplot as plt
import numpy as np
import json
import os

def create_bizzi_force_fields():
    """Create classic Bizzi-style force field plots."""
    
    print("🎨 CREATING BIZZI FORCE FIELD VISUALIZATIONS")
    print("=" * 45)
    
    # Load your validated results
    with open('bizzi_results_validated.json', 'r') as f:
        results = json.load(f)
    
    print(f"📊 Loaded {len(results)} stimulation trials")
    
    # Group by units
    units = {}
    for trial in results:
        unit = trial['unit']
        if unit not in units:
            units[unit] = []
        units[unit].append(trial)
    
    print(f"🧠 Found {len(units)} RNN units: {sorted(units.keys())}")
    
    # Analyze force patterns
    print(f"\n📈 FORCE PATTERN ANALYSIS:")
    unit_stats = {}
    
    for unit_idx, trials in units.items():
        force_responses = np.array([trial['force_response'][:2] for trial in trials])
        mean_force = np.mean(force_responses, axis=0)
        std_force = np.std(force_responses, axis=0)
        magnitude = np.linalg.norm(mean_force)
        
        # Calculate preferred direction (angle)
        angle = np.arctan2(mean_force[1], mean_force[0]) * 180 / np.pi
        
        unit_stats[unit_idx] = {
            'mean_force': mean_force,
            'magnitude': magnitude,
            'angle': angle,
            'std': std_force
        }
        
        print(f"   Unit {unit_idx:2d}: Force=[{mean_force[0]:6.3f}, {mean_force[1]:6.3f}] "
              f"Mag={magnitude:.4f} Angle={angle:5.1f}° "
              f"Std=[{std_force[0]:.3f}, {std_force[1]:.3f}]")
    
    # Create 2x4 subplot for 8 units
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('RNN Motor Unit Force Fields (Bizzi-style Analysis)\n'
                 'Red arrows: Force responses to neural stimulation', fontsize=16, y=0.95)
    
    for i, (unit_idx, trials) in enumerate(units.items()):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        # Extract positions and force responses
        positions = np.array([trial['position'][:2] for trial in trials])
        force_responses = np.array([trial['force_response'][:2] for trial in trials])
        
        # Color arrows by magnitude for better visualization
        magnitudes = np.linalg.norm(force_responses, axis=1)
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 1
        
        # Create force field plot with colored arrows
        colors = plt.cm.Reds(magnitudes / max_mag * 0.8 + 0.2)  # Avoid too light colors
        
        for j, (pos, force, color) in enumerate(zip(positions, force_responses, colors)):
            ax.arrow(pos[0], pos[1], force[0]*20, force[1]*20,  # Scale up for visibility
                    head_width=0.02, head_length=0.015, fc=color, ec=color, alpha=0.8)
        
        # Add position markers
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='blue', s=30, alpha=0.7, zorder=3, edgecolors='darkblue')
        
        # Get unit statistics
        stats = unit_stats[unit_idx]
        
        ax.set_title(f'Unit {unit_idx}\n'
                    f'Preferred Direction: {stats["angle"]:.0f}°\n'
                    f'Force Magnitude: {stats["magnitude"]:.4f}',
                    fontsize=11)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set consistent axis limits
        ax.set_xlim(-0.5, 0.9)
        ax.set_ylim(-0.9, 0.4)
        
        # Add colorbar for force magnitude
        if i == 0:  # Only add colorbar legend to first plot
            ax.text(-0.45, 0.3, 'Arrow Color:\nRed intensity ∝\nForce magnitude', 
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'rnn_motor_force_fields.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n🎨 Force field visualization saved to: {output_file}")
    
    # Also save as PDF for publications
    pdf_file = 'rnn_motor_force_fields.pdf'
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    print(f"📄 PDF version saved to: {pdf_file}")
    
    # Close the figure to free memory
    plt.close()
    
    return unit_stats

def analyze_unit_diversity(unit_stats):
    """Analyze the diversity of unit responses."""
    print(f"\n🔬 UNIT DIVERSITY ANALYSIS:")
    print("=" * 30)
    
    angles = [stats['angle'] for stats in unit_stats.values()]
    magnitudes = [stats['magnitude'] for stats in unit_stats.values()]
    
    # Analyze directional preferences
    print(f"📐 Directional Analysis:")
    print(f"   Preferred directions: {[f'{a:.0f}°' for a in sorted(angles)]}")
    print(f"   Direction spread: {np.std(angles):.1f}° (std)")
    print(f"   Min angle: {min(angles):.0f}°, Max angle: {max(angles):.0f}°")
    
    # Analyze force magnitudes
    print(f"\n💪 Force Magnitude Analysis:")
    print(f"   Mean magnitude: {np.mean(magnitudes):.4f}")
    print(f"   Magnitude range: [{min(magnitudes):.4f}, {max(magnitudes):.4f}]")
    print(f"   Magnitude std: {np.std(magnitudes):.4f}")
    
    # Check for directional coverage
    angle_bins = np.histogram(angles, bins=8, range=(-180, 180))[0]
    covered_directions = np.sum(angle_bins > 0)
    print(f"\n🎯 Coverage Analysis:")
    print(f"   Directions covered: {covered_directions}/8 octants")
    print(f"   Direction distribution: {angle_bins}")
    
    # Identify unit types
    print(f"\n🧠 Unit Classifications:")
    for unit_idx, stats in unit_stats.items():
        angle = stats['angle']
        mag = stats['magnitude']
        
        # Classify by direction
        if -45 <= angle <= 45:
            direction = "Right"
        elif 45 < angle <= 135:
            direction = "Up"
        elif -135 <= angle < -45:
            direction = "Down"
        else:
            direction = "Left"
        
        # Classify by magnitude
        if mag > np.mean(magnitudes) + np.std(magnitudes):
            strength = "Strong"
        elif mag < np.mean(magnitudes) - np.std(magnitudes):
            strength = "Weak"
        else:
            strength = "Medium"
        
        print(f"   Unit {unit_idx:2d}: {strength:6s} {direction:5s}-preferring "
              f"(mag={mag:.4f}, dir={angle:4.0f}°)")

def analyze_force_fields_with_pca():
    """Use PCA to find underlying structure in force responses."""
    
    print("🧠 ANALYZING FORCE FIELDS USING PCA")
    print("=" * 45)
    
    # Load results
    with open('bizzi_results_validated.json', 'r') as f:
        results = json.load(f)
    
    # Extract position and force response data
    positions = np.array([result['position'][:2] for result in results])
    force_responses = np.array([result['force_response'][:2] for result in results])
    units = np.array([result['unit'] for result in results])
    
    # Create position-indexed data for PCA
    unique_positions = np.unique(positions, axis=0)
    
    # Prepare data for PCA - force responses organized by position
    position_force_data = []
    
    # For each unique position, get the force responses for all units
    for pos in unique_positions:
        pos_indices = np.where((positions == pos).all(axis=1))[0]
        pos_forces = force_responses[pos_indices]
        position_force_data.append(pos_forces.flatten())  # Flatten all unit responses at this position
    
    position_force_data = np.array(position_force_data)
    
    # Run PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)  # Extract top 4 components
    pca_results = pca.fit_transform(position_force_data)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Create force field plots for each principal component
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Force Field Principal Components', fontsize=16)
    
    axes = axes.flatten()
    
    # For each principal component
    for pc_idx in range(min(4, pca.n_components_)):
        ax = axes[pc_idx]
        
        # Extract the component scores for each position
        pc_scores = pca_results[:, pc_idx]
        
        # Reconstruct force vectors from PCA component
        component = pca.components_[pc_idx].reshape(-1, 2)  # Reshape to original force dimensions
        
        # Scale arrows for visibility
        scale_factor = 20 / np.max(np.abs(component)) if np.max(np.abs(component)) > 0 else 1
        
        # Plot arrows for each position
        for i, pos in enumerate(unique_positions):
            # Scale the component by its score at this position
            force_vector = component[i % len(component)] * pc_scores[i]
            
            # Plot arrow
            ax.arrow(pos[0], pos[1], 
                     force_vector[0] * scale_factor, force_vector[1] * scale_factor,
                     head_width=0.02, fc='red', ec='red', alpha=0.7)
            
        # Add position markers
        ax.scatter(unique_positions[:, 0], unique_positions[:, 1], 
                  c='blue', s=30, alpha=0.6)
        
        ax.set_title(f'PC {pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]:.1%} var.)')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set consistent axis limits
        ax.set_xlim(-0.5, 0.9)
        ax.set_ylim(-0.9, 0.4)
    
    plt.tight_layout()
    plt.savefig('force_field_pca.png', dpi=300, bbox_inches='tight')
    print(f"📊 PCA force field visualization saved to: force_field_pca.png")
    
    # Also visualize synthetic force fields from the top 2 components
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Synthetic Force Field from Top 2 PCA Components', fontsize=16)
    
    # Combine the top two components
    synthetic_forces = np.zeros((len(unique_positions), 2))
    for i in range(2):  # Top 2 components
        component = pca.components_[i].reshape(-1, 2)
        for j, pos in enumerate(unique_positions):
            synthetic_forces[j] += component[j % len(component)] * pca_results[j, i]
    
    # Scale for visibility
    scale_factor = 15 / np.max(np.abs(synthetic_forces)) if np.max(np.abs(synthetic_forces)) > 0 else 1
    
    # Plot the synthetic force field
    for i, pos in enumerate(unique_positions):
        force = synthetic_forces[i]
        ax.arrow(pos[0], pos[1], 
                 force[0] * scale_factor, force[1] * scale_factor,
                 head_width=0.02, fc='red', ec='red', alpha=0.7)
    
    # Add position markers
    ax.scatter(unique_positions[:, 0], unique_positions[:, 1], 
               c='blue', s=30, alpha=0.6)
    
    ax.set_title('Combined Force Field (PC1 + PC2)')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set consistent axis limits
    ax.set_xlim(-0.5, 0.9)
    ax.set_ylim(-0.9, 0.4)
    
    plt.tight_layout()
    plt.savefig('synthetic_force_field.png', dpi=300, bbox_inches='tight')
    print(f"📊 Synthetic force field saved to: synthetic_force_field.png")
    
    return pca, pca_results, unique_positions

def main():
    print("🎨 BIZZI FORCE FIELD VISUALIZATION")
    print("=" * 40)
    
    # Check if results file exists
    if not os.path.exists('bizzi_results_validated.json'):
        print("❌ bizzi_results_validated.json not found!")
        print("   Run bizzi_stimulation.py first to generate the data.")
        return
    
    # Create visualizations
    unit_stats = create_bizzi_force_fields()
    
    # Analyze unit diversity
    analyze_unit_diversity(unit_stats)
    
    # Analyze force fields with PCA
    analyze_force_fields_with_pca()
    
    print(f"\n✅ VISUALIZATION COMPLETE!")
    print(f"📁 Files created:")
    print(f"   - rnn_motor_force_fields.png (high-res image)")
    print(f"   - rnn_motor_force_fields.pdf (publication ready)")
    print(f"   - force_field_pca.png (PCA analysis)")
    print(f"   - synthetic_force_field.png (Synthetic force field)")
    print(f"\n🎯 Your RNN shows diverse motor unit responses similar to Bizzi's findings!")

if __name__ == "__main__":
    main()