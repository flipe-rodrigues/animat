import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from bizzi_stimulation import BizziAnalyzer

def run_full_unit_test(analyzer, positions=None, max_positions=20):
    """Test ALL 64 units across the workspace."""
    
    print("🔬 TESTING ALL 64 RNN UNITS FOR CONVERGENT FORCE FIELDS")
    print("=" * 50)
    
    # Load positions if not provided
    if positions is None:
        positions = analyzer.load_workspace_positions()
        # Use subset for manageable experiment
        if len(positions) > max_positions:
            indices = np.linspace(0, len(positions)-1, max_positions, dtype=int)
            positions = [positions[i] for i in indices]
    
    print(f"📍 Testing at {len(positions)} positions")
    
    # Test ALL RNN units (not just every 8th)
    all_units = range(64)
    stimulation_strength = 0.05
    
    print(f"🧠 Testing ALL {len(all_units)} units with stimulation={stimulation_strength}")
    print(f"   Total trials: {len(positions) * len(all_units)}")
    
    all_results = []
    total_time = time.time()
    
    for pos_idx, position in enumerate(positions):
        print(f"\n📍 Position {pos_idx+1}/{len(positions)}: [{position[0]:5.2f}, {position[1]:5.2f}]")
        
        # Reset with arm locked at this position
        if not analyzer.reset_with_arm_locked_at_position(position):
            print(f"   ⚠️ Reset failed, skipping position")
            continue
        
        # Get baseline action at this position (only once per position)
        baseline_action = analyzer.get_policy_action(position)
        
        # Test all units at this locked position
        for unit_idx in all_units:
            try:
                # Get stimulated response
                stimulated_action = analyzer.get_stimulated_action(position, unit_idx, stimulation_strength)
                
                # Calculate force response (subtract baseline - this accounts for resting muscle tension)
                force_response = stimulated_action - baseline_action
                
                all_results.append({
                    'position': position.copy(),
                    'unit': unit_idx,
                    'baseline_forces': baseline_action,
                    'stimulated_forces': stimulated_action,
                    'force_response': force_response,
                    'stimulation_strength': stimulation_strength
                })
                
            except Exception as e:
                print(f"   ❌ Unit {unit_idx} failed: {e}")
    
    total_elapsed = time.time() - total_time
    
    print(f"\n🎯 EXPERIMENT COMPLETE!")
    print(f"   Collected {len(all_results)} stimulation trials")
    print(f"   Total time: {total_elapsed:.2f}s")
    print(f"   Trials per second: {len(all_results)/total_elapsed:.1f}")
    
    # Save results
    output_file = "bizzi_results_all_units.json"
    with open(output_file, 'w') as f:
        json_results = []
        for result in all_results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Results saved to {output_file}")
    
    return all_results

def analyze_convergence(results):
    """Quantify convergence of force fields for each unit."""
    
    print("\n🔍 ANALYZING FORCE FIELD CONVERGENCE")
    print("=" * 40)
    
    # Group by units
    units = {}
    for result in results:
        unit = result['unit']
        if unit not in units:
            units[unit] = []
        units[unit].append(result)
    
    # Calculate convergence metrics
    convergence_metrics = {}
    
    for unit_idx, trials in units.items():
        positions = np.array([trial['position'][:2] for trial in trials])
        forces = np.array([trial['force_response'][:2] for trial in trials])
        
        # Skip if not enough data
        if len(positions) < 3:
            continue
        
        # Calculate center of positions
        center = np.mean(positions, axis=0)
        
        # Calculate vectors from center to each position
        vectors_to_center = center - positions  # pointing TO center
        
        # Normalize vectors
        vector_norms = np.linalg.norm(vectors_to_center, axis=1)
        vector_norms[vector_norms == 0] = 1.0  # Avoid division by zero
        normalized_to_center = vectors_to_center / vector_norms.reshape(-1, 1)
        
        # Normalize force vectors
        force_norms = np.linalg.norm(forces, axis=1)
        force_norms[force_norms == 0] = 1.0  # Avoid division by zero
        normalized_forces = forces / force_norms.reshape(-1, 1)
        
        # Calculate alignment (dot product) between normalized vectors
        # 1 = perfect alignment toward center, -1 = directly away from center
        alignments = np.sum(normalized_to_center * normalized_forces, axis=1)
        
        # Calculate convergence metrics
        mean_alignment = np.mean(alignments)
        
        # Weighted by magnitude
        weighted_alignment = np.sum(alignments * force_norms) / np.sum(force_norms)
        
        # Calculate curl (rotational component)
        curl_components = []
        for i, pos in enumerate(positions):
            neighbors = []
            # Find 3 closest neighbors
            distances = np.linalg.norm(positions - pos, axis=1)
            neighbor_indices = np.argsort(distances)[1:4]  # Exclude self
            
            for j in neighbor_indices:
                if distances[j] < 0.5:  # Only use nearby points
                    direction = positions[j] - pos
                    perpendicular = np.array([-direction[1], direction[0]])  # 90° rotation
                    perpendicular = perpendicular / np.linalg.norm(perpendicular)
                    
                    # Project force onto perpendicular direction
                    curl_component = np.dot(forces[i], perpendicular)
                    curl_components.append(curl_component)
        
        mean_curl = np.mean(curl_components) if curl_components else 0
        
        convergence_metrics[unit_idx] = {
            'mean_alignment': mean_alignment,
            'weighted_alignment': weighted_alignment,
            'mean_curl': mean_curl,
            'n_positions': len(positions),
            'mean_force_magnitude': np.mean(force_norms)
        }
        
    # Sort units by convergence (most convergent first)
    sorted_units = sorted(convergence_metrics.items(), 
                         key=lambda x: x[1]['weighted_alignment'], reverse=True)
    
    print(f"📊 Convergence Rankings (positive = convergent, negative = divergent):")
    for i, (unit_idx, metrics) in enumerate(sorted_units[:10]):
        print(f"   {i+1}. Unit {unit_idx:2d}: Alignment = {metrics['weighted_alignment']:6.3f}, "
              f"Curl = {metrics['mean_curl']:6.3f}, Magnitude = {metrics['mean_force_magnitude']:.4f}")
    
    print(f"\n📊 Divergence Rankings (negative = divergent):")
    for i, (unit_idx, metrics) in enumerate(sorted_units[-10:]):
        print(f"   {i+1}. Unit {unit_idx:2d}: Alignment = {metrics['weighted_alignment']:6.3f}, "
              f"Curl = {metrics['mean_curl']:6.3f}, Magnitude = {metrics['mean_force_magnitude']:.4f}")
    
    # Return top convergent and divergent units
    top_convergent = [unit for unit, _ in sorted_units[:5]]
    top_divergent = [unit for unit, _ in sorted_units[-5:]]
    
    return convergence_metrics, top_convergent, top_divergent

def visualize_selected_units(results, selected_units):
    """Visualize force fields for selected units."""
    
    # Group by units
    units = {}
    for result in results:
        unit = result['unit']
        if unit in selected_units:
            if unit not in units:
                units[unit] = []
            units[unit].append(result)
    
    # Create subplot for each selected unit
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Selected RNN Unit Force Fields - Testing All Units', fontsize=16)
    
    axes = axes.flatten()
    
    for i, unit_idx in enumerate(selected_units):
        if i >= len(axes) or unit_idx not in units:
            continue
            
        ax = axes[i]
        trials = units[unit_idx]
        
        # Extract positions and force responses
        positions = np.array([trial['position'][:2] for trial in trials])
        forces = np.array([trial['force_response'][:2] for trial in trials])
        
        # Calculate metrics
        mean_force = np.mean(forces, axis=0)
        magnitude = np.linalg.norm(mean_force)
        center = np.mean(positions, axis=0)
        
        # Plot force vectors
        ax.quiver(positions[:, 0], positions[:, 1],
                 forces[:, 0], forces[:, 1],
                 scale=0.05, angles='xy', scale_units='xy',
                 color='r', alpha=0.7, width=0.003)
        
        # Plot positions
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=30, alpha=0.6)
        
        # Mark field center
        ax.scatter([center[0]], [center[1]], c='green', s=100, marker='*', alpha=0.7)
        
        # Set title and labels
        ax.set_title(f'Unit {unit_idx}\nMean Force: [{mean_force[0]:.3f}, {mean_force[1]:.3f}]\n'
                     f'Magnitude: {magnitude:.4f}')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set consistent axis limits
        ax.set_xlim(-0.5, 0.9)
        ax.set_ylim(-0.9, 0.4)
    
    plt.tight_layout()
    plt.savefig('selected_force_fields.png', dpi=300)
    print(f"📊 Selected unit visualization saved to: selected_force_fields.png")
    
    plt.close()

def main():
    print("🧠 BIZZI FORCE FIELD ANALYSIS - ALL 64 UNITS")
    print("=" * 50)
    
    # Check if results exist
    if os.path.exists("bizzi_results_all_units.json"):
        print("📂 Loading existing results...")
        with open("bizzi_results_all_units.json", 'r') as f:
            results = json.load(f)
    else:
        # Run experiment for all units
        checkpoint_path = "logs/ppo_rnn/policy_good_final.pth"
        analyzer = BizziAnalyzer(checkpoint_path, device='cpu')
        results = run_full_unit_test(analyzer)
    
    # Analyze convergence
    convergence_metrics, top_convergent, top_divergent = analyze_convergence(results)
    
    # Visualize top convergent and divergent units
    selected_units = top_convergent + top_divergent
    print(f"\n🎯 Selected units for visualization:")
    print(f"   Convergent: {top_convergent}")
    print(f"   Divergent: {top_divergent}")
    
    visualize_selected_units(results, selected_units)
    
    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()