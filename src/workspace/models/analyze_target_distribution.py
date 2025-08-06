import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering for headless systems

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Fix the path setup - go to workspace directory
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent  # Go up from models/ to workspace/
sys.path.insert(0, str(workspace_root))

from envs.dm_env import make_arm_env
from wrappers.rl_wrapper import set_seeds, create_env

def analyze_target_distribution(num_test_episodes=100,
                               model_path="arm_final2", 
                               vecnorm_path="vec_normalize2.pkl",
                               success_threshold=0.06,
                               episode_duration_limit=50):
    """Analyze target distribution and success patterns."""
    
    print(f"🎯 ANALYZING TARGET DISTRIBUTION")
    print("=" * 50)
    print(f"Testing {num_test_episodes} episodes...")
    
    # Load the trained SAC model
    print(f"Loading SAC model from {model_path}...")
    model = SAC.load(model_path)
    print(f"✅ SAC model loaded successfully")
    
    # Create encoder (same as training)
    from encoders.encoders import ModalitySpecificEncoder
    encoder = ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)
    print(f"✅ Created {type(encoder).__name__}: {encoder.input_dim}D → {encoder.output_dim}D")
    
    # Load normalization stats
    print(f"Loading normalization stats from {vecnorm_path}...")
    dummy_env = create_env(random_seed=42, encoder=encoder)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_vec)
    
    obs_mean = vec_normalize.obs_rms.mean
    obs_std = np.sqrt(vec_normalize.obs_rms.var + vec_normalize.epsilon)
    dummy_vec.close()
    
    print(f"✅ Loaded normalization stats")
    
    # Data collection
    all_targets = []
    successful_targets = []
    failed_targets = []
    episode_results = []
    
    print(f"\n🧪 Testing episodes...")
    
    for episode in range(num_test_episodes):
        if episode % 20 == 0:
            print(f"  Progress: {episode}/{num_test_episodes} ({episode/num_test_episodes*100:.1f}%)")
        
        seed = 1000 + episode  # Use consistent seeds
        set_seeds(seed)
        
        # Create environment
        dm_env = make_arm_env(random_seed=seed)
        raw_env = create_env(random_seed=seed, encoder=encoder, base_env=dm_env)
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        
        # Get initial target position
        timestep = dm_env.reset()  # Reset to get initial state
        target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos.copy()

        #print(f"Debug - Episode {episode}: target_pos = {target_pos}")  # Debug line

        target_xy = target_pos[:2]  # Just X, Y coordinates
        target_distance_from_origin = np.linalg.norm(target_xy)
        target_angle = np.arctan2(target_xy[1], target_xy[0])  # Angle from positive X axis
        
        # Debug print to verify targets are different
        if episode < 5:  # Only print first few
            print(f"  Target {episode}: pos=({target_xy[0]:.3f}, {target_xy[1]:.3f}), dist={target_distance_from_origin:.3f}")
        
        all_targets.append({
            'x': target_xy[0],
            'y': target_xy[1], 
            'distance': target_distance_from_origin,
            'angle_rad': target_angle,
            'angle_deg': np.degrees(target_angle),
            'episode': episode,
            'seed': seed
        })
        
        # Run episode
        obs = eval_vec.reset()
        obs = (obs - obs_mean[None, :]) / obs_std[None, :]
        obs = np.clip(obs, -vec_normalize.clip_obs, vec_normalize.clip_obs)
        
        done = False
        step_count = 0
        min_distance = float('inf')
        final_distance = float('inf')
        
        while not done and step_count < episode_duration_limit:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_vec.step(action)
            
            obs = (obs - obs_mean[None, :]) / obs_std[None, :]
            obs = np.clip(obs, -vec_normalize.clip_obs, vec_normalize.clip_obs)
            
            # Track distances
            hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
            current_target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
            distance = np.linalg.norm(hand_pos - current_target_pos)
            
            min_distance = min(min_distance, distance)
            final_distance = distance
            step_count += 1
            done = done[0] if isinstance(done, np.ndarray) else done
        
        # Determine success
        is_success = final_distance < success_threshold
        
        episode_result = {
            'episode': episode,
            'target_x': target_xy[0],
            'target_y': target_xy[1],
            'target_distance': target_distance_from_origin,
            'target_angle_deg': np.degrees(target_angle),
            'final_distance': final_distance,
            'min_distance': min_distance,
            'success': is_success,
            'steps': step_count
        }
        
        episode_results.append(episode_result)
        
        if is_success:
            successful_targets.append(all_targets[-1])
        else:
            failed_targets.append(all_targets[-1])
        
        eval_vec.close()
    
    # Analysis and visualization
    print(f"\n📊 ANALYSIS RESULTS:")
    print("=" * 30)
    
    total_episodes = len(episode_results)
    successful_episodes = len(successful_targets)
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
    
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {success_rate:.1%}")
    
    if successful_episodes == 0:
        print("❌ No successful episodes found!")
        return episode_results, None
    
    # Convert to numpy arrays for analysis
    all_targets_array = np.array([[t['x'], t['y']] for t in all_targets])
    successful_targets_array = np.array([[t['x'], t['y']] for t in successful_targets])
    failed_targets_array = np.array([[t['x'], t['y']] for t in failed_targets]) if failed_targets else np.array([]).reshape(0, 2)
    
    # Distance analysis
    successful_distances = [t['distance'] for t in successful_targets]
    failed_distances = [t['distance'] for t in failed_targets]
    
    print(f"\n🎯 TARGET DISTANCE ANALYSIS:")
    print(f"All targets - mean distance: {np.mean([t['distance'] for t in all_targets]):.3f}")
    if successful_distances:
        print(f"Successful - mean distance: {np.mean(successful_distances):.3f} ± {np.std(successful_distances):.3f}")
        print(f"Successful - distance range: [{np.min(successful_distances):.3f}, {np.max(successful_distances):.3f}]")
    if failed_distances:
        print(f"Failed - mean distance: {np.mean(failed_distances):.3f} ± {np.std(failed_distances):.3f}")
        print(f"Failed - distance range: [{np.min(failed_distances):.3f}, {np.max(failed_distances):.3f}]")
    
    # Angular analysis
    successful_angles = [t['angle_deg'] for t in successful_targets]
    failed_angles = [t['angle_deg'] for t in failed_targets]
    
    print(f"\n📐 TARGET ANGLE ANALYSIS:")
    if successful_angles:
        print(f"Successful angles - mean: {np.mean(successful_angles):.1f}° ± {np.std(successful_angles):.1f}°")
        print(f"Successful angles - range: [{np.min(successful_angles):.1f}°, {np.max(successful_angles):.1f}°]")
    
    # Create visualizations
    create_target_distribution_plots(all_targets_array, successful_targets_array, failed_targets_array, 
                                   successful_distances, failed_distances,
                                   successful_angles, failed_angles)
    
    # Spatial bias analysis
    analyze_spatial_bias(successful_targets, failed_targets)
    
    return episode_results, {
        'all_targets': all_targets,
        'successful_targets': successful_targets,
        'failed_targets': failed_targets,
        'success_rate': success_rate
    }

def create_target_distribution_plots(all_targets, successful_targets, failed_targets,
                                   successful_distances, failed_distances,
                                   successful_angles, failed_angles):
    """Create comprehensive visualization plots."""
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Spatial distribution scatter plot
    ax1 = plt.subplot(2, 4, 1)
    plt.scatter(all_targets[:, 0], all_targets[:, 1], c='lightgray', alpha=0.6, s=30, label='All targets')
    if len(successful_targets) > 0:
        plt.scatter(successful_targets[:, 0], successful_targets[:, 1], c='green', alpha=0.8, s=50, label='Successful')
    if len(failed_targets) > 0:
        plt.scatter(failed_targets[:, 0], failed_targets[:, 1], c='red', alpha=0.6, s=30, label='Failed')
    plt.xlabel('Target X position')
    plt.ylabel('Target Y position')
    plt.title('Target Distribution: Success vs Failure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2. Success rate heatmap (2D histogram)
    ax2 = plt.subplot(2, 4, 2)
    if len(all_targets) > 0:
        # Create grid for heatmap
        x_edges = np.linspace(all_targets[:, 0].min() - 0.05, all_targets[:, 0].max() + 0.05, 10)
        y_edges = np.linspace(all_targets[:, 1].min() - 0.05, all_targets[:, 1].max() + 0.05, 10)
        
        # Count successes and totals in each bin
        total_hist, _, _ = np.histogram2d(all_targets[:, 0], all_targets[:, 1], bins=[x_edges, y_edges])
        if len(successful_targets) > 0:
            success_hist, _, _ = np.histogram2d(successful_targets[:, 0], successful_targets[:, 1], bins=[x_edges, y_edges])
        else:
            success_hist = np.zeros_like(total_hist)
        
        # Calculate success rate per bin (avoid division by zero)
        success_rate_grid = np.divide(success_hist, total_hist, out=np.zeros_like(success_hist), where=total_hist!=0)
        
        im = plt.imshow(success_rate_grid.T, origin='lower', cmap='RdYlGn', vmin=0, vmax=1,
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        plt.colorbar(im, label='Success Rate')
        plt.xlabel('Target X position')
        plt.ylabel('Target Y position')
        plt.title('Success Rate Heatmap')
    
    # 3. Distance distribution
    ax3 = plt.subplot(2, 4, 3)
    if successful_distances:
        plt.hist(successful_distances, bins=15, alpha=0.7, color='green', label='Successful', density=True)
    if failed_distances:
        plt.hist(failed_distances, bins=15, alpha=0.7, color='red', label='Failed', density=True)
    plt.xlabel('Target Distance from Origin')
    plt.ylabel('Density')
    plt.title('Target Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Angular distribution
    ax4 = plt.subplot(2, 4, 4)
    if successful_angles:
        plt.hist(successful_angles, bins=20, alpha=0.7, color='green', label='Successful', density=True)
    if failed_angles:
        plt.hist(failed_angles, bins=20, alpha=0.7, color='red', label='Failed', density=True)
    plt.xlabel('Target Angle (degrees)')
    plt.ylabel('Density')
    plt.title('Target Angle Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Polar plot of success by angle
    ax5 = plt.subplot(2, 4, 5, projection='polar')
    if successful_angles:
        successful_angles_rad = np.radians(successful_angles)
        ax5.scatter(successful_angles_rad, [1]*len(successful_angles_rad), c='green', alpha=0.7, s=30, label='Successful')
    if failed_angles:
        failed_angles_rad = np.radians(failed_angles)
        ax5.scatter(failed_angles_rad, [0.5]*len(failed_angles_rad), c='red', alpha=0.5, s=20, label='Failed')
    ax5.set_ylim(0, 1.2)
    ax5.set_title('Angular Success Pattern')
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 6. Success rate by distance bins
    ax6 = plt.subplot(2, 4, 6)
    all_distances = successful_distances + failed_distances  # Remove the list comprehension
    if all_distances:
        distance_bins = np.linspace(min(all_distances), max(all_distances), 8)
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        success_rates_by_distance = []
        
        for i in range(len(distance_bins)-1):
            in_bin_successful = sum(1 for d in successful_distances if distance_bins[i] <= d < distance_bins[i+1])
            in_bin_total = sum(1 for d in all_distances if distance_bins[i] <= d < distance_bins[i+1])
            rate = in_bin_successful / in_bin_total if in_bin_total > 0 else 0
            success_rates_by_distance.append(rate)
        
        plt.bar(bin_centers, success_rates_by_distance, width=(distance_bins[1]-distance_bins[0])*0.8, 
                color='skyblue', alpha=0.7, edgecolor='navy')
        plt.xlabel('Target Distance from Origin')
        plt.ylabel('Success Rate')
        plt.title('Success Rate by Target Distance')
        plt.grid(True, alpha=0.3)
    
    # 7. Success rate by angle bins
    ax7 = plt.subplot(2, 4, 7)
    all_angles_list = successful_angles + failed_angles
    if all_angles_list:
        angle_bins = np.linspace(-180, 180, 9)
        bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        success_rates_by_angle = []
        
        for i in range(len(angle_bins)-1):
            in_bin_successful = sum(1 for a in successful_angles if angle_bins[i] <= a < angle_bins[i+1])
            in_bin_total = sum(1 for a in all_angles_list if angle_bins[i] <= a < angle_bins[i+1])
            rate = in_bin_successful / in_bin_total if in_bin_total > 0 else 0
            success_rates_by_angle.append(rate)
        
        plt.bar(bin_centers, success_rates_by_angle, width=(angle_bins[1]-angle_bins[0])*0.8,
                color='lightcoral', alpha=0.7, edgecolor='darkred')
        plt.xlabel('Target Angle (degrees)')
        plt.ylabel('Success Rate')
        plt.title('Success Rate by Target Angle')
        plt.grid(True, alpha=0.3)
    
    # 8. Workspace coverage
    ax8 = plt.subplot(2, 4, 8)
    if len(all_targets) > 0:
        # Draw workspace boundary (approximate reachable area)
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Approximate arm reach (adjust based on your arm model)
        max_reach = 0.9  # Maximum reach
        min_reach = 0.1  # Minimum reach
        
        plt.fill_between(max_reach * np.cos(theta), max_reach * np.sin(theta), 
                        min_reach * np.cos(theta), min_reach * np.sin(theta),
                        alpha=0.1, color='gray', label='Approximate workspace')
        
        # Plot targets
        plt.scatter(all_targets[:, 0], all_targets[:, 1], c='lightgray', alpha=0.4, s=20, label='All targets')
        if len(successful_targets) > 0:
            plt.scatter(successful_targets[:, 0], successful_targets[:, 1], c='green', alpha=0.8, s=40, label='Successful')
        if len(failed_targets) > 0:
            plt.scatter(failed_targets[:, 0], failed_targets[:, 1], c='red', alpha=0.6, s=20, label='Failed')
        
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Workspace Coverage')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/target_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plots saved as: models/target_distribution_analysis.png")

def analyze_spatial_bias(successful_targets, failed_targets):
    """Analyze spatial biases in success patterns."""
    
    print(f"\n🧭 SPATIAL BIAS ANALYSIS:")
    print("-" * 30)
    
    if not successful_targets:
        print("No successful targets to analyze!")
        return
    
    # Quadrant analysis
    quadrants = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}  # Success counts
    quadrant_totals = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}  # Total attempts
    
    def get_quadrant(x, y):
        if x >= 0 and y >= 0: return 'Q1'
        elif x < 0 and y >= 0: return 'Q2'
        elif x < 0 and y < 0: return 'Q3'
        else: return 'Q4'
    
    # Count successes by quadrant
    for target in successful_targets:
        quad = get_quadrant(target['x'], target['y'])
        quadrants[quad] += 1
    
    # Count totals by quadrant
    all_targets = successful_targets + failed_targets
    for target in all_targets:
        quad = get_quadrant(target['x'], target['y'])
        quadrant_totals[quad] += 1
    
    print("Quadrant success rates:")
    for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
        rate = quadrants[quad] / quadrant_totals[quad] if quadrant_totals[quad] > 0 else 0
        print(f"  {quad} (x{'≥' if quad in ['Q1','Q4'] else '<'}0, y{'≥' if quad in ['Q1','Q2'] else '<'}0): "
              f"{quadrants[quad]}/{quadrant_totals[quad]} ({rate:.1%})")
    
    # Distance preference analysis
    successful_dists = [t['distance'] for t in successful_targets]
    failed_dists = [t['distance'] for t in failed_targets]
    
    if successful_dists and failed_dists:
        print(f"\nDistance preference:")
        print(f"  Successful mean distance: {np.mean(successful_dists):.3f}")
        print(f"  Failed mean distance: {np.mean(failed_dists):.3f}")
        print(f"  Difference: {np.mean(successful_dists) - np.mean(failed_dists):.3f}")
        
        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(successful_dists, failed_dists)
        print(f"  T-test p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Find best/worst performing regions
    if len(successful_targets) >= 3:
        successful_positions = np.array([[t['x'], t['y']] for t in successful_targets])
        centroid = np.mean(successful_positions, axis=0)
        print(f"\nSuccess centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
        
        # Find clusters (simple approach)
        distances_from_centroid = [np.linalg.norm(np.array([t['x'], t['y']]) - centroid) for t in successful_targets]
        tight_cluster_radius = np.percentile(distances_from_centroid, 50)  # 50% of successes within this radius
        print(f"50% of successes within {tight_cluster_radius:.3f} units of centroid")

if __name__ == "__main__":
    # Analyze target distribution
    results, analysis_data = analyze_target_distribution(
        num_test_episodes=200,  # Test many episodes
        model_path="models/arm_final2",
        vecnorm_path="models/vec_normalize2.pkl",
        success_threshold=0.06,
        episode_duration_limit=50
    )
    
    if analysis_data:
        print(f"\n🎉 Analysis complete!")
        print(f"   Success rate: {analysis_data['success_rate']:.1%}")
        print(f"   Check models/target_distribution_analysis.png for visualizations")
    else:
        print(f"\n❌ Analysis failed - no successful episodes found")