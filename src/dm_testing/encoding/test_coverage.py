# %%
import numpy as np
import pickle

def estimate_target_positions_from_encoded_obs(observations):
    """
    Estimate target positions from grid-encoded observations.
    
    Observation structure (38D total):
    - [0:12]  : muscle sensors (normalized)
    - [12:28] : target XY encoded as 4×4=16 Gaussians  
    - [28]    : target Z (direct, should be ~0)
    - [29:38] : other dimensions
    """
    
    # Extract the encoded target part
    encoded_xy = observations[:, 12:28]  # 16 Gaussians for XY
    target_z = observations[:, 28]       # Direct Z coordinate
    
    # Recreate the grid centers (same as in GridEncoder)
    grid_size = 4
    min_val = [-0.65, -0.90] 
    max_val = [0.90, 0.35]
    
    x_centers = np.linspace(min_val[0], max_val[0], grid_size)
    y_centers = np.linspace(min_val[1], max_val[1], grid_size)
    
    # Create meshgrid and flatten to get all 16 (x,y) center positions
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_centers = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1)  # [16, 2]
    
    # Estimate target positions using weighted average of activated Gaussians
    estimated_targets = []
    
    for i, activations in enumerate(encoded_xy):
        # Weighted average of grid centers based on Gaussian activations
        weights = activations / (activations.sum() + 1e-8)  # Normalize weights
        estimated_xy = np.sum(grid_centers * weights[:, np.newaxis], axis=0)
        
        # Combine with Z coordinate
        estimated_target = np.array([estimated_xy[0], estimated_xy[1], target_z[i]])
        estimated_targets.append(estimated_target)
    
    return np.array(estimated_targets)


def analyze_target_coverage(demo_path="sac_demonstrations_50steps_successful_selective5.pkl"):
    """Analyze target coverage from demonstrations."""
    
    print(f"Analyzing target coverage from {demo_path}...")
    
    with open(demo_path, 'rb') as f:
        data = pickle.load(f)
    
    observations = data['observations']
    episode_starts = data['episode_starts']
    
    print(f"Total observations: {len(observations)}")
    print(f"Observation shape: {observations.shape}")
    print(f"Number of episodes: {len(episode_starts)}")
    
    # Estimate target positions
    estimated_targets = estimate_target_positions_from_encoded_obs(observations)
    
    # Get unique targets per episode (first observation of each episode)
    episode_targets = []
    for i, start_idx in enumerate(episode_starts):
        episode_targets.append(estimated_targets[start_idx])
    
    episode_targets = np.array(episode_targets)
    
    # Analyze coverage
    print(f"\n📊 TARGET COVERAGE ANALYSIS:")
    print("=" * 40)
    print(f"Unique episode targets: {len(episode_targets)}")
    print(f"X range: [{episode_targets[:, 0].min():.3f}, {episode_targets[:, 0].max():.3f}]")
    print(f"Y range: [{episode_targets[:, 1].min():.3f}, {episode_targets[:, 1].max():.3f}]")
    print(f"Z range: [{episode_targets[:, 2].min():.3f}, {episode_targets[:, 2].max():.3f}]")
    
    # Find unique targets (round to avoid floating point issues)
    unique_targets = np.unique(np.round(episode_targets, decimals=3), axis=0)
    print(f"Unique target positions: {len(unique_targets)}")
    
    # Visualize target coverage
    visualize_target_coverage(episode_targets, unique_targets)
    
    return episode_targets, unique_targets


def visualize_target_coverage(all_targets, unique_targets):
    """Visualize target position coverage."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(unique_targets[:, 0], unique_targets[:, 1], unique_targets[:, 2], 
               alpha=0.7, s=50, c='red', label='Unique targets')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.set_title(f'Target Positions 3D\n({len(unique_targets)} unique)')
    ax1.legend()
    
    # 2D projection XY
    ax2 = fig.add_subplot(132)
    ax2.scatter(unique_targets[:, 0], unique_targets[:, 1], alpha=0.7, s=50, c='red')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Target Coverage (X-Y)\nRange: X[{unique_targets[:, 0].min():.2f}, {unique_targets[:, 0].max():.2f}] Y[{unique_targets[:, 1].min():.2f}, {unique_targets[:, 1].max():.2f}]')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Show workspace boundaries
    ax2.axvline(-0.65, color='gray', linestyle='--', alpha=0.5, label='Workspace bounds')
    ax2.axvline(0.90, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(-0.90, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(0.35, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Coverage density heatmap
    ax3 = fig.add_subplot(133)
    hist, xedges, yedges = np.histogram2d(unique_targets[:, 0], unique_targets[:, 1], bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(hist.T, extent=extent, origin='lower', cmap='viridis', alpha=0.8)
    ax3.scatter(unique_targets[:, 0], unique_targets[:, 1], alpha=0.5, s=20, c='red')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Target Density Heatmap')
    plt.colorbar(im, ax=ax3, label='Target count')
    
    plt.tight_layout()
    plt.show()
    
    # Print coverage statistics
    print(f"\n📈 COVERAGE STATISTICS:")
    print(f"Workspace bounds: X[-0.65, 0.90], Y[-0.90, 0.35]")
    print(f"Actual coverage:  X[{unique_targets[:, 0].min():.3f}, {unique_targets[:, 0].max():.3f}], Y[{unique_targets[:, 1].min():.3f}, {unique_targets[:, 1].max():.3f}]")
    
    # Calculate coverage percentages
    x_range_workspace = 0.90 - (-0.65)  # 1.55
    y_range_workspace = 0.35 - (-0.90)  # 1.25
    x_range_covered = unique_targets[:, 0].max() - unique_targets[:, 0].min()
    y_range_covered = unique_targets[:, 1].max() - unique_targets[:, 1].min()
    
    x_coverage = (x_range_covered / x_range_workspace) * 100
    y_coverage = (y_range_covered / y_range_workspace) * 100
    
    print(f"Coverage: X={x_coverage:.1f}%, Y={y_coverage:.1f}%")
    
    # Check for gaps in coverage
    x_spacing = np.diff(np.sort(unique_targets[:, 0]))
    y_spacing = np.diff(np.sort(unique_targets[:, 1]))
    max_x_gap = np.max(x_spacing) if len(x_spacing) > 0 else 0
    max_y_gap = np.max(y_spacing) if len(y_spacing) > 0 else 0
    
    print(f"Largest gaps: X={max_x_gap:.3f}, Y={max_y_gap:.3f}")
    
    if max_x_gap > 0.1 or max_y_gap > 0.1:
        print("⚠️  Large gaps detected in target coverage!")
    else:
        print("✅ Target coverage appears fairly uniform")


# Add this to analyze your current demonstrations
if __name__ == "__main__":
    # Analyze target coverage
    targets, unique_targets = analyze_target_coverage("sac_demonstrations_50steps_successful_selective5.pkl")
# %%
