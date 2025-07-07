import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from shimmy_wrapper import GridEncoder
from matplotlib.colors import LinearSegmentedColormap

# Set up the encoder
grid_size = 4
encoder = GridEncoder(
    grid_size=grid_size,
    min_val=[-0.65, -0.90],
    max_val=[0.90, 0.35],
    sigma_scale=0.8
)

# Define one target position to test
target_x, target_y = 0.2, -0.3  # You can change this to any position

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Get grid centers
centers = encoder.grid_centers.numpy()

# Calculate activation for this specific target
target_tensor = torch.tensor([[target_x, target_y]], dtype=torch.float32)
with torch.no_grad():
    activations = encoder(target_tensor)[0]  # Get first (and only) sample

# Plot workspace
ax.set_xlim(-0.7, 1.0)
ax.set_ylim(-1.0, 0.4)

# Create a high-resolution blue-to-red colormap
colors = ['#0066CC', '#4488DD', '#88AAEE', '#CCCCFF', '#FFFFFF', 
          '#FFCCCC', '#EE8888', '#DD4444', '#CC0000']
n_bins = 256  # High resolution
cmap = LinearSegmentedColormap.from_list('blue_red', colors, N=n_bins)

# Draw grid cells as circles with activation-based colors and sizes
# FIXED: Use constant alpha instead of variable alpha
scatter = ax.scatter(centers[:, 0], centers[:, 1], 
                    s=[200 + 800 * act.item() for act in activations],
                    c=activations.numpy(), 
                    cmap=cmap,  # Use high-resolution colormap
                    alpha=0.9,  # Fixed alpha for all circles
                    edgecolors='black',
                    linewidth=1,
                    vmin=0, vmax=1)

# Add activation values near each cell
for i, (center, activation) in enumerate(zip(centers, activations)):
    ax.annotate(f'{activation.item():.3f}', 
               (center[0], center[1] - 0.08),  # Slightly below each circle
               ha='center', va='center', 
               fontsize=9, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Mark the target position
ax.scatter(target_x, target_y, 
          marker='*', s=400, c='yellow', 
          edgecolors='black', linewidth=2,
          label='Target')

# Add title and formatting
ax.set_title(f'Gaussian Target Encoding\nTarget: ({target_x:.1f}, {target_y:.1f}, 0.0)', fontsize=14)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add colorbar with higher resolution
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Activation Level', rotation=270, labelpad=20)
# Add more ticks for better resolution
cbar.set_ticks(np.linspace(0, 1, 11))  # 0.0, 0.1, 0.2, ..., 1.0

plt.tight_layout()
plt.savefig('grid_encoding_example.png', dpi=150, bbox_inches='tight')

print("✅ Grid encoding visualization generated!")
print(f"📁 File saved: grid_encoding_example.png")
print(f"🎯 Target position: ({target_x}, {target_y})")