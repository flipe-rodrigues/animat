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

# Zero out encoded target influence for stimulation
print("Zeroing out encoded target influence for stimulation...")
best_rnn.policy.W_in.data[:, 12:] = 0  # Zero 81D encoded target

print("="*80)
print("COMPREHENSIVE BIZZI-STYLE STIMULATION ANALYSIS")
print("="*80)

# %%
"""
..######...#######..##.....##.########.....###....########..########
.##....##.##.....##.###...###.##.....##...##.##...##.....##.##......
.##.......##.....##.####.####.##.....##..##...##..##.....##.##......
.##.......##.....##.##.###.##.########..##.....##.########..######..
.##.......##.....##.##.....##.##........#########.##...##...##......
.##....##.##.....##.##.....##.##........##.....##.##....##..##......
..######...#######..##.....##.##........##.....##.##.....##.########
"""

def analyze_force_differences(mechanical_forces, neural_forces, stimulated_forces, positions):
    """Compare mechanical vs neural vs stimulated force fields"""
    
    # Calculate differences
    neural_vs_mechanical = neural_forces - mechanical_forces
    stimulated_vs_mechanical = stimulated_forces - mechanical_forces
    stimulated_vs_neural = stimulated_forces - neural_forces
    
    # Extract components
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    return {
        'positions': positions,
        'x_pos': x_pos,
        'y_pos': y_pos,
        'mechanical': mechanical_forces,
        'neural': neural_forces,
        'stimulated': stimulated_forces,
        'neural_vs_mechanical': neural_vs_mechanical,
        'stimulated_vs_mechanical': stimulated_vs_mechanical,
        'stimulated_vs_neural': stimulated_vs_neural
    }

def plot_comprehensive_analysis(analysis_data, unit_idx):
    """Create comprehensive plots with proper scaling"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Calculate global scale for consistent visualization
    all_forces = np.concatenate([
        analysis_data['mechanical'].flatten(),
        analysis_data['neural'].flatten(), 
        analysis_data['stimulated'].flatten()
    ])
    global_scale = np.percentile(np.abs(all_forces[~np.isnan(all_forces)]), 95)
    if global_scale == 0:
        global_scale = 1
    
    # Raw force fields
    for i, (name, forces) in enumerate([
        ('Mechanical (Rest)', analysis_data['mechanical']),
        ('Neural (Baseline)', analysis_data['neural']),
        ('Stimulated', analysis_data['stimulated'])
    ]):
        ax = axes[0, i]
        
        # Use consistent scaling
        q = ax.quiver(analysis_data['x_pos'], analysis_data['y_pos'], 
                     forces[:, 0], forces[:, 1], 
                     scale=global_scale*20, alpha=0.8, 
                     color=['blue', 'green', 'red'][i])
        
        ax.set_title(f'{name} Force Field\nMax: {np.max(np.linalg.norm(forces, axis=1)):.4f}', 
                    fontsize=12, weight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Difference fields with separate scaling
    diff_data = [
        ('Neural - Mechanical', analysis_data['neural_vs_mechanical']),
        ('Stimulated - Mechanical', analysis_data['stimulated_vs_mechanical']),
        ('Stimulated - Neural', analysis_data['stimulated_vs_neural'])
    ]
    
    for i, (name, diff_forces) in enumerate(diff_data):
        ax = axes[1, i]
        
        # Individual scaling for differences
        diff_scale = np.percentile(np.abs(diff_forces.flatten()), 95)
        if diff_scale == 0:
            diff_scale = 1
            
        ax.quiver(analysis_data['x_pos'], analysis_data['y_pos'], 
                 diff_forces[:, 0], diff_forces[:, 1], 
                 scale=diff_scale*20, alpha=0.8, 
                 color=['purple', 'orange', 'darkred'][i])
        
        ax.set_title(f'{name}\nMax: {np.max(np.linalg.norm(diff_forces, axis=1)):.4f}', 
                    fontsize=12, weight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle(f'Unit {unit_idx}: Comprehensive Force Field Analysis', fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save plot
    config.save_plot(f"comprehensive_unit_{unit_idx}_analysis.png")
    plt.show()

def calculate_convergence_metrics(analysis_data):
    """Calculate convergence metrics for the stimulated vs neural difference"""
    
    diff_forces = analysis_data['stimulated_vs_neural']
    positions = analysis_data['positions']
    
    # Force magnitudes
    force_magnitudes = np.linalg.norm(diff_forces, axis=1)
    
    # Skip if no significant forces
    if np.max(force_magnitudes) < 1e-6:
        return 0.0
    
    # Center of workspace
    center = np.mean(positions, axis=0)
    
    # Vectors from each position to center
    to_center_vecs = center - positions
    to_center_norms = np.linalg.norm(to_center_vecs, axis=1)
    
    # Avoid division by zero
    valid_indices = to_center_norms > 1e-6
    if np.sum(valid_indices) < 3:
        return 0.0
    
    # Normalized vectors toward center
    to_center_unit = to_center_vecs[valid_indices] / to_center_norms[valid_indices, np.newaxis]
    
    # Normalized force vectors
    force_norms = force_magnitudes[valid_indices]
    force_unit = diff_forces[valid_indices] / np.maximum(force_norms[:, np.newaxis], 1e-10)
    
    # Alignment with center-pointing vectors
    alignments = np.sum(force_unit * to_center_unit, axis=1)
    
    # Weighted convergence score
    weights = force_norms / np.sum(force_norms)
    convergence_score = np.sum(weights * alignments)
    
    return convergence_score

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

print("\nProcessing selected units with comprehensive analysis...")

for unit_idx in config.STIMULATION_UNITS:
    print(f"\n{'='*60}")
    print(f"ANALYZING UNIT {unit_idx}")
    print(f"{'='*60}")
    
    # 1. Mechanical forces (true resting state)
    print("1. Measuring mechanical forces (no neural control)...")
    mechanical_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
        rest_mode='mechanical'
    )
    
    # 2. Neural baseline forces (normal RNN control, NO stimulation)
    print("2. Measuring neural baseline forces...")
    neural_data = env.stimulate(
        best_rnn,
        units=np.array([]),  # ← NO UNITS STIMULATED!
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
        rest_mode='neural'
    )
    
    # 3. Stimulated forces (RNN control WITH unit stimulation)
    print("3. Measuring stimulated forces...")
    stimulated_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),  # ← STIMULATE THE UNIT!
        action_modifier=1,
        delay=1,
        seed=0,
        render=False,
        rest_mode='neural'
    )
    
    # Process the three DIFFERENT datasets
    datasets = {
        'mechanical': mechanical_data,
        'neural': neural_data,
        'stimulated': stimulated_data  # ← Now actually different!
    }
    
    processed_data = {}
    
    for name, data in datasets.items():
        # Extract arrays
        position_vecs = np.nan_to_num(np.array(data["position"]))
        force_vecs = np.nan_to_num(np.array(data["force"]))
        time_array = np.array(data["time"])
        stimulated_array = np.array(data["stimulated"])
        
        positions = []
        forces = []
        
        # For mechanical: use rest periods (should all be rest anyway)
        # For neural: use rest periods (no stimulation happening)  
        # For stimulated: use stimulation periods
        for t in range(1, int(reacher.data.time) + 1):
            if name == 'stimulated':
                # Use stimulation periods for the stimulated condition
                start_time = t - 0.1
                stop_time = t
                indices = (time_array > start_time) & (time_array <= stop_time) & stimulated_array
            else:
                # Use rest periods for mechanical and neural baseline
                start_time = t - 0.5 - 0.1
                stop_time = t - 0.5
                indices = (time_array > start_time) & (time_array <= stop_time) & ~stimulated_array
            
            if np.any(indices):
                avg_position = np.mean(position_vecs[indices], axis=0)
                avg_force = np.mean(force_vecs[indices], axis=0)
                positions.append(avg_position)
                forces.append(avg_force)
        
        processed_data[name] = {
            'positions': np.array(positions),
            'forces': np.array(forces)
        }
    
    # Ensure all datasets have same length (take minimum)
    min_length = min(len(processed_data[name]['positions']) for name in processed_data)
    for name in processed_data:
        processed_data[name]['positions'] = processed_data[name]['positions'][:min_length]
        processed_data[name]['forces'] = processed_data[name]['forces'][:min_length]
    
    # Create analysis data structure
    analysis_data = analyze_force_differences(
        processed_data['mechanical']['forces'],
        processed_data['neural']['forces'], 
        processed_data['stimulated']['forces'],
        processed_data['stimulated']['positions']  # Use stimulated positions as reference
    )
    
    # Calculate convergence
    convergence_score = calculate_convergence_metrics(analysis_data)
    
    print(f"3. Analysis results:")
    print(f"   Convergence score: {convergence_score:.4f}")
    print(f"   Mechanical force magnitude (mean): {np.mean(np.linalg.norm(analysis_data['mechanical'], axis=1)):.4f}")
    print(f"   Neural force magnitude (mean): {np.mean(np.linalg.norm(analysis_data['neural'], axis=1)):.4f}")
    print(f"   Stimulated force magnitude (mean): {np.mean(np.linalg.norm(analysis_data['stimulated'], axis=1)):.4f}")
    
    # Debug control signals - fix the variable scope issue
    print(f"4. Control signal analysis:")
    ctrl_data = np.array(neural_data["ctrl"])
    stimulated_mask = np.array(neural_data["stimulated"])
    
    rest_ctrl = ctrl_data[~stimulated_mask]
    stim_ctrl = ctrl_data[stimulated_mask]
    
    # Also analyze mechanical data
    mech_ctrl_data = np.array(mechanical_data["ctrl"])
    mech_stimulated_mask = np.array(mechanical_data["stimulated"])
    mech_rest_ctrl = mech_ctrl_data[~mech_stimulated_mask]
    
    print(f"   Mechanical rest control (mean): {np.mean(mech_rest_ctrl, axis=0)}")
    print(f"   Mechanical rest control (std): {np.std(mech_rest_ctrl, axis=0)}")
    print(f"   Neural rest control (mean): {np.mean(rest_ctrl, axis=0)}")
    print(f"   Neural rest control (std): {np.std(rest_ctrl, axis=0)}")
    print(f"   Stimulated control (mean): {np.mean(stim_ctrl, axis=0)}")
    print(f"   Stimulated control (std): {np.std(stim_ctrl, axis=0)}")
    
    # Generate comprehensive plots
    print("5. Generating comprehensive plots...")
    plot_comprehensive_analysis(analysis_data, unit_idx)

print(f"\n✅ All plots saved to: {config.OUTPUT_DIR}")
print("Comprehensive analysis complete!")


# %%
