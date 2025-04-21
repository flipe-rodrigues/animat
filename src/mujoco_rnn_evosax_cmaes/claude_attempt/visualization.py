"""
Visualization utilities for the arm reaching task.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from matplotlib.animation import FuncAnimation
import mujoco
from mujoco import viewer

def plot_training_history(history_path: str, output_dir: str = "plots"):
    """
    Plot training metrics from a saved history file.
    
    Args:
        history_path: Path to the training history pickle file
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Extract metrics
    generations = [entry['generation'] for entry in history]
    mean_fitness = [entry['mean_fitness'] for entry in history]
    best_fitness = [entry['best_fitness'] for entry in history]
    best_overall = [entry['best_overall'] for entry in history]
    
    # Plot fitness over generations
    plt.figure(figsize=(12, 6))
    plt.plot(generations, mean_fitness, label='Mean Fitness', color='blue', alpha=0.7)
    plt.plot(generations, best_fitness, label='Best Fitness', color='green')
    plt.plot(generations, best_overall, label='Best Overall', color='red', linestyle='--')
    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Negative Distance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'fitness_history.png'))
    plt.close()
    
    # Calculate and plot improvement rate
    if len(generations) > 1:
        # Calculate delta between consecutive best_overall values
        improvements = np.diff(best_overall)
        
        plt.figure(figsize=(12, 6))
        plt.bar(generations[1:], improvements, color='purple', alpha=0.7)
        plt.title('Improvement Rate over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Improvement')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'improvement_rate.png'))
        plt.close()
    
    print(f"Training plots saved to {output_dir}")

def visualize_trajectory(positions: List[np.ndarray], target_pos: np.ndarray, output_path: str):
    """
    Visualize the trajectory of the end effector.
    
    Args:
        positions: List of end effector positions over time
        target_pos: Target position
        output_path: Path to save the visualization
    """
    # Convert to numpy array
    positions = np.array(positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # Plot start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
    
    # Plot target
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='purple', s=100, label='Target')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End Effector Trajectory')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    print(f"Trajectory visualization saved to {output_path}")

def create_trajectory_animation(positions: List[np.ndarray], target_pos: np.ndarray, output_path: str):
    """
    Create an animation of the end effector trajectory.
    
    Args:
        positions: List of end effector positions over time
        target_pos: Target position
        output_path: Path to save the animation
    """
    # Convert to numpy array
    positions = np.array(positions)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set limits
    x_min, x_max = np.min(positions[:, 0]) - 0.1, np.max(positions[:, 0]) + 0.1
    y_min, y_max = np.min(positions[:, 1]) - 0.1, np.max(positions[:, 1]) + 0.1
    z_min, z_max = np.min(positions[:, 2]) - 0.1, np.max(positions[:, 2]) + 0.1
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # Plot target
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='purple', s=100, label='Target')
    
    # Initialize line and point
    line, = ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
    point, = ax.plot([], [], [], 'ro', markersize=8, label='End Effector')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End Effector Trajectory Animation')
    
    # Add legend
    ax.legend()
    
    # Animation update function
    def update(frame):
        # Update line data (show trajectory up to current frame)
        line.set_data(positions[:frame, 0], positions[:frame, 1])
        line.set_3d_properties(positions[:frame, 2])
        
        # Update point position
        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        point.set_3d_properties([positions[frame, 2]])
        
        return line, point
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(positions), interval=50, blit=True)
    
    # Save animation
    anim.save(output_path, writer='pillow', fps=20)
    plt.close()
    
    print(f"Trajectory animation saved to {output_path}")

def plot_muscle_activations(activations: List[np.ndarray], output_path: str):
    """
    Plot muscle activations over time.
    
    Args:
        activations: List of muscle activation vectors over time
        output_path: Path to save the plot
    """
    # Convert to numpy array
    activations = np.array(activations)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot each muscle activation
    muscle_names = ['Shoulder Flexor', 'Shoulder Extensor', 'Elbow Flexor', 'Elbow Extensor']
    colors = ['red', 'blue', 'green', 'purple']
    
    for i in range(4):
        plt.plot(activations[:, i], label=muscle_names[i], color=colors[i])
    
    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Activation')
    plt.title('Muscle Activations Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    print(f"Muscle activation plot saved to {output_path}")

def visualize_reaching_performance(distances_list: List[np.ndarray], labels: List[str], output_path: str):
    """
    Visualize reaching performance across multiple runs or models.
    
    Args:
        distances_list: List of distance arrays from different runs
        labels: Labels for each run
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    for i, distances in enumerate(distances_list):
        plt.plot(distances, label=labels[i])
    
    plt.xlabel('Time Step')
    plt.ylabel('Distance to Target')
    plt.title('Distance to Target Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    print(f"Performance comparison saved to {output_path}")

def record_mujoco_video(model_path: str, data_path: str, output_path: str, duration: float = 10.0):
    """
    Record a video of the MuJoCo simulation.
    
    Args:
        model_path: Path to the MuJoCo model file
        data_path: Path to the MuJoCo data file
        output_path: Path to save the video
        duration: Duration of the video in seconds
    """
    # Load model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Load saved data if provided
    if data_path:
        with open(data_path, 'rb') as f:
            saved_data = pickle.load(f)
            # Apply saved data to the MuJoCo data object
            # This depends on how the data was saved
    
    # Initialize viewer
    with viewer.launch_passive(model, data) as v:
        # Record video
        v.cam.lookat = [0, 0, 0]  # Set camera look-at point
        v.cam.distance = 1.5      # Set camera distance
        v.cam.elevation = -20     # Set camera elevation
        v.cam.azimuth = 90        # Set camera azimuth
        
        # Start recording
        v.record_start(output_path)
        
        # Simulate for the specified duration
        frames = int(duration / model.opt.timestep)
        for i in range(frames):
            mujoco.mj_step(model, data)
            v.sync()
        
        # Stop recording
        v.record_stop()
    
    print(f"Video saved to {output_path}")
