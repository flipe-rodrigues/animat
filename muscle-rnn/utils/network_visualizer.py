"""
Real-Time Network Activity Visualizer

Renders a spatial depiction of the neural network with units as circles
whose colors reflect their activation levels. Can be composited with
MuJoCo simulation frames for synchronized playback.

Layout (left to right):
    [Proprioceptive]    [RNN Hidden]    [Output]
       Ia  II  Ib                      Alpha  γs  γd
       |   |   |                         |
        Muscles                       Muscles

    [Target Grid]
      (2D spatial)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import torch

from core.constants import (
    DEFAULT_NUM_MUSCLES,
    DEFAULT_RNN_HIDDEN_SIZE,
    DEFAULT_TARGET_GRID_SIZE,
    DEFAULT_SHOW_RNN_UNITS,
    DEFAULT_DPI,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_VIDEO_FPS,
)


@dataclass
class UnitPosition:
    """Position and metadata for a single unit."""
    x: float
    y: float
    radius: float
    label: str
    module: str
    index: int


@dataclass 
class ConnectionSpec:
    """Specification for a connection between units."""
    from_module: str
    from_idx: int
    to_module: str
    to_idx: int
    weight: float = 1.0


class NetworkActivityVisualizer:
    """
    Visualizes neural network activity as a spatial diagram.
    
    Usage:
        viz = NetworkActivityVisualizer(num_muscles=4, rnn_hidden_size=128)
        
        # During simulation loop:
        frame = viz.render_frame(activations_dict)
        
        # Or composite with MuJoCo:
        combined = viz.render_with_mujoco(activations_dict, mujoco_frame)
    """
    
    def __init__(
        self,
        num_muscles: int = DEFAULT_NUM_MUSCLES,
        rnn_hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE,
        target_grid_size: int = DEFAULT_TARGET_GRID_SIZE,
        figsize: Tuple[float, float] = (14, 8),
        dpi: int = DEFAULT_DPI,
        show_connections: bool = True,
        show_rnn_units: int = DEFAULT_SHOW_RNN_UNITS,
        colormap: str = 'RdBu_r',
        unit_radius: float = 0.15,
    ):
        self.num_muscles = num_muscles
        self.rnn_hidden_size = rnn_hidden_size
        self.target_grid_size = target_grid_size
        self.show_rnn_units = min(show_rnn_units, rnn_hidden_size)
        self.show_connections = show_connections
        self.unit_radius = unit_radius
        
        # Color mapping
        self.cmap = plt.get_cmap(colormap)
        self.norm = Normalize(vmin=-1, vmax=1)  # Normalized activations
        
        # Create figure
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        self.has_drawn_colorbar = False
        
        # Unit positions (computed once)
        self.units: Dict[str, List[UnitPosition]] = {}
        self.connections: List[ConnectionSpec] = []
        
        # Setup layout
        self._setup_layout()
        
    def _setup_layout(self):
        """Compute positions for all units."""
        
        # Layout parameters
        left_x = 1.5      # Proprioceptive modules
        center_x = 5.5    # RNN
        right_x = 9.5     # Output modules
        
        # Vertical layout: shifted down to make room for target grid above
        y_top = 5.5
        y_bottom = 0.5
        
        r = self.unit_radius
        
        # ===== PROPRIOCEPTIVE MODULE (left) =====
        # Arrange as 3 columns (Ia, II, Ib) x num_muscles rows
        proprio_cols = {'Ia': left_x - 0.8, 'II': left_x, 'Ib': left_x + 0.8}
        muscle_ys = np.linspace(y_top - 0.5, y_bottom + 0.5, self.num_muscles)
        
        for sensor_type, col_x in proprio_cols.items():
            self.units[sensor_type] = []
            for i, y in enumerate(muscle_ys):
                self.units[sensor_type].append(UnitPosition(
                    x=col_x, y=y, radius=r,
                    label=f'{sensor_type}_{i}',
                    module=sensor_type,
                    index=i
                ))
        
        # ===== TARGET GRID (above RNN) =====
        self.units['target'] = []
        grid_size = self.target_grid_size
        target_x_start = center_x - 0.8  # Center above RNN
        target_x_end = center_x + 0.8
        target_y_start = y_top + 1.5
        target_y_end = y_top + 2.7
        
        target_xs = np.linspace(target_x_start, target_x_end, grid_size)
        target_ys = np.linspace(target_y_start, target_y_end, grid_size)
        
        for j, ty in enumerate(target_ys):
            for i, tx in enumerate(target_xs):
                self.units['target'].append(UnitPosition(
                    x=tx, y=ty, radius=r * 0.6,
                    label=f'tgt_{i}_{j}',
                    module='target',
                    index=j * grid_size + i
                ))
        
        # ===== RNN HIDDEN (center) =====
        # Arrange in a grid pattern
        self.units['rnn'] = []
        n_show = self.show_rnn_units
        rnn_cols = int(np.ceil(np.sqrt(n_show)))
        rnn_rows = int(np.ceil(n_show / rnn_cols))
        
        rnn_xs = np.linspace(center_x - 1.2, center_x + 1.2, rnn_cols)
        rnn_ys = np.linspace(y_top - 0.5, y_bottom + 0.5, rnn_rows)
        
        idx = 0
        for row, y in enumerate(rnn_ys):
            for col, x in enumerate(rnn_xs):
                if idx >= n_show:
                    break
                self.units['rnn'].append(UnitPosition(
                    x=x, y=y, radius=r * 0.7,
                    label=f'h_{idx}',
                    module='rnn',
                    index=idx
                ))
                idx += 1
        
        # ===== OUTPUT MODULE (right) =====
        # Arrange as 3 columns (Alpha, γ_static, γ_dynamic) x num_muscles rows
        output_cols = {'alpha': right_x - 0.8, 'gamma_s': right_x, 'gamma_d': right_x + 0.8}
        
        for output_type, col_x in output_cols.items():
            self.units[output_type] = []
            for i, y in enumerate(muscle_ys):
                self.units[output_type].append(UnitPosition(
                    x=col_x, y=y, radius=r,
                    label=f'{output_type}_{i}',
                    module=output_type,
                    index=i
                ))
        
        # ===== CONNECTIONS =====
        if self.show_connections:
            self._setup_connections()
    
    def _setup_connections(self):
        """Define connections to draw between units."""
        self.connections = []
        
        # Proprioceptive -> RNN (all sensors feed into RNN)
        # Only show connections to first few RNN units to avoid clutter
        for sensor_type in ['Ia', 'II', 'Ib']:
            for i in range(self.num_muscles):
                # Connect each sensor to a subset of RNN units
                for rnn_idx in range(0, min(8, self.show_rnn_units), 2):
                    self.connections.append(ConnectionSpec(
                        sensor_type, i, 'rnn', rnn_idx, weight=0.5
                    ))
        
        # Target -> RNN (target info feeds into RNN)
        # Connect target grid corners to RNN
        target_corners = [0, self.target_grid_size - 1, 
                          self.target_grid_size * (self.target_grid_size - 1),
                          self.target_grid_size ** 2 - 1]
        for tgt_idx in target_corners:
            for rnn_idx in range(0, min(8, self.show_rnn_units), 2):
                self.connections.append(ConnectionSpec(
                    'target', tgt_idx, 'rnn', rnn_idx, weight=0.4
                ))
        
        # RNN -> Motor Outputs
        for output_type in ['alpha', 'gamma_s', 'gamma_d']:
            for out_idx in range(self.num_muscles):
                # Connect from subset of RNN units
                for rnn_idx in range(0, min(8, self.show_rnn_units), 2):
                    self.connections.append(ConnectionSpec(
                        'rnn', rnn_idx, output_type, out_idx, weight=0.5
                    ))
    
    def _init_figure(self):
        """Initialize matplotlib figure."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        self.ax.set_xlim(-0.5, 11.5)
        self.ax.set_ylim(-0.5, 9.5)  # Adjusted for target grid above RNN
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
    def _get_unit_pos(self, module: str, idx: int) -> Tuple[float, float]:
        """Get (x, y) position of a unit."""
        if module in self.units and idx < len(self.units[module]):
            u = self.units[module][idx]
            return u.x, u.y
        return 0, 0
    
    def render_frame(
        self,
        activations: Dict[str, np.ndarray],
        title: str = "",
        step: int = None,
        phase: str = None,
    ) -> np.ndarray:
        """
        Render a single frame of network activity.
        
        Args:
            activations: Dict with keys like 'Ia', 'II', 'Ib', 'rnn', 
                        'alpha', 'gamma_s', 'gamma_d', 'target'
                        Each value is a 1D array of activations.
            title: Optional title
            step: Optional step number
            phase: Optional trial phase
            
        Returns:
            RGB image as numpy array [H, W, 3]
        """
        if self.fig is None:
            self._init_figure()
        
        self.ax.clear()
        self.ax.set_xlim(-0.5, 11.5)
        self.ax.set_ylim(-0.5, 9.5)  # Adjusted for target grid above RNN
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Draw connections first (behind units)
        if self.show_connections:
            self._draw_connections(activations)
        
        # Draw units
        for module_name, units in self.units.items():
            acts = activations.get(module_name, np.zeros(len(units)))
            
            # Ensure correct length
            if len(acts) < len(units):
                acts = np.pad(acts, (0, len(units) - len(acts)))
            elif len(acts) > len(units):
                acts = acts[:len(units)]
            
            self._draw_units(units, acts, module_name)
        
        # Draw labels
        self._draw_labels()
        
        # Title and info
        title_parts = []
        if title:
            title_parts.append(title)
        if step is not None:
            title_parts.append(f"Step: {step}")
        if phase:
            title_parts.append(f"Phase: {phase}")
        
        if title_parts:
            self.ax.set_title(" | ".join(title_parts), fontsize=12, fontweight='bold')
        
        # Convert to image
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]

        return img
    
    def _draw_units(
        self, 
        units: List[UnitPosition], 
        activations: np.ndarray,
        module_name: str
    ):
        """Draw a group of units with activation colors."""
        circles = []
        colors = []
        
        for unit, act in zip(units, activations):
            circle = plt.Circle((unit.x, unit.y), unit.radius)
            circles.append(circle)
            colors.append(self.cmap(self.norm(act)))
        
        collection = PatchCollection(circles, facecolors=colors, edgecolors='black', linewidths=0.5)
        self.ax.add_collection(collection)
    
    def _draw_connections(self, activations: Dict[str, np.ndarray]):
        """Draw connections between units."""
        lines = []
        colors = []
        
        for conn in self.connections:
            x1, y1 = self._get_unit_pos(conn.from_module, conn.from_idx)
            x2, y2 = self._get_unit_pos(conn.to_module, conn.to_idx)
            
            # Get activation of source unit for color
            acts = activations.get(conn.from_module, np.zeros(self.num_muscles))
            if conn.from_idx < len(acts):
                act = acts[conn.from_idx]
            else:
                act = 0
            
            lines.append([(x1, y1), (x2, y2)])
            
            # Color based on activation and weight
            alpha = min(0.6, 0.1 + abs(act) * 0.2) * conn.weight
            if act > 0:
                colors.append((0.8, 0.2, 0.2, alpha))  # Red for positive
            else:
                colors.append((0.2, 0.2, 0.8, alpha))  # Blue for negative
        
        lc = LineCollection(lines, colors=colors, linewidths=0.5)
        self.ax.add_collection(lc)
    
    def _draw_labels(self):
        """Draw module labels."""
        label_style = dict(fontsize=10, fontweight='bold', ha='center')
        
        # Proprioceptive
        self.ax.text(1.5, 6.3, 'Proprioceptive', **label_style)
        self.ax.text(0.7, 5.8, 'Ia', fontsize=8, ha='center')
        self.ax.text(1.5, 5.8, 'II', fontsize=8, ha='center')
        self.ax.text(2.3, 5.8, 'Ib', fontsize=8, ha='center')
        
        # Target (above RNN)
        self.ax.text(5.5, 8.5, 'Target Grid', **label_style)
        
        # RNN
        self.ax.text(5.5, 6.3, 'RNN Hidden', **label_style)
        
        # Output
        self.ax.text(9.5, 6.3, 'Motor Output', **label_style)
        self.ax.text(8.7, 5.8, 'α', fontsize=9, ha='center')
        self.ax.text(9.5, 5.8, 'γs', fontsize=9, ha='center')
        self.ax.text(10.3, 5.8, 'γd', fontsize=9, ha='center')
        
        # Muscle labels on the sides
        for i in range(self.num_muscles):
            y = self.units['Ia'][i].y
            self.ax.text(-0.2, y, f'M{i+1}', fontsize=8, ha='right', va='center')
            self.ax.text(11.2, y, f'M{i+1}', fontsize=8, ha='left', va='center')
        
        # Add colorbar
        if not self.has_drawn_colorbar:
            self.has_drawn_colorbar = True
            sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=self.ax, orientation='vertical', 
                                    fraction=0.02, pad=0.02)
            cbar.set_label('Activation', fontsize=9)
    
    def render_with_mujoco(
        self,
        activations: Dict[str, np.ndarray],
        mujoco_frame: np.ndarray,
        layout: str = 'horizontal',  # 'horizontal', 'vertical', 'overlay'
        title: str = "",
        step: int = None,
        phase: str = None,
    ) -> np.ndarray:
        """
        Render network activity alongside MuJoCo simulation frame.
        
        Args:
            activations: Network activations dict
            mujoco_frame: RGB frame from MuJoCo renderer [H, W, 3]
            layout: How to combine frames
            title, step, phase: Display info
            
        Returns:
            Combined RGB image
        """
        # Render network
        net_frame = self.render_frame(activations, title, step, phase)
        
        # Resize MuJoCo frame to match height
        import cv2
        
        if layout == 'horizontal':
            # Stack horizontally
            target_h = net_frame.shape[0]
            scale = target_h / mujoco_frame.shape[0]
            new_w = int(mujoco_frame.shape[1] * scale)
            mj_resized = cv2.resize(mujoco_frame, (new_w, target_h))
            
            combined = np.concatenate([mj_resized, net_frame], axis=1)
            
        elif layout == 'vertical':
            # Stack vertically
            target_w = net_frame.shape[1]
            scale = target_w / mujoco_frame.shape[1]
            new_h = int(mujoco_frame.shape[0] * scale)
            mj_resized = cv2.resize(mujoco_frame, (target_w, new_h))
            
            combined = np.concatenate([mj_resized, net_frame], axis=0)
            
        elif layout == 'overlay':
            # Picture-in-picture (MuJoCo small in corner)
            pip_scale = 0.3
            pip_h = int(net_frame.shape[0] * pip_scale)
            pip_w = int(mujoco_frame.shape[1] * (pip_h / mujoco_frame.shape[0]))
            mj_small = cv2.resize(mujoco_frame, (pip_w, pip_h))
            
            combined = net_frame.copy()
            # Place in top-right corner
            combined[10:10+pip_h, -pip_w-10:-10] = mj_small
        
        else:
            combined = net_frame
        
        return combined
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def extract_activations_from_info(
    info: Dict[str, Any],
    obs: np.ndarray,
    num_muscles: int,
    target_grid_size: int = 4,
    target_encoder: Optional[Any] = None,
    env_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract activations dict from controller forward info and observation.
    
    Args:
        info: Dict returned by controller.forward()
        obs: Current observation array [proprio, target_xyz]
        num_muscles: Number of muscles
        target_grid_size: Size of target grid
        target_encoder: TargetEncoder instance for encoding XYZ to grid
        env_info: Environment info dict (contains 'target_visible')
        
    Returns:
        Dict suitable for NetworkActivityVisualizer.render_frame()
    """
    activations = {}
    
    # Sensory outputs
    if 'sensory_outputs' in info:
        sensory = info['sensory_outputs']
        if 'type_Ia' in sensory:
            activations['Ia'] = sensory['type_Ia'].cpu().numpy().flatten()
        if 'type_II' in sensory:
            activations['II'] = sensory['type_II'].cpu().numpy().flatten()
        if 'type_Ib' in sensory:
            activations['Ib'] = sensory['type_Ib'].cpu().numpy().flatten()
    
    # RNN hidden
    if 'rnn_hidden' in info:
        activations['rnn'] = info['rnn_hidden'].cpu().numpy().flatten()
    
    # Motor outputs
    if 'alpha' in info:
        activations['alpha'] = info['alpha'].cpu().numpy().flatten()
    if 'gamma_static' in info:
        activations['gamma_s'] = info['gamma_static'].cpu().numpy().flatten()
    if 'gamma_dynamic' in info:
        activations['gamma_d'] = info['gamma_dynamic'].cpu().numpy().flatten()
    
    # Target encoding: check if target is visible first
    target_visible = True
    if env_info is not None:
        target_visible = env_info.get('target_visible', True)
    
    if not target_visible:
        # Target not visible - show zeros
        activations['target'] = np.zeros(target_grid_size ** 2)
    elif target_encoder is not None:
        # Encode XYZ to Gaussian grid
        # Obs layout: [proprio (num_muscles*3), target_xyz (3)]
        proprio_dim = num_muscles * 3
        target_xyz = obs[proprio_dim:proprio_dim + 3]
        target_tensor = torch.tensor(target_xyz, dtype=torch.float32).unsqueeze(0)
        encoded = target_encoder.encode(target_tensor).squeeze(0).numpy()
        activations['target'] = encoded
    else:
        # Fallback: zeros if no encoder provided
        activations['target'] = np.zeros(target_grid_size ** 2)
    
    return activations


def record_episode_with_network(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    output_video: Optional[str] = None,
    fps: int = DEFAULT_VIDEO_FPS,
    layout: str = 'horizontal',
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Record an episode with synchronized network activity visualization.
    
    Args:
        controller: Trained controller
        xml_path: Path to MuJoCo XML
        sensor_stats: Sensor normalization stats
        max_steps: Maximum episode steps
        output_video: Path to save video (None = don't save)
        fps: Video framerate
        layout: 'horizontal', 'vertical', or 'overlay'
        seed: Random seed for reproducibility
        
    Returns:
        Dict with trajectory data and frames
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from envs.reaching import ReachingEnv
    from models.controllers import ModelConfig
    from models.modules.target import TargetEncoder
    
    # Get model config from controller
    config = controller.config
    
    # Create target encoder for visualization
    target_grid_size = int(np.sqrt(config.num_target_units))
    target_encoder = TargetEncoder(
        grid_size=target_grid_size,
        sigma=config.target_sigma,
        workspace_bounds=config.workspace_bounds,
    )
    
    # Create environment and visualizer
    env = ReachingEnv(xml_path, render_mode='rgb_array', sensor_stats=sensor_stats)
    viz = NetworkActivityVisualizer(
        num_muscles=config.num_muscles,
        rnn_hidden_size=config.rnn_hidden_size,
        target_grid_size=target_grid_size
    )
    
    device = next(controller.parameters()).device
    controller.eval()
    
    # Storage
    combined_frames = []
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'infos': [],
        'activations': []
    }
    
    obs, env_info = env.reset(seed=seed)
    controller.init_hidden(1, device)
    
    with torch.no_grad():
        for step in range(max_steps):
            trajectory['observations'].append(obs.copy())
            
            # Get MuJoCo frame
            mj_frame = env.render()
            
            # Controller forward pass
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action, _, net_info = controller.forward(obs_tensor)
            action_np = action.squeeze(0).cpu().numpy()
            
            # Extract activations
            activations = extract_activations_from_info(
                net_info, obs, config.num_muscles, 
                target_grid_size,
                target_encoder=target_encoder,
                env_info=env_info,
            )
            trajectory['activations'].append(activations)
            
            # Render combined frame
            phase = env_info.get('phase', '')
            combined = viz.render_with_mujoco(
                activations, mj_frame, layout=layout,
                step=step, phase=phase
            )
            combined_frames.append(combined)
            
            # Step environment
            obs, reward, terminated, truncated, env_info = env.step(action_np)
            
            trajectory['actions'].append(action_np)
            trajectory['rewards'].append(reward)
            trajectory['infos'].append(env_info)
            
            if terminated or truncated:
                break
    
    env.close()
    viz.close()
    
    # Save video
    if output_video and combined_frames:
        import cv2
        h, w = combined_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        
        for frame in combined_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved to {output_video}")
    
    trajectory['combined_frames'] = combined_frames
    return trajectory


# Convenience function for quick visualization
def visualize_network_live(
    checkpoint_path: str,
    xml_path: str,
    sensor_stats_path: Optional[str] = None,
    output_video: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_EPISODE_STEPS
):
    """
    Quick function to visualize a trained network.
    
    Usage:
        visualize_network_live(
            "outputs/best_controller.pt",
            "arm.xml",
            output_video="network_activity.mp4"
        )
    """
    import pickle
    from pathlib import Path
    
    # Load controller
    from visualization import load_controller
    controller, config, _ = load_controller(checkpoint_path)
    
    # Load sensor stats
    if sensor_stats_path is None:
        sensor_stats_path = Path(checkpoint_path).parent / 'sensor_stats.pkl'
    
    if Path(sensor_stats_path).exists():
        with open(sensor_stats_path, 'rb') as f:
            sensor_stats = pickle.load(f)
    else:
        print("Warning: No sensor stats found, using defaults")
        sensor_stats = None
    
    # Record
    trajectory = record_episode_with_network(
        controller, xml_path, sensor_stats,
        max_steps=max_steps,
        output_video=output_video
    )
    
    print(f"Episode length: {len(trajectory['rewards'])} steps")
    print(f"Total reward: {sum(trajectory['rewards']):.2f}")
    
    return trajectory


if __name__ == '__main__':
    # Demo with random activations
    print("Testing NetworkActivityVisualizer...")
    
    viz = NetworkActivityVisualizer(num_muscles=4, rnn_hidden_size=128, target_grid_size=4)
    
    # Generate random activations
    activations = {
        'Ia': np.random.randn(4),
        'II': np.random.randn(4),
        'Ib': np.random.randn(4),
        'rnn': np.random.randn(32),
        'alpha': np.random.rand(4),  # 0-1 for motor output
        'gamma_s': np.random.rand(4) * 2,  # 0-2 for gamma
        'gamma_d': np.random.rand(4) * 2,
        'target': np.random.rand(16),  # 4x4 grid
    }
    
    # Render frame
    frame = viz.render_frame(activations, title="Demo", step=42, phase="reach")
    
    print(f"Frame shape: {frame.shape}")
    
    # Save test image
    plt.imsave('network_viz_test.png', frame)
    print("Saved test image to network_viz_test.png")
    
    viz.close()
