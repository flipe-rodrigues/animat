"""
Unified Episode Recorder with Network Visualization

Records episodes with synchronized MuJoCo simulation and network activity.
Produces:
- Combined video with simulation + network activity side by side
- Episode summary statistics and plots
- Individual frames for inspection

This replaces the fragmented recording approach with a single-pass recorder
that captures everything together.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EpisodeData:
    """Complete episode data from a single recording pass."""

    # Observations and actions
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)

    # Network activations
    sensory_Ia: List[np.ndarray] = field(default_factory=list)
    sensory_II: List[np.ndarray] = field(default_factory=list)
    sensory_Ib: List[np.ndarray] = field(default_factory=list)
    rnn_hidden: List[np.ndarray] = field(default_factory=list)
    alpha: List[np.ndarray] = field(default_factory=list)
    gamma_static: List[np.ndarray] = field(default_factory=list)
    gamma_dynamic: List[np.ndarray] = field(default_factory=list)
    target_encoding: List[np.ndarray] = field(default_factory=list)

    # Frames
    mujoco_frames: List[np.ndarray] = field(default_factory=list)
    network_frames: List[np.ndarray] = field(default_factory=list)
    combined_frames: List[np.ndarray] = field(default_factory=list)

    # Metadata
    seed: Optional[int] = None
    target_position: Optional[np.ndarray] = None

    def to_arrays(self):
        """Convert lists to numpy arrays where applicable."""
        for attr in [
            "observations",
            "actions",
            "rewards",
            "sensory_Ia",
            "sensory_II",
            "sensory_Ib",
            "rnn_hidden",
            "alpha",
            "gamma_static",
            "gamma_dynamic",
            "target_encoding",
        ]:
            val = getattr(self, attr)
            if val and isinstance(val, list):
                setattr(self, attr, np.array(val))
        return self


class NetworkDiagram:
    """
    Renders network architecture diagram with live activations.

    Layout:
        [Target Grid]      <- Top center

    [Proprio]  [RNN]  [Motor]
      Ia II Ib  ...   α γs γd
    """

    def __init__(
        self,
        num_muscles: int = 4,
        rnn_hidden_size: int = 32,
        target_grid_size: int = 4,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100,
        show_rnn_units: int = 32,
    ):
        self.num_muscles = num_muscles
        self.rnn_hidden_size = rnn_hidden_size
        self.target_grid_size = target_grid_size
        self.show_rnn_units = min(show_rnn_units, rnn_hidden_size)

        self.figsize = figsize
        self.dpi = dpi

        # Custom black-to-orange colormap
        self.cmap = LinearSegmentedColormap.from_list(
            "black_orange", ["black", "#FF8C00"]  # Dark orange
        )
        self.norm = Normalize(vmin=0, vmax=1)

        # Layout coordinates
        self._setup_positions()

        # Figure (lazy init)
        self.fig = None
        self.ax = None
        self.has_drawn_colorbar = False

    def _setup_positions(self):
        """Compute unit positions for the diagram."""
        # Layout parameters
        self.positions = {}

        # Main row Y positions
        main_y_top = 4.0
        main_y_bottom = 0.5
        muscle_ys = np.linspace(main_y_top, main_y_bottom, self.num_muscles)

        # Proprioceptive (left, x=1-3)
        for i, sensor in enumerate(["Ia", "II", "Ib"]):
            self.positions[sensor] = [(1 + i * 0.8, y) for y in muscle_ys]

        # RNN (center, x=5-7)
        n_show = self.show_rnn_units
        rnn_cols = int(np.ceil(np.sqrt(n_show)))
        rnn_rows = int(np.ceil(n_show / rnn_cols))
        rnn_xs = np.linspace(4.5, 7.5, rnn_cols)
        rnn_ys = np.linspace(main_y_top, main_y_bottom, rnn_rows)

        rnn_pos = []
        idx = 0
        for y in rnn_ys:
            for x in rnn_xs:
                if idx < n_show:
                    rnn_pos.append((x, y))
                    idx += 1
        self.positions["rnn"] = rnn_pos

        # Motor output (right, x=9-11)
        for i, motor in enumerate(["alpha", "gamma_s", "gamma_d"]):
            self.positions[motor] = [(9 + i * 0.8, y) for y in muscle_ys]

        # Target grid (top center, x=5-7, y=5.5-7)
        grid_xs = np.linspace(5, 7, self.target_grid_size)
        grid_ys = np.linspace(5.5, 7, self.target_grid_size)
        target_pos = []
        for y in grid_ys:
            for x in grid_xs:
                target_pos.append((x, y))
        self.positions["target"] = target_pos

    def _init_figure(self):
        """Initialize matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-0.5, 12)
        self.ax.set_ylim(-0.5, 8)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def render(
        self,
        activations: Dict[str, np.ndarray],
        step: int = None,
        phase: str = None,
        target_visible: bool = True,
    ) -> np.ndarray:
        """
        Render network diagram with current activations.

        Args:
            activations: Dict with 'Ia', 'II', 'Ib', 'rnn', 'alpha',
                        'gamma_s', 'gamma_d', 'target' arrays
            step: Current step number
            phase: Trial phase
            target_visible: Whether target is visible

        Returns:
            RGB image as numpy array
        """
        if self.fig is None:
            self._init_figure()

        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-0.5, 12)
        self.ax.set_ylim(-0.5, 8)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Draw each module
        modules = ["Ia", "II", "Ib", "rnn", "alpha", "gamma_s", "gamma_d", "target"]
        radii = {
            "Ia": 0.15,
            "II": 0.15,
            "Ib": 0.15,
            "rnn": 0.12,
            "alpha": 0.15,
            "gamma_s": 0.15,
            "gamma_d": 0.15,
            "target": 0.12,
        }

        for module in modules:
            positions = self.positions.get(module, [])
            acts = activations.get(module, np.zeros(len(positions)))

            # Handle target visibility
            if module == "target" and not target_visible:
                acts = np.zeros(len(positions))

            # Sanitize activations: replace NaN/Inf with 0, clip to [0, 1]
            acts = np.nan_to_num(acts, nan=0.0, posinf=1.0, neginf=0.0)
            acts = np.clip(acts, 0.0, 1.0)

            # Ensure correct length
            if len(acts) < len(positions):
                acts = np.pad(acts, (0, len(positions) - len(acts)))
            elif len(acts) > len(positions):
                acts = acts[: len(positions)]

            r = radii.get(module, 0.15)
            circles = []
            colors = []

            for (x, y), act in zip(positions, acts):
                circles.append(plt.Circle((x, y), r))
                colors.append(self.cmap(self.norm(act)))

            if circles:
                collection = PatchCollection(
                    circles, facecolors=colors, edgecolors="none"
                )
                self.ax.add_collection(collection)

        # Draw labels
        label_style = dict(fontsize=10, fontweight="bold", ha="center", color="white")
        self.ax.text(2, 4.8, "Sensory Input", **label_style)
        self.ax.text(1, 4.4, "Ia", fontsize=8, ha="center", color="white")
        self.ax.text(1.8, 4.4, "II", fontsize=8, ha="center", color="white")
        self.ax.text(2.6, 4.4, "Ib", fontsize=8, ha="center", color="white")

        self.ax.text(6, 7.5, "Target Encoding", **label_style)
        self.ax.text(6, 4.8, "RNN Core", **label_style)

        self.ax.text(10, 4.8, "Motor Output", **label_style)
        self.ax.text(9, 4.4, "α", fontsize=9, ha="center", color="white")
        self.ax.text(9.8, 4.4, "γs", fontsize=9, ha="center", color="white")
        self.ax.text(10.6, 4.4, "γd", fontsize=9, ha="center", color="white")

        # Muscle labels
        muscle_ys = np.linspace(4.0, 0.5, self.num_muscles)
        for i, y in enumerate(muscle_ys):
            self.ax.text(
                0.5, y, f"M{i+1}", fontsize=8, ha="right", va="center", color="white"
            )
            self.ax.text(
                11.2, y, f"M{i+1}", fontsize=8, ha="left", va="center", color="white"
            )

        # Add colorbar (once)
        if not self.has_drawn_colorbar:
            self.has_drawn_colorbar = True
            sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            sm.set_array([])
            cbar = self.fig.colorbar(
                sm,
                ax=self.ax,
                orientation="vertical",
                fraction=0.012,
                pad=0.01,
                shrink=0.6,
            )
            cbar.set_label("Activation", fontsize=8, color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # Title with step/phase info
        title_parts = []
        if step is not None:
            title_parts.append(f"Step: {step}")
        if phase:
            title_parts.append(f"Phase: {phase}")
        if title_parts:
            self.ax.set_title(
                " | ".join(title_parts), fontsize=11, fontweight="bold", color="white"
            )

        # Convert to image
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        return img[:, :, :3]

    def close(self):
        """Close figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


class EpisodeRecorder:
    """
    Records complete episodes with network visualization.

    Single-pass recording ensures simulation and network activity
    are perfectly synchronized.
    """

    def __init__(
        self,
        controller: torch.nn.Module,
        xml_path: str,
        sensor_stats: Dict,
        target_grid_size: int = 4,
        target_sigma: float = 0.5,
        workspace_bounds: Optional[Dict] = None,
    ):
        self.controller = controller
        self.xml_path = xml_path
        self.sensor_stats = sensor_stats

        # Get attributes from controller's config
        self.num_muscles = controller.config.num_muscles
        num_core_units = controller.config.num_core_units
        ws_bounds = workspace_bounds or controller.config.workspace_bounds

        # Store for target encoding visualization
        self.target_grid_size = target_grid_size
        self.target_sigma = target_sigma
        self.workspace_bounds = ws_bounds

        # Create target encoder for visualization (reuse controller's if available)
        if hasattr(controller, 'target_encoder'):
            self.target_encoder = controller.target_encoder
        else:
            from models.modules import TargetEncoder
            self.target_encoder = TargetEncoder(
                grid_size=target_grid_size,
                sigma=target_sigma,
                workspace_bounds=ws_bounds,
            )

        # Network diagram renderer
        self.diagram = NetworkDiagram(
            num_muscles=self.num_muscles,
            rnn_hidden_size=num_core_units,
            target_grid_size=target_grid_size,
        )

        self.device = next(controller.parameters()).device

    def record(
        self,
        max_steps: int = 300,
        seed: Optional[int] = None,
        render_mujoco: bool = True,
        render_network: bool = True,
    ) -> EpisodeData:
        """
        Record a complete episode.

        Args:
            max_steps: Maximum episode length
            seed: Random seed for reproducibility
            render_mujoco: Whether to render MuJoCo frames
            render_network: Whether to render network diagrams

        Returns:
            EpisodeData with all recorded data
        """
        from envs.reaching import ReachingEnv

        # Create environment
        render_mode = "rgb_array" if render_mujoco else None
        env = ReachingEnv(
            self.xml_path, render_mode=render_mode, sensor_stats=self.sensor_stats
        )

        # Initialize
        self.controller.eval()
        data = EpisodeData(seed=seed)

        # Reset with seed
        obs, info = env.reset(seed=seed)
        self.controller._reset_state()

        # Store target position
        data.target_position = info.get("target_position", np.zeros(3))
        data.infos.append(info)

        with torch.no_grad():
            for step in range(max_steps):
                # Determine target visibility from observation (target is zeros when not visible)
                proprio_dim = self.num_muscles * 3
                target_xyz = obs[proprio_dim : proprio_dim + 3]
                target_visible = np.any(target_xyz != 0)

                # Render MuJoCo frame
                if render_mujoco:
                    mj_frame = env.render()
                    if mj_frame is not None:
                        data.mujoco_frames.append(mj_frame)

                # Controller forward pass
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                action, net_info = self.controller.forward(obs_tensor)
                action_np = action.squeeze(0).cpu().numpy()

                # Sanitize action: replace NaN/Inf, clip to [0, 1] for muscle activations
                action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=0.0)
                action_np = np.clip(action_np, 0.0, 1.0)

                # Extract network activations
                self._extract_activations(data, net_info, obs, target_visible)

                # Render network diagram
                if render_network:
                    activations = self._get_activations_dict(data, -1)
                    net_frame = self.diagram.render(
                        activations,
                        step=step,
                        phase=info.get("phase", ""),
                        target_visible=target_visible,
                    )
                    data.network_frames.append(net_frame)

                # Combine frames
                if (
                    render_mujoco
                    and render_network
                    and data.mujoco_frames
                    and data.network_frames
                ):
                    combined = self._combine_frames(
                        data.mujoco_frames[-1], data.network_frames[-1]
                    )
                    data.combined_frames.append(combined)

                # Store observation and action
                data.observations.append(obs.copy())
                data.actions.append(action_np.copy())

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action_np)

                data.rewards.append(reward)
                data.infos.append(info)

                if terminated or truncated:
                    break

        env.close()
        self.diagram.close()

        return data.to_arrays()

    def _extract_activations(
        self,
        data: EpisodeData,
        net_info: Dict,
        obs: np.ndarray,
        target_visible: bool,
    ):
        """Extract network activations from controller output."""

        def sanitize(arr: np.ndarray) -> np.ndarray:
            """Replace NaN/Inf with 0 and clip to [0, 1]."""
            return np.clip(
                np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
            )

        # Sensory outputs (now a tuple: spindle_Ia, spindle_II, golgi_Ib)
        if "sensory_outputs" in net_info:
            sensory = net_info["sensory_outputs"]
            data.sensory_Ia.append(sanitize(sensory[0].squeeze().cpu().numpy()))
            data.sensory_II.append(sanitize(sensory[1].squeeze().cpu().numpy()))
            data.sensory_Ib.append(sanitize(sensory[2].squeeze().cpu().numpy()))

        # Core state (renamed from rnn_hidden)
        if "core_state" in net_info and net_info["core_state"] is not None:
            data.rnn_hidden.append(
                sanitize(net_info["core_state"].squeeze().cpu().numpy())
            )

        # Motor outputs
        if "alpha" in net_info:
            data.alpha.append(sanitize(net_info["alpha"].squeeze().cpu().numpy()))
        if "gamma_static" in net_info:
            data.gamma_static.append(
                sanitize(net_info["gamma_static"].squeeze().cpu().numpy())
            )
        if "gamma_dynamic" in net_info:
            data.gamma_dynamic.append(
                sanitize(net_info["gamma_dynamic"].squeeze().cpu().numpy())
            )

        # Target encoding
        if target_visible:
            # Get target XYZ from observation
            proprio_dim = self.num_muscles * 3
            target_xyz = obs[proprio_dim : proprio_dim + 3]
            target_tensor = torch.tensor(target_xyz, dtype=torch.float32).unsqueeze(0)
            encoded = self.target_encoder(target_tensor).squeeze().detach().cpu().numpy()
            data.target_encoding.append(sanitize(encoded))
        else:
            # Target not visible - zeros
            num_target_units = self.target_grid_size ** 2
            data.target_encoding.append(np.zeros(num_target_units))

    def _get_activations_dict(
        self, data: EpisodeData, idx: int
    ) -> Dict[str, np.ndarray]:
        """Get activations dict for diagram rendering."""
        return {
            "Ia": (
                data.sensory_Ia[idx] if data.sensory_Ia else np.zeros(self.num_muscles)
            ),
            "II": (
                data.sensory_II[idx] if data.sensory_II else np.zeros(self.num_muscles)
            ),
            "Ib": (
                data.sensory_Ib[idx] if data.sensory_Ib else np.zeros(self.num_muscles)
            ),
            "rnn": data.rnn_hidden[idx] if data.rnn_hidden else np.zeros(32),
            "alpha": data.alpha[idx] if data.alpha else np.zeros(self.num_muscles),
            "gamma_s": (
                data.gamma_static[idx]
                if data.gamma_static
                else np.zeros(self.num_muscles)
            ),
            "gamma_d": (
                data.gamma_dynamic[idx]
                if data.gamma_dynamic
                else np.zeros(self.num_muscles)
            ),
            "target": (
                data.target_encoding[idx] if data.target_encoding else np.zeros(16)
            ),
        }

    def _combine_frames(
        self,
        mj_frame: np.ndarray,
        net_frame: np.ndarray,
    ) -> np.ndarray:
        """Combine MuJoCo and network frames side by side."""
        import cv2

        # Resize MuJoCo frame to match network frame height
        target_h = net_frame.shape[0]
        scale = target_h / mj_frame.shape[0]
        new_w = int(mj_frame.shape[1] * scale)
        mj_resized = cv2.resize(mj_frame, (new_w, target_h))

        # Stack horizontally
        combined = np.concatenate([mj_resized, net_frame], axis=1)
        return combined


def record_and_save(
    controller: torch.nn.Module,
    xml_path: str,
    sensor_stats: Dict,
    output_dir: str,
    max_steps: int = 300,
    seed: Optional[int] = None,
    fps: int = 60,
) -> EpisodeData:
    """
    Record episode and save all outputs.

    Saves:
    - combined_video.mp4: Side-by-side simulation + network
    - episode_summary.png: Summary plots

    Args:
        controller: Trained controller
        xml_path: MuJoCo XML path
        sensor_stats: Sensor normalization stats
        output_dir: Directory to save outputs
        max_steps: Max episode length
        seed: Random seed
        fps: Video framerate

    Returns:
        EpisodeData with all recorded data
    """
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get target grid size from controller's config
    target_grid_size = controller.config.target_grid_size
    target_sigma = controller.config.target_sigma
    workspace_bounds = controller.config.workspace_bounds

    # Record
    recorder = EpisodeRecorder(
        controller=controller,
        xml_path=xml_path,
        sensor_stats=sensor_stats,
        target_grid_size=target_grid_size,
        target_sigma=target_sigma,
        workspace_bounds=workspace_bounds,
    )

    print(f"Recording episode (seed={seed})...")
    data = recorder.record(max_steps=max_steps, seed=seed)

    # Save combined video
    if data.combined_frames is not None and len(data.combined_frames) > 0:
        video_path = str(output_path / "combined_video.mp4")
        _save_video(data.combined_frames, video_path, fps)
        print(f"  Saved: {video_path}")

    # Save summary plot
    plot_path = str(output_path / "episode_summary.png")
    plot_episode_summary(data, output_path=plot_path)
    print(f"  Saved: {plot_path}")

    # Print stats
    total_reward = sum(data.rewards) if hasattr(data.rewards, "__iter__") else 0
    final_phase = data.infos[-1].get("phase", "unknown") if data.infos else "unknown"
    print(f"\nEpisode Summary:")
    print(f"  Length: {len(data.rewards)} steps")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Phase: {final_phase}")
    print(f"  Target Position: {data.target_position}")

    return data


def _save_video(frames: List[np.ndarray], path: str, fps: int = 60):
    """Save frames as video."""
    import cv2

    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def plot_episode_summary(
    data: EpisodeData,
    output_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot comprehensive episode summary with phase shading.

    Layout (6 rows x 2 cols):
    - Row 1: Muscle Lengths (Type II) | Muscle Velocities (Type Ia)
    - Row 2: Muscle Forces (Type Ib) | Target Grid Encoding
    - Row 3: Motor Output (full width) - alpha, gamma_static, gamma_dynamic
    - Row 4: Hand Kinematics (full width)
    - Row 5: Reward Components (full width)
    - Row 6: Cumulative Reward | Phase Timeline
    """
    import matplotlib.patches as patches

    num_steps = len(data.rewards) if hasattr(data.rewards, "__len__") else 0
    if num_steps == 0:
        print("Warning: No data to plot")
        return

    time = np.arange(num_steps) * 0.01  # Assuming 10ms timestep
    num_muscles = data.sensory_Ia.shape[1] if hasattr(data.sensory_Ia, "shape") else 4

    # Phase colors and map
    phase_map = {"pre_delay": 0, "reach": 1, "hold": 2, "post_delay": 3, "done": 4}
    phase_colors = {
        "pre_delay": "#FFE4B5",
        "reach": "#E6F3FF",
        "hold": "#E6FFE6",
        "post_delay": "#FFE6FF",
        "done": "#F0F0F0",
    }

    # Extract phase info
    phases = [info.get("phase", "unknown") for info in data.infos[1 : num_steps + 1]]

    def add_phase_shading(ax, phases, time):
        """Add colored background for each phase."""
        if len(phases) == 0:
            return
        current_phase = phases[0]
        start_idx = 0

        for i, phase in enumerate(phases):
            if phase != current_phase or i == len(phases) - 1:
                end_idx = i if phase != current_phase else i + 1
                if current_phase in phase_colors:
                    ax.axvspan(
                        time[start_idx],
                        time[min(end_idx, len(time) - 1)],
                        alpha=0.3,
                        color=phase_colors[current_phase],
                        zorder=0,
                    )
                current_phase = phase
                start_idx = i

    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.25)

    # ===== Row 1: Muscle Lengths (Type II) =====
    ax1 = fig.add_subplot(gs[0, 0])
    if hasattr(data.sensory_II, "shape"):
        for i in range(num_muscles):
            ax1.plot(time, data.sensory_II[:, i], label=f"Muscle {i+1}", linewidth=1.5)
    add_phase_shading(ax1, phases, time)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Length (norm)")
    ax1.set_title("Muscle Lengths (Type II Sensory Input)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ===== Row 1: Muscle Velocities (Type Ia) =====
    ax2 = fig.add_subplot(gs[0, 1])
    if hasattr(data.sensory_Ia, "shape"):
        for i in range(num_muscles):
            ax2.plot(time, data.sensory_Ia[:, i], label=f"Muscle {i+1}", linewidth=1.5)
    add_phase_shading(ax2, phases, time)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (norm)")
    ax2.set_title("Muscle Velocities (Type Ia Sensory Input)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ===== Row 2: Muscle Forces (Type Ib) =====
    ax3 = fig.add_subplot(gs[1, 0])
    if hasattr(data.sensory_Ib, "shape"):
        for i in range(num_muscles):
            ax3.plot(time, data.sensory_Ib[:, i], label=f"Muscle {i+1}", linewidth=1.5)
    add_phase_shading(ax3, phases, time)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Force (norm)")
    ax3.set_title("Muscle Forces (Type Ib / GTO Input)", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ===== Row 2: Target Grid Encoding =====
    ax4 = fig.add_subplot(gs[1, 1])
    if hasattr(data.target_encoding, "shape") and data.target_encoding.shape[0] > 0:
        target_dim = data.target_encoding.shape[1]
        im = ax4.imshow(
            data.target_encoding.T,
            aspect="auto",
            cmap="viridis",
            extent=[time[0], time[-1], 0, target_dim],
        )
        plt.colorbar(im, ax=ax4, label="Activation")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Target Grid Unit")
    ax4.set_title("Target Grid Encoding (Gaussian-tuned)", fontweight="bold")

    # ===== Row 3: Motor Output - Alpha (left) and Gamma (right) =====
    ax_alpha = fig.add_subplot(gs[2, 0])
    ax_gamma = fig.add_subplot(gs[2, 1])

    # Plot alpha (muscle activations)
    if hasattr(data.alpha, "shape") and data.alpha.shape[0] > 0:
        for i in range(num_muscles):
            ax_alpha.plot(
                time, data.alpha[:num_steps, i], label=f"M{i+1}", linewidth=1.5
            )

    add_phase_shading(ax_alpha, phases, time)
    ax_alpha.set_xlabel("Time (s)")
    ax_alpha.set_ylabel("Activation")
    ax_alpha.set_title("Alpha Motor Output (α)", fontweight="bold")
    ax_alpha.legend(loc="upper right", fontsize=8)
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.set_ylim(-0.1, 1.1)

    # Plot gamma_static and gamma_dynamic
    if hasattr(data.gamma_static, "shape") and data.gamma_static.shape[0] > 0:
        for i in range(num_muscles):
            ax_gamma.plot(
                time,
                data.gamma_static[:num_steps, i],
                label=f"γs{i+1}",
                linewidth=1.5,
                linestyle="-",
            )

    if hasattr(data.gamma_dynamic, "shape") and data.gamma_dynamic.shape[0] > 0:
        for i in range(num_muscles):
            ax_gamma.plot(
                time,
                data.gamma_dynamic[:num_steps, i],
                label=f"γd{i+1}",
                linewidth=1.5,
                linestyle="--",
            )

    add_phase_shading(ax_gamma, phases, time)
    ax_gamma.set_xlabel("Time (s)")
    ax_gamma.set_ylabel("Activation")
    ax_gamma.set_title("Gamma Motor Output (γs, γd)", fontweight="bold")
    ax_gamma.legend(loc="upper right", fontsize=7, ncol=2)
    ax_gamma.grid(True, alpha=0.3)
    ax_gamma.set_ylim(-0.1, 2.1)  # Gamma can go up to 2

    # ===== Row 4: Hand Kinematics (full width) =====
    ax5 = fig.add_subplot(gs[3, :])

    # Distance to target
    distances = [
        info.get("distance_to_target", np.nan) for info in data.infos[1 : num_steps + 1]
    ]
    ln1 = ax5.plot(
        time, distances[:num_steps], "b-", linewidth=2, label="Distance to Target"
    )
    ax5.axhline(y=0.05, color="b", linestyle="--", alpha=0.5, label="Reach Threshold")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Distance (m)", color="b")
    ax5.tick_params(axis="y", labelcolor="b")

    # Hand speed on secondary axis
    hand_positions = np.array(
        [info.get("hand_position", [0, 0, 0]) for info in data.infos[1 : num_steps + 1]]
    )
    if len(hand_positions) > 1:
        hand_velocity = np.diff(hand_positions, axis=0) / 0.01
        hand_speed = np.linalg.norm(hand_velocity, axis=1)
        hand_speed = np.concatenate([[0], hand_speed])
    else:
        hand_speed = np.zeros(num_steps)

    ax5b = ax5.twinx()
    ln2 = ax5b.plot(
        time, hand_speed[:num_steps], "r-", linewidth=2, alpha=0.7, label="Hand Speed"
    )
    ax5b.set_ylabel("Speed (m/s)", color="r")
    ax5b.tick_params(axis="y", labelcolor="r")

    add_phase_shading(ax5, phases, time)

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc="upper right")
    ax5.set_title("Hand Kinematics: Distance to Target & Speed", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # ===== Row 5: Reward Components (full width) =====
    ax6 = fig.add_subplot(gs[4, :])

    # Compute reward components
    distance_reward = -np.array(distances[:num_steps])
    reach_bonus = (np.array(distances[:num_steps]) < 0.05).astype(float) * 0.5

    if hasattr(data.alpha, "shape"):
        energy_penalty = -0.01 * np.sum(data.alpha**2, axis=1)
    else:
        energy_penalty = np.zeros(num_steps)

    ax6.plot(
        time, distance_reward, "b-", linewidth=1.5, label="Distance Reward", alpha=0.8
    )
    ax6.plot(time, reach_bonus, "g-", linewidth=1.5, label="Reach Bonus", alpha=0.8)
    ax6.plot(
        time,
        energy_penalty[:num_steps],
        "r-",
        linewidth=1.5,
        label="Energy Penalty",
        alpha=0.8,
    )
    ax6.plot(time, data.rewards, "k-", linewidth=2, label="Total Reward")

    add_phase_shading(ax6, phases, time)
    ax6.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Reward")
    ax6.set_title("Reward Components Over Time", fontweight="bold")
    ax6.legend(loc="upper right", ncol=4)
    ax6.grid(True, alpha=0.3)

    # ===== Row 6: Cumulative Reward =====
    ax7 = fig.add_subplot(gs[5, 0])
    ax7.plot(time, np.cumsum(data.rewards), "k-", linewidth=2)
    add_phase_shading(ax7, phases, time)
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Cumulative Reward")
    ax7.set_title("Cumulative Reward", fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # Final success indicator
    final_phase = phases[-1] if phases else "unknown"
    success = final_phase == "done"
    result_text = "SUCCESS" if success else f"Phase: {final_phase}"
    result_color = "green" if success else "orange"
    ax7.text(
        0.98,
        0.95,
        result_text,
        transform=ax7.transAxes,
        fontsize=12,
        fontweight="bold",
        color=result_color,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ===== Row 6: Phase Timeline =====
    ax8 = fig.add_subplot(gs[5, 1])
    phase_numeric = [phase_map.get(p, -1) for p in phases]
    ax8.plot(time, phase_numeric, "k-", linewidth=2, drawstyle="steps-post")
    ax8.fill_between(time, phase_numeric, step="post", alpha=0.3)
    ax8.set_yticks(list(phase_map.values()))
    ax8.set_yticklabels(list(phase_map.keys()))
    ax8.set_xlabel("Time (s)")
    ax8.set_title("Trial Phase Timeline", fontweight="bold")
    ax8.grid(True, alpha=0.3, axis="x")

    # Add phase color legend
    legend_patches = [
        patches.Patch(color=color, label=phase, alpha=0.5)
        for phase, color in phase_colors.items()
    ]
    ax8.legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.suptitle("Episode Summary", fontsize=14, fontweight="bold", y=1.0)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Episode summary saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Demo with random data
    print("Testing NetworkDiagram...")

    diagram = NetworkDiagram(
        num_muscles=4,
        rnn_hidden_size=32,
        target_grid_size=4,
    )

    activations = {
        "Ia": np.random.randn(4),
        "II": np.random.randn(4),
        "Ib": np.random.randn(4),
        "rnn": np.random.randn(32),
        "alpha": np.random.rand(4),
        "gamma_s": np.random.rand(4),
        "gamma_d": np.random.rand(4),
        "target": np.random.rand(4**2),
    }

    frame = diagram.render(activations, step=42, phase="reach", target_visible=True)
    print(f"Frame shape: {frame.shape}")

    plt.imsave("episode_recorder_test.png", frame)
    print("Saved test image to episode_recorder_test.png")

    diagram.close()
