"""
Utilities for MuJoCo Plant Interface, Model Parsing, Visualization, and Analysis

Plant & Model Parsing:
- MuJoCoPlant: Physics interface for muscle-driven systems
- parse_mujoco_xml: Parse MuJoCo XML to extract model information
- ParsedModel: Dataclass with joints, muscles, sensors, bodies
- get_model_dimensions: Calculate network dimensions from parsed model
- infer_muscle_sensor_mapping: Map sensors to muscles
- calibrate_sensors: Gather sensor statistics for normalization

Visualization:
- load_controller: Load trained controller from checkpoint
- evaluate_controller: Run evaluation episodes
- record_episode: Record trajectory with optional rendering
- save_video: Save frames as MP4
- plot_trajectory: Plot motor outputs and kinematics
- plot_episode_summary: Comprehensive 5-row episode analysis
- plot_reflex_connections: Visualize Ia→Alpha, II→Alpha weights
- print_weight_summary: Print weight statistics
- inspect_checkpoint: Full checkpoint analysis
- compare_controllers: Compare multiple controllers

Episode Recording (Unified):
- EpisodeRecorder: Single-pass recording with synchronized visualization
- EpisodeData: Complete episode data container
- record_and_save: Convenience function to record and save all outputs

Network Visualization (Legacy):
- NetworkActivityVisualizer: Real-time network activity display
- record_episode_with_network: Record with network activity overlay
"""

from .visualization import (
    load_controller,
    evaluate_controller,
    record_episode,
    save_video,
    plot_trajectory,
    plot_episode_summary,
    plot_reflex_connections,
    print_weight_summary,
    plot_training_curves,
    inspect_checkpoint,
    compare_controllers,
)

from .network_visualizer import (
    NetworkActivityVisualizer,
    record_episode_with_network,
)

from .episode_recorder import (
    EpisodeRecorder,
    EpisodeData,
    NetworkDiagram,
    record_and_save,
    plot_episode_summary as plot_episode_summary_new,
)

__all__ = [
    # Model parsing
    'parse_mujoco_xml',
    'ParsedModel',
    'JointInfo',
    'MuscleInfo',
    'SensorInfo',
    'BodyInfo',
    'get_model_dimensions',
    'infer_muscle_sensor_mapping',
    # Visualization
    'load_controller',
    'evaluate_controller',
    'record_episode',
    'save_video',
    'plot_trajectory',
    'plot_episode_summary',
    'plot_reflex_connections',
    'print_weight_summary',
    'plot_training_curves',
    'inspect_checkpoint',
    'compare_controllers',
    # Network visualization (legacy)
    'NetworkActivityVisualizer',
    'record_episode_with_network',
    # Episode recording (unified)
    'EpisodeRecorder',
    'EpisodeData',
    'NetworkDiagram',
    'record_and_save',
]
