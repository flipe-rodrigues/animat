#!/usr/bin/env python3
"""
Main Entry Point for Muscle Arm RNN Control

This script provides a unified interface for:
- Training controllers (CMA-ES or Distillation)
- Evaluating trained controllers
- Recording and visualizing episodes
- Analyzing model architecture
"""

import os
# Fix OpenMP duplicate library error on Windows
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import sys
from pathlib import Path
import json
import pickle

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

# Default MuJoCo model path
DEFAULT_XML_PATH = "mujoco/arm.xml"

from core.config import ModelConfig
from core.constants import (
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_CALIBRATION_EPISODES,
    DEFAULT_CALIBRATION_STEPS,
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_CMAES_SIGMA,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_INSPECTION_EVERY,
    DEFAULT_TEACHER_EPOCHS,
    DEFAULT_STUDENT_EPOCHS,
    DEFAULT_VIDEO_FPS,
    DEFAULT_TARGET_GRID_SIZE,
)
from envs.plant import parse_mujoco_xml, get_model_dimensions, calibrate_sensors
from envs.reaching import ReachingEnv
from models import RNNController, MLPController, create_controller
from training.train_cmaes import run_cmaes_training
from training.train_distillation import run_distillation_training
from utils.visualization import (
    load_controller,
    evaluate_controller,
    record_episode,
    save_video,
    plot_trajectory,
    plot_training_curves,
    print_weight_summary,
    plot_all_weights,
)


def cmd_info(args):
    """Show information about a MuJoCo model."""
    parsed = parse_mujoco_xml(args.xml_path)
    dims = get_model_dimensions(parsed)

    print(f"\n{'='*60}")
    print(f"Model: {parsed.model_name}")
    print(f"{'='*60}")

    print(f"\nTimestep: {parsed.timestep}s")

    print(f"\nJoints ({parsed.num_joints}):")
    for j in parsed.joints:
        range_str = f"[{j.range[0]:.1f}, {j.range[1]:.1f}]" if j.range else "unlimited"
        print(f"  - {j.name}: {j.joint_type}, range={range_str}")

    print(f"\nMuscles ({parsed.num_muscles}):")
    for m in parsed.muscles:
        print(f"  - {m.name}: force={m.force}N, ctrl_range={m.ctrl_range}")

    print(f"\nSensors ({parsed.num_sensors}):")
    for s in parsed.sensors:
        print(f"  - {s.name}: type={s.sensor_type}, target={s.target}")

    print(f"\nBodies ({len(parsed.bodies)}):")
    for b in parsed.bodies:
        mocap_str = " (mocap)" if b.is_mocap else ""
        print(f"  - {b.name}: pos={b.pos}{mocap_str}")

    print(f"\nNetwork Dimensions:")
    for k, v in dims.items():
        print(f"  {k}: {v}")

    # Create model config to show expected observation/action dims
    config = ModelConfig(
        num_muscles=parsed.num_muscles,
        num_sensors=parsed.num_sensors,
        num_target_units=DEFAULT_TARGET_GRID_SIZE ** 2,
    )

    print(f"\nExpected Environment Interface:")
    print(f"  Observation dim: {config.input_size}")
    print(f"  Action dim: {config.action_size}")


def cmd_calibrate(args):
    """Run sensor calibration and save statistics."""
    print(f"Calibrating sensors for {args.xml_path}...")

    stats = calibrate_sensors(
        args.xml_path, num_episodes=args.episodes, max_steps=args.max_steps
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nSensor statistics saved to {output_path}")
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def cmd_train(args):
    """Train a controller."""
    if args.method == "cmaes":
        import multiprocessing as mp
        num_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
        print(f"Starting CMA-ES training with {num_workers} workers...")
        results = run_cmaes_training(
            xml_path=args.xml_path,
            output_dir=args.output_dir,
            num_generations=args.generations,
            population_size=args.population,
            sigma_init=args.sigma,
            num_workers=args.workers,
            use_multiprocessing=not args.no_multiprocessing,
            calibration_episodes=args.calibration_episodes,
            save_checkpoint_every=args.save_checkpoint_every,
            inspection_every=args.inspection_every,
        )

    elif args.method == "distillation":
        print(f"Starting distillation training...")
        results = run_distillation_training(
            xml_path=args.xml_path,
            output_dir=args.output_dir,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            calibration_episodes=args.calibration_episodes,
        )

    print(f"\nTraining complete!")
    print(f"Results saved to {args.output_dir}")


def cmd_evaluate(args):
    """Evaluate a trained controller."""
    # Load sensor stats
    stats_path = Path(args.checkpoint).parent / "sensor_stats.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            sensor_stats = pickle.load(f)
    else:
        print(f"Warning: sensor_stats.pkl not found at {stats_path}")
        print("Running quick calibration...")
        sensor_stats = calibrate_sensors(args.xml_path, num_episodes=20)

    # Load controller
    controller_type = "mlp" if "mlp" in args.checkpoint.lower() else "rnn"
    controller, config, checkpoint = load_controller(args.checkpoint, controller_type)

    print(f"Loaded {controller_type.upper()} controller from {args.checkpoint}")
    print(
        f"Controller has {sum(p.numel() for p in controller.parameters())} parameters"
    )

    # Evaluate
    print(f"\nEvaluating over {args.episodes} episodes...")
    results = evaluate_controller(
        controller=controller,
        xml_path=args.xml_path,
        sensor_stats=sensor_stats,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=True,
    )

    print(f"\n{'='*40}")
    print("Evaluation Results")
    print(f"{'='*40}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} steps")
    print(f"Mean Final Distance: {results['mean_final_distance']:.4f}m")

    if results["mean_reach_time"] is not None:
        print(f"Mean Reach Time: {results['mean_reach_time']:.1f} steps")

    # Save results
    if args.save_results:
        output_path = Path(args.checkpoint).parent / "eval_results.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    k: v
                    for k, v in results.items()
                    if k not in ["episode_rewards", "episode_lengths"]
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {output_path}")


def cmd_record(args):
    """Record and visualize an episode."""
    # Load sensor stats
    stats_path = Path(args.checkpoint).parent / "sensor_stats.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            sensor_stats = pickle.load(f)
    else:
        print("Running quick calibration...")
        sensor_stats = calibrate_sensors(args.xml_path, num_episodes=20)

    # Load controller
    controller_type = "mlp" if "mlp" in args.checkpoint.lower() else "rnn"
    controller, config, _ = load_controller(args.checkpoint, controller_type)

    print(f"Recording episode...")
    trajectory = record_episode(
        controller=controller,
        xml_path=args.xml_path,
        sensor_stats=sensor_stats,
        max_steps=args.max_steps,
    )

    # Save video
    if args.video and trajectory["frames"]:
        save_video(trajectory["frames"], args.video)

    # Plot trajectory
    if args.plot:
        plot_trajectory(trajectory, output_path=args.plot, show=not args.no_display)
    elif not args.no_display:
        plot_trajectory(trajectory, show=True)

    # Summary
    total_reward = sum(trajectory["rewards"])
    final_phase = trajectory["infos"][-1].get("phase", "unknown")
    final_distance = trajectory["infos"][-1].get("distance_to_target", float("nan"))

    print(f"\nEpisode Summary:")
    print(f"  Length: {len(trajectory['rewards'])} steps")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Phase: {final_phase}")
    print(f"  Final Distance: {final_distance:.4f}m")


def cmd_plot_history(args):
    """Plot training history."""
    plot_training_curves(
        args.history_path, output_path=args.output, show=not args.no_display
    )


def cmd_inspect(args):
    """Inspect network weights and optionally run episode analysis."""

    # If xml_path provided, do full inspection with episode
    if hasattr(args, "xml_path") and args.xml_path:
        from utils.visualization import inspect_checkpoint

        inspect_checkpoint(
            checkpoint_path=args.checkpoint,
            xml_path=args.xml_path,
            output_dir=args.output_dir,
            num_episodes=args.episodes if hasattr(args, "episodes") else 1,
            max_steps=args.max_steps if hasattr(args, "max_steps") else 300,
            show=not args.no_display,
        )
        return

    # Otherwise just do weight inspection
    controller_type = "mlp" if "mlp" in args.checkpoint.lower() else "rnn"
    controller, config, checkpoint = load_controller(args.checkpoint, controller_type)

    print(f"Loaded {controller_type.upper()} controller from {args.checkpoint}")
    print(f"Parameters: {sum(p.numel() for p in controller.parameters()):,}")

    if "fitness" in checkpoint:
        print(f"Training fitness: {checkpoint['fitness']:.3f}")
    if "generation" in checkpoint:
        print(f"Training generation: {checkpoint['generation']}")

    # Print summary
    print_weight_summary(controller)

    # Generate plots if output directory specified
    if args.output_dir:
        plot_all_weights(controller, args.output_dir, show=not args.no_display)
    elif not args.no_display:
        # Show plots interactively
        from utils.visualization import (
            plot_weight_distributions,
            plot_reflex_connections,
            plot_sensory_weights,
            plot_rnn_weights,
        )

        if args.weights:
            plot_weight_distributions(controller, show=True)
        if args.reflex:
            plot_reflex_connections(controller, show=True)
        if args.sensory:
            plot_sensory_weights(controller, show=True)
        if args.rnn:
            plot_rnn_weights(controller, show=True)

        # If no specific flag, show all
        if not any([args.weights, args.reflex, args.sensory, args.rnn]):
            plot_reflex_connections(controller, show=True)
            plot_sensory_weights(controller, show=True)


def cmd_visualize(args):
    """Visualize network activity during simulation."""
    from utils.episode_recorder import record_and_save

    # Load sensor stats
    stats_path = Path(args.checkpoint).parent / "sensor_stats.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            sensor_stats = pickle.load(f)
    else:
        print("Warning: sensor_stats.pkl not found, running calibration...")
        sensor_stats = calibrate_sensors(args.xml_path, num_episodes=20)

    # Load controller
    controller_type = "mlp" if "mlp" in args.checkpoint.lower() else "rnn"
    controller, config, _ = load_controller(args.checkpoint, controller_type)

    print(f"Loaded {controller_type.upper()} controller")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.checkpoint).parent / "visualizations"
    
    # Record with unified recorder
    data = record_and_save(
        controller=controller,
        xml_path=args.xml_path,
        sensor_stats=sensor_stats,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        seed=args.seed if hasattr(args, 'seed') else None,
        fps=args.fps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Muscle Arm RNN Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model info
  python run.py info arm.xml
  
  # Train with CMA-ES
  python run.py train arm.xml --method cmaes --generations 500
  
  # Train with distillation
  python run.py train arm.xml --method distillation --teacher-epochs 100
  
  # Evaluate a trained controller
  python run.py evaluate arm.xml outputs/cmaes/best_controller_final.pt
  
  # Record and visualize an episode
  python run.py record arm.xml outputs/cmaes/best_controller_final.pt --video episode.mp4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("xml_path", nargs="?", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML file")

    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate sensors")
    cal_parser.add_argument("xml_path", nargs="?", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML file")
    cal_parser.add_argument(
        "--output", "-o", default="sensor_stats.pkl", help="Output path"
    )
    cal_parser.add_argument(
        "--episodes", type=int, default=DEFAULT_CALIBRATION_EPISODES, help="Calibration episodes"
    )
    cal_parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_CALIBRATION_STEPS, help="Max steps per episode"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a controller")
    train_parser.add_argument("xml_path", nargs="?", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML file")
    train_parser.add_argument(
        "--method",
        choices=["cmaes", "distillation"],
        default="cmaes",
        help="Training method",
    )
    train_parser.add_argument(
        "--output-dir", default="outputs", help="Output directory"
    )
    train_parser.add_argument(
        "--calibration-episodes",
        type=int,
        default=DEFAULT_CALIBRATION_EPISODES // 2,
        help="Episodes for sensor calibration",
    )

    # CMA-ES specific
    train_parser.add_argument(
        "--generations", type=int, default=DEFAULT_NUM_GENERATIONS, help="CMA-ES generations"
    )
    train_parser.add_argument(
        "--population", type=int, default=DEFAULT_POPULATION_SIZE, help="CMA-ES population size"
    )
    train_parser.add_argument(
        "--sigma", type=float, default=DEFAULT_CMAES_SIGMA, help="CMA-ES initial sigma"
    )
    train_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: cpu_count - 1)",
    )
    train_parser.add_argument(
        "--no-multiprocessing", action="store_true", help="Disable multiprocessing"
    )
    train_parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Save checkpoint every N generations",
    )
    train_parser.add_argument(
        "--inspection-every",
        type=int,
        default=DEFAULT_INSPECTION_EVERY,
        help="Run full inspection every N generations (0=disabled)",
    )

    # Distillation specific
    train_parser.add_argument(
        "--teacher-epochs", type=int, default=DEFAULT_TEACHER_EPOCHS, help="Teacher epochs"
    )
    train_parser.add_argument(
        "--student-epochs", type=int, default=DEFAULT_STUDENT_EPOCHS, help="Student epochs"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a controller")
    eval_parser.add_argument("checkpoint", help="Path to controller checkpoint")
    eval_parser.add_argument("--xml", dest="xml_path", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML file")
    eval_parser.add_argument(
        "--episodes", type=int, default=DEFAULT_CALIBRATION_EPISODES, help="Evaluation episodes"
    )
    eval_parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Max steps per episode"
    )
    eval_parser.add_argument(
        "--save-results", action="store_true", help="Save results to file"
    )

    # Record command
    rec_parser = subparsers.add_parser("record", help="Record an episode")
    rec_parser.add_argument("checkpoint", help="Path to controller checkpoint")
    rec_parser.add_argument("--xml", dest="xml_path", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML file")
    rec_parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Max steps")
    rec_parser.add_argument("--video", help="Output video path")
    rec_parser.add_argument("--plot", help="Output plot path")
    rec_parser.add_argument(
        "--no-display", action="store_true", help="Don't show plots"
    )

    # Plot history command
    plot_parser = subparsers.add_parser("plot", help="Plot training history")
    plot_parser.add_argument("history_path", help="Path to history JSON file")
    plot_parser.add_argument("--output", "-o", help="Output image path")
    plot_parser.add_argument(
        "--no-display", action="store_true", help="Don't show plot"
    )

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect network weights and run episode analysis"
    )
    inspect_parser.add_argument("checkpoint", help="Path to controller checkpoint")
    inspect_parser.add_argument(
        "--xml",
        dest="xml_path",
        help="MuJoCo XML for episode analysis (enables full inspection)",
    )
    inspect_parser.add_argument("--output-dir", "-o", help="Output directory for plots")
    inspect_parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to analyze"
    )
    inspect_parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Max steps per episode"
    )
    inspect_parser.add_argument(
        "--no-display", action="store_true", help="Don't show plots"
    )
    inspect_parser.add_argument(
        "--weights", action="store_true", help="Show weight distributions"
    )
    inspect_parser.add_argument(
        "--reflex", action="store_true", help="Show reflex connections"
    )
    inspect_parser.add_argument(
        "--sensory", action="store_true", help="Show sensory weights"
    )
    inspect_parser.add_argument("--rnn", action="store_true", help="Show RNN weights")

    # Visualize command (network activity)
    vis_parser = subparsers.add_parser(
        "visualize", help="Visualize network activity during simulation"
    )
    vis_parser.add_argument("checkpoint", help="Path to controller checkpoint")
    vis_parser.add_argument("--xml", dest="xml_path", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML file")
    vis_parser.add_argument(
        "--output", "-o", default=None, help="Output directory (default: checkpoint_dir/visualizations)"
    )
    vis_parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Max simulation steps"
    )
    vis_parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video framerate")
    vis_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Dispatch to command
    if args.command == "info":
        cmd_info(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "record":
        cmd_record(args)
    elif args.command == "plot":
        cmd_plot_history(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
