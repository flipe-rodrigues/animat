import argparse
import jax
import jax.numpy as jnp
from evojax.trainer import Trainer

from policy import TanhRNNPolicy
from task import LimbReachingTask
from algo import get_optimizer


def train(config):
    """Train the policy using PGPE.

    Args:
        config: Training configuration.
    """
    # Set up the training task
    train_task = LimbReachingTask(
        xml_file=config.xml_file,
        max_steps=config.max_steps,
        target_change_prob=config.target_change_prob,
    )

    # Set up the test task (using same configuration as training task)
    test_task = LimbReachingTask(
        xml_file=config.xml_file,
        max_steps=config.max_steps,
        target_change_prob=config.target_change_prob,
    )

    # Set up the policy
    policy = TanhRNNPolicy(hidden_dim=config.hidden_dim)

    # Set up the optimizer
    solver = get_optimizer(
        policy=policy,
        pop_size=config.pop_size,
        init_stdev=config.init_stdev,
        seed=config.seed,
    )

    # Set up the trainer
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        test_n_repeats=config.n_repeats,
        n_evaluations=config.n_test,
        seed=config.seed,
        debug=config.debug,
    )

    # Start training
    best_params = trainer.run(demo_mode=config.demo_mode)

    # Save the best parameters
    if config.save_path:
        import pickle
        import os

        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
        with open(config.save_path, "wb") as f:
            pickle.dump(best_params, f)
        print(f"Saved best parameters to {config.save_path}")


import os


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument(
        "--xml_file",
        type=str,
        default="arm_model.xml",
        help="Path to the MuJoCo XML file.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode."
    )
    parser.add_argument(
        "--target_change_prob",
        type=float,
        default=0.1,
        help="Probability of changing target position each step.",
    )

    # Policy parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension of the RNN."
    )

    # Training parameters
    parser.add_argument(
        "--pop_size", type=int, default=64, help="Population size for PGPE."
    )
    parser.add_argument(
        "--init_stdev",
        type=float,
        default=0.1,
        help="Initial standard deviation for PGPE.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval."
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Number of repeats for each evaluation.",
    )
    parser.add_argument(
        "--test_interval", type=int, default=50, help="Testing interval."
    )
    parser.add_argument(
        "--n_test", type=int, default=10, help="Number of test episodes."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument(
        "--demo_mode", action="store_true", help="Demo mode (no training)."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/best_policy.pkl",
        help="Path to save the best policy parameters.",
    )

    config = parser.parse_args()
    train(config)


if __name__ == "__main__":
    main()
