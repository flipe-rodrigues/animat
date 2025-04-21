"""
Main script for training and evaluating the RNN controller for the arm reaching task.
"""
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
from typing import Dict, Any

from environment import ArmReachingEnv
from rnn_model import SimpleRNN
from evolutionary_trainer import EvolutionaryTrainer

def train(args):
    """
    Train the RNN controller.
    
    Args:
        args: Command-line arguments
    """
    print("Initializing training...")
    
    # Set random seed - compatible with JAX 0.5.2
    key = jax.random.PRNGKey(args.seed)
    
    # Create environment
    env = ArmReachingEnv(
        model_path=args.model_path,
        episode_length=args.steps_per_target,
        render=args.render
    )
    
    # Create trainer
    trainer = EvolutionaryTrainer(
        env=env,
        popsize=args.popsize,
        hidden_size=args.hidden_size,
        n_targets=args.n_targets,
        steps_per_target=args.steps_per_target,
        save_path=args.save_path,
        seed=args.seed
    )
    
    # Train the model
    print(f"Starting training for {args.n_generations} generations...")
    results = trainer.train(
        n_generations=args.n_generations,
        log_interval=args.log_interval
    )
    
    print(f"Training complete! Best fitness: {results['best_fitness']:.4f}")
    
    # Save training history
    history_path = os.path.join(args.save_path, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(results['history'], f)
    
    print(f"Training history saved to {history_path}")
    
    return results

def evaluate(args):
    """
    Evaluate a trained RNN controller.
    
    Args:
        args: Command-line arguments
    """
    print("Initializing evaluation...")
    
    # Set random seed - compatible with JAX 0.5.2
    key = jax.random.PRNGKey(args.seed)
    
    # Create environment
    env = ArmReachingEnv(
        model_path=args.model_path,
        episode_length=args.steps_per_target,
        render=args.render
    )
    
    # Create RNN
    rnn = SimpleRNN(
        input_size=env.input_dim,
        hidden_size=args.hidden_size,
        output_size=env.output_dim
    )
    
    # Load parameters
    print(f"Loading model from {args.load_model}...")
    params = None
    try:
        with open(args.load_model, 'rb') as f:
            params_np = pickle.load(f)
            # Convert to JAX arrays - compatible with JAX 0.5.2
            params = {k: jnp.array(v) for k, v in params_np.items()}
    except (IOError, pickle.PickleError) as e:
        print(f"Error loading model: {e}")
    
    if params is None:
        raise ValueError(f"Failed to load model from {args.load_model}")
    
    # Define RNN policy function
    def policy_fn(params, obs, h_state):
        action, h_state = rnn.predict(params, obs, h_state)
        return action, h_state
    
    # Evaluate on multiple targets
    total_distance = 0.0
    total_reward = 0.0
    n_targets = args.eval_targets if args.eval_targets else args.n_targets
    
    print(f"Evaluating on {n_targets} targets...")
    for i in range(n_targets):
        # Generate new random key - compatible with JAX 0.5.2
        key, subkey = jax.random.split(key)
        
        # Reset the environment
        observation = env.reset(subkey)
        
        # Initialize hidden state
        h_state = jnp.zeros((params['w_hh'].shape[0],))
        
        # Track metrics for this target
        target_reward = 0.0
        target_distance = 0.0
        
        # Simulation loop
        for step in range(args.steps_per_target):
            # Get action from policy
            action, h_state = rnn.predict(params, observation, h_state)
            
            # Step the environment
            next_observation, reward, done, info = env.step(action)
            
            # Update metrics
            target_reward += reward
            target_distance = info["distance"]  # Current distance to target
            
            # Update observation
            observation = next_observation
            
            # Sleep to slow down rendering if needed
            if args.render and args.render_delay > 0:
                time.sleep(args.render_delay)
        
        # Update overall metrics
        total_reward += target_reward
        total_distance += target_distance
        
        print(f"Target {i+1}/{n_targets}: Reward = {target_reward:.4f}, Final Distance = {target_distance:.4f}")
    
    # Report average metrics
    avg_reward = total_reward / n_targets
    avg_distance = total_distance / n_targets
    
    print(f"Evaluation complete!")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average final distance: {avg_distance:.4f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_distance': avg_distance
    }

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate an RNN controller for the arm reaching task")
    
    # General parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                      help="Mode: train or evaluate")
    parser.add_argument("--model_path", type=str, default="arm.xml",
                      help="Path to the MuJoCo model XML file")
    parser.add_argument("--save_path", type=str, default="models",
                      help="Directory to save models and results")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    # Training parameters
    parser.add_argument("--n_generations", type=int, default=100,
                      help="Number of generations for training")
    parser.add_argument("--popsize", type=int, default=64,
                      help="Population size for CMA-ES")
    parser.add_argument("--hidden_size", type=int, default=32,
                      help="Hidden layer size for the RNN")
    parser.add_argument("--n_targets", type=int, default=10,
                      help="Number of targets per evaluation during training")
    parser.add_argument("--steps_per_target", type=int, default=100,
                      help="Number of steps per target")
    parser.add_argument("--log_interval", type=int, default=10,
                      help="Interval for logging training progress")
    
    # Evaluation parameters
    parser.add_argument("--load_model", type=str, default=None,
                      help="Path to a saved model for evaluation")
    parser.add_argument("--eval_targets", type=int, default=None,
                      help="Number of targets for evaluation")
    parser.add_argument("--render", action="store_true",
                      help="Enable rendering")
    parser.add_argument("--render_delay", type=float, default=0.01,
                      help="Delay between rendered frames")
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Run in the specified mode
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if args.load_model is None:
            raise ValueError("Must specify --load_model for evaluation mode")
        evaluate(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()