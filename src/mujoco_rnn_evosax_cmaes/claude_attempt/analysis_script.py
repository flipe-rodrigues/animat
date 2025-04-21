"""
Analysis tools for training results.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import argparse
import glob

def load_training_history(path: str) -> List[Dict[str, Any]]:
    """
    Load training history from file.
    
    Args:
        path: Path to the training history file
        
    Returns:
        List of dictionaries with training metrics per generation
    """
    with open(path, 'rb') as f:
        history = pickle.load(f)
    return history

def analyze_convergence(history: List[Dict[str, Any]], window_size: int = 10) -> Dict[str, Any]:
    """
    Analyze convergence of training.
    
    Args:
        history: Training history
        window_size: Window size for moving average
        
    Returns:
        Dictionary with convergence metrics
    """
    # Extract metrics
    generations = [entry['generation'] for entry in history]
    best_fitness = [entry['best_fitness'] for entry in history]
    best_overall = [entry['best_overall'] for entry in history]
    
    # Calculate improvement rate
    improvements = np.diff(best_overall)
    
    # Calculate moving average of improvement
    if len(improvements) >= window_size:
        moving_avg = np.convolve(
            improvements, 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
    else:
        moving_avg = np.mean(improvements) * np.ones(1)
    
    # Check for convergence
    # We consider converged if the improvement is very small
    convergence_threshold = 0.001
    if len(moving_avg) > 0 and abs(moving_avg[-1]) < convergence_threshold:
        converged = True
        convergence_gen = generations[-(len(generations) - len(moving_avg))]
    else:
        converged = False
        convergence_gen = None
    
    # Check for plateaus
    plateau_threshold = 0.01
    plateaus = []
    plateau_start = None
    
    for i in range(1, len(best_overall)):
        improvement = best_overall[i] - best_overall[i-1]
        
        if abs(improvement) < plateau_threshold:
            if plateau_start is None:
                plateau_start = generations[i-1]
        else:
            if plateau_start is not None:
                plateau_end = generations[i-1]
                plateaus.append((plateau_start, plateau_end))
                plateau_start = None
    
    # If still in a plateau at the end
    if plateau_start is not None:
        plateaus.append((plateau_start, generations[-1]))
    
    return {
        'converged': converged,
        'convergence_generation': convergence_gen,
        'final_fitness': best_overall[-1] if best_overall else None,
        'total_improvement': best_overall[-1] - best_overall[0] if best_overall else None,
        'improvement_rate': np.mean(improvements) if len(improvements) > 0 else None,
        'plateaus': plateaus
    }

def compare_hyperparameters(results_dir: str, param_name: str) -> Dict[str, Any]:
    """
    Compare results across different hyperparameter values.
    
    Args:
        results_dir: Directory containing subdirectories with results
        param_name: Name of the hyperparameter to compare
        
    Returns:
        Dictionary with comparison metrics
    """
    # Find all history files
    history_files = glob.glob(os.path.join(results_dir, f"*_{param_name}_*", "training_history.pkl"))
    
    if not history_files:
        raise ValueError(f"No history files found matching the pattern {param_name}")
    
    # Extract parameter values and load histories
    param_values = []
    histories = []
    final_fitness = []
    convergence_gens = []
    
    for file_path in history_files:
        # Extract parameter value from directory name
        dir_name = os.path.basename(os.path.dirname(file_path))
        param_value = float(dir_name.split(f"{param_name}_")[1].split("_")[0])
        param_values.append(param_value)
        
        # Load history
        history = load_training_history(file_path)
        histories.append(history)
        
        # Analyze convergence
        analysis = analyze_convergence(history)
        final_fitness.append(analysis['final_fitness'])
        if analysis['convergence_generation'] is not None:
            convergence_gens.append(analysis['convergence_generation'])
        else:
            convergence_gens.append(float('inf'))
    
    # Sort by parameter value
    sorted_indices = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_indices]
    final_fitness = [final_fitness[i] for i in sorted_indices]
    convergence_gens = [convergence_gens[i] for i in sorted_indices]
    histories = [histories[i] for i in sorted_indices]
    
    return {
        'param_name': param_name,
        'param_values': param_values,
        'final_fitness': final_fitness,
        'convergence_gens': convergence_gens,
        'histories': histories
    }

def plot_hyperparameter_comparison(comparison: Dict[str, Any], output_dir: str) -> None:
    """
    Plot comparison of hyperparameter settings.
    
    Args:
        comparison: Result of compare_hyperparameters
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    param_name = comparison['param_name']
    param_values = comparison['param_values']
    final_fitness = comparison['final_fitness']
    convergence_gens = comparison['convergence_gens']
    histories = comparison['histories']
    
    # Plot final fitness vs parameter value
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, final_fitness, 'o-', markersize=8)
    plt.xlabel(f"{param_name.capitalize()}")
    plt.ylabel("Final Fitness")
    plt.title(f"Final Fitness vs {param_name.capitalize()}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{param_name}_fitness.png"))
    plt.close()
    
    # Plot convergence generation vs parameter value
    plt.figure(figsize=(10, 6))
    # Filter out infinity values
    valid_indices = [i for i, gen in enumerate(convergence_gens) if gen != float('inf')]
    if valid_indices:
        valid_param_values = [param_values[i] for i in valid_indices]
        valid_convergence_gens = [convergence_gens[i] for i in valid_indices]
        plt.plot(valid_param_values, valid_convergence_gens, 'o-', markersize=8)
        plt.xlabel(f"{param_name.capitalize()}")
        plt.ylabel("Convergence Generation")
        plt.title(f"Convergence Speed vs {param_name.capitalize()}")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{param_name}_convergence.png"))
    plt.close()
    
    # Plot training curves for all parameter values
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(histories):
        generations = [entry['generation'] for entry in history]
        best_overall = [entry['best_overall'] for entry in history]
        plt.plot(generations, best_overall, label=f"{param_name}={param_values[i]}")
    
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Training Progress for Different {param_name.capitalize()} Values")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{param_name}_progress.png"))
    plt.close()

def analyze_model_performance(model_paths: List[str], env_params: Dict[str, Any], 
                             n_evaluations: int = 10, output_dir: str = "analysis") -> Dict[str, Any]:
    """
    Analyze and compare performance of multiple trained models.
    
    Args:
        model_paths: List of paths to trained models
        env_params: Parameters for the environment
        n_evaluations: Number of evaluations per model
        output_dir: Directory to save results
        
    Returns:
        Dictionary with performance metrics
    """
    # Imports done here to avoid circular dependencies
    import jax
    import jax.numpy as jnp
    from environment import ArmReachingEnv
    from rnn_model import SimpleRNN
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = ArmReachingEnv(**env_params)
    
    # Results storage
    results = {
        'model_names': [],
        'mean_distances': [],
        'std_distances': [],
        'success_rates': [],
        'trajectories': []
    }
    
    # Evaluate each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path).split('.')[0]
        print(f"Evaluating {model_name}...")
        
        # Load model parameters
        with open(model_path, 'rb') as f:
            params_np = pickle.load(f)
            params = {k: jnp.array(v) for k, v in params_np.items()}
        
        # Infer hidden size from parameters
        hidden_size = params['w_hh'].shape[0]
        
        # Create RNN
        rnn = SimpleRNN(
            input_size=env.input_dim,
            hidden_size=hidden_size,
            output_size=env.output_dim
        )
        
        # Evaluation statistics
        final_distances = []
        success_count = 0  # Count targets reached within threshold
        all_trajectories = []
        
        # Set random seed
        key = jax.random.PRNGKey(42)
        
        # Evaluate on multiple targets
        for i in range(n_evaluations):
            # Generate new key
            key, subkey = jax.random.split(key)
            
            # Reset environment
            observation = env.reset(subkey)
            
            # Initialize hidden state
            h_state = jnp.zeros((hidden_size,))
            
            # Track trajectory
            trajectory = [env.data.sensor("end_effector_pos").data.copy()]
            
            # Run simulation
            for _ in range(env.episode_length):
                # Get action from policy
                action, h_state = rnn.predict(params, observation, h_state)
                
                # Step environment
                observation, reward, done, info = env.step(action)
                
                # Store position
                trajectory.append(env.data.sensor("end_effector_pos").data.copy())
            
            # Get final distance
            final_distance = info["distance"]
            final_distances.append(final_distance)
            
            # Check if target reached
            success_threshold = 0.1  # Distance threshold for "success"
            if final_distance < success_threshold:
                success_count += 1
            
            # Store trajectory
            all_trajectories.append(np.array(trajectory))
        
        # Calculate statistics
        mean_distance = np.mean(final_distances)
        std_distance = np.std(final_distances)
        success_rate = success_count / n_evaluations
        
        # Store results
        results['model_names'].append(model_name)
        results['mean_distances'].append(mean_distance)
        results['std_distances'].append(std_distance)
        results['success_rates'].append(success_rate)
        results['trajectories'].append(all_trajectories)
        
        print(f"  Mean Distance: {mean_distance:.4f} ± {std_distance:.4f}")
        print(f"  Success Rate: {success_rate:.2f}")
    
    # Plot comparison of models
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results['model_names']))
    plt.bar(x, results['mean_distances'], yerr=results['std_distances'], 
            alpha=0.7, capsize=10)
    plt.xticks(x, results['model_names'], rotation=45, ha='right')
    plt.ylabel('Mean Final Distance')
    plt.title('Model Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    plt.bar(x, results['success_rates'], alpha=0.7)
    plt.xticks(x, results['model_names'], rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Model Success Rate Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rates.png"))
    plt.close()
    
    return results

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Analyze training results')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Convergence analysis
    conv_parser = subparsers.add_parser('convergence', help='Analyze convergence')
    conv_parser.add_argument('--history', type=str, required=True,
                           help='Path to training history file')
    conv_parser.add_argument('--window', type=int, default=10,
                           help='Window size for moving average')
    
    # Hyperparameter comparison
    hyper_parser = subparsers.add_parser('hyperparameter', help='Compare hyperparameters')
    hyper_parser.add_argument('--results_dir', type=str, required=True,
                            help='Directory with results from different hyperparameter values')
    hyper_parser.add_argument('--param_name', type=str, required=True,
                            help='Name of the hyperparameter to compare')
    hyper_parser.add_argument('--output_dir', type=str, default='plots',
                            help='Directory to save plots')
    
    # Model performance analysis
    perf_parser = subparsers.add_parser('performance', help='Analyze model performance')
    perf_parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                           help='Paths to trained models')
    perf_parser.add_argument('--model_path', type=str, default='arm_fixed.xml',
                           help='Path to MuJoCo model file')
    perf_parser.add_argument('--episode_length', type=int, default=100,
                           help='Episode length for evaluation')
    perf_parser.add_argument('--n_evaluations', type=int, default=10,
                           help='Number of evaluations per model')
    perf_parser.add_argument('--output_dir', type=str, default='analysis',
                           help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.command == 'convergence':
        history = load_training_history(args.history)
        analysis = analyze_convergence(history, args.window)
        
        print("Convergence Analysis:")
        print(f"  Converged: {analysis['converged']}")
        if analysis['convergence_generation'] is not None:
            print(f"  Convergence Generation: {analysis['convergence_generation']}")
        print(f"  Final Fitness: {analysis['final_fitness']:.4f}")
        print(f"  Total Improvement: {analysis['total_improvement']:.4f}")
        print(f"  Average Improvement Rate: {analysis['improvement_rate']:.6f}")
        
        if analysis['plateaus']:
            print("  Plateaus:")
            for start, end in analysis['plateaus']:
                print(f"    Generations {start} to {end}")
    
    elif args.command == 'hyperparameter':
        comparison = compare_hyperparameters(args.results_dir, args.param_name)
        plot_hyperparameter_comparison(comparison, args.output_dir)
        
        print(f"Hyperparameter Comparison for {args.param_name}:")
        for i, value in enumerate(comparison['param_values']):
            print(f"  {args.param_name}={value}:")
            print(f"    Final Fitness: {comparison['final_fitness'][i]:.4f}")
            if comparison['convergence_gens'][i] != float('inf'):
                print(f"    Convergence Generation: {comparison['convergence_gens'][i]}")
            else:
                print(f"    Did not converge")
    
    elif args.command == 'performance':
        env_params = {
            'model_path': args.model_path,
            'episode_length': args.episode_length,
            'render': False
        }
        
        results = analyze_model_performance(
            args.model_paths, env_params, 
            args.n_evaluations, args.output_dir
        )
        
        print("Model Performance Analysis:")
        for i, model_name in enumerate(results['model_names']):
            print(f"  {model_name}:")
            print(f"    Mean Distance: {results['mean_distances'][i]:.4f} ± {results['std_distances'][i]:.4f}")
            print(f"    Success Rate: {results['success_rates'][i]:.2f}")

if __name__ == "__main__":
    main()