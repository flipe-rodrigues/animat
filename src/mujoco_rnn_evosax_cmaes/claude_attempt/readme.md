# 2-Joint 4-Muscle Arm Reaching Task with RNN Control

This project implements a 2-joint, 4-muscle arm in MuJoCo that learns to reach targets using a simple tanh RNN controller. The training is done using the CMA-ES evolutionary algorithm, with JAX and EvoSAX providing parallelization.

## Overview

The system consists of:
- A 2-joint (shoulder and elbow) arm with 4 muscles (flexors and extensors for each joint)
- 12 sensors (length, velocity, and force for each muscle)
- A tanh RNN controller that processes sensor data and target coordinates
- CMA-ES evolutionary algorithm for training the RNN

## Project Structure

- `arm.xml`: MuJoCo model file defining the 2-joint arm
- `fix_model.py`: Script to add missing site definitions to the model
- `environment.py`: MuJoCo environment for the arm reaching task
- `rnn_model.py`: Implementation of the simple tanh RNN controller
- `evolutionary_trainer.py`: CMA-ES trainer for the RNN controller
- `main.py`: Main script for training and evaluation
- `example.py`: Example script showing basic usage
- `visualization.py`: Utilities for visualizing results
- `target_generator.py`: Generator for different target sequences

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Fix the MuJoCo model (adds missing site definitions):
```bash
python fix_model.py
```

## Usage

### Quick Example

Run a quick demo:
```bash
python example.py
```

### Training

Train a new model:
```bash
python main.py --mode train --n_generations 100 --popsize 64
```

Training parameters:
- `--n_generations`: Number of generations for training (default: 100)
- `--popsize`: Population size for CMA-ES (default: 64)
- `--hidden_size`: Hidden layer size for the RNN (default: 32)
- `--n_targets`: Number of targets per evaluation (default: 10)
- `--steps_per_target`: Steps per target (default: 100)

### Evaluation

Evaluate a trained model:
```bash
python main.py --mode eval --load_model models/final_best_model.pkl --render
```

Evaluation parameters:
- `--load_model`: Path to saved model for evaluation
- `--eval_targets`: Number of targets for evaluation
- `--render`: Enable rendering
- `--render_delay`: Delay between rendered frames (default: 0.01)

## Generating Target Sequences

Generate different types of target sequences:
```python
from target_generator import TargetGenerator

# Initialize generator
generator = TargetGenerator(seed=42)

# Generate random targets
random_targets = generator.generate_random_targets(n_targets=10)

# Generate a grid of targets
grid_targets = generator.generate_grid_targets(nx=3, ny=3, nz=3)

# Generate circular targets
circular_targets = generator.generate_circular_targets(
    n_targets=12, radius=0.5, plane='xz'
)

# Generate a figure-eight pattern
figure_eight_fn = lambda t: generator.figure_eight(t, scale=0.5, plane='xz')
figure_eight_targets = generator.generate_sequential_pattern(
    figure_eight_fn, n_cycles=2, points_per_cycle=20
)

# Save targets
generator.save_targets(random_targets, "targets/random_targets.npy")
```

## Visualizing Results

Visualize training history:
```python
from visualization import plot_training_history

plot_training_history("models/training_history.pkl", "plots")
```

## Advanced Training

For more advanced training scenarios, you can modify:
- The network architecture in `rnn_model.py`
- The reward function in `environment.py`
- The evolutionary parameters in `evolutionary_trainer.py`

## Notes on Implementation

- The project uses MuJoCo MJX for JAX-accelerated physics simulation
- EvoSAX provides efficient parallelized implementation of CMA-ES
- JAX enables hardware acceleration (GPU/TPU) of the entire pipeline
- The RNN is intentionally simple (tanh activation, no GRU/LSTM) as specified

## Performance Considerations

For best performance:
- Use hardware acceleration (GPU/TPU) where available
- Increase the population size for better exploration
- Adjust the steps_per_target parameter based on task difficulty
- For faster training, start with a smaller hidden size and increase as needed

## Extending the Project

This framework can be extended in several ways:
- Adding more complex muscle models
- Implementing different network architectures
- Adding obstacle avoidance constraints
- Training against dynamic targets
- Implementing curriculum learning by gradually increasing target difficulty

## Troubleshooting

Common issues:
- If the arm behaves erratically, check the muscle site placements
- If training is unstable, try decreasing the learning rate
- If the model isn't reaching targets, increase training time and hidden size
- If simulations are slow, ensure you're using MJX properly with hardware acceleration

## References

- MuJoCo: https://mujoco.org/
- MuJoCo MJX: https://mujoco.readthedocs.io/en/latest/mjx.html
- JAX: https://github.com/google/jax
- EvoSAX: https://github.com/RobertTLange/evosax