# Muscle-RNN: Muscle-Driven Arm Control with Recurrent Neural Networks

A biologically-inspired neural controller for muscle-driven robotic arms in MuJoCo.

## Project Structure

```
muscle-rnn/
├── core/                      # Core abstractions and configuration
│   ├── __init__.py
│   ├── base.py               # Base classes (BaseController, BaseTrainer)
│   ├── config.py             # Configuration classes (ModelConfig, TrainingConfig)
│   └── constants.py          # Project-wide constants
├── models/                    # Neural network models
│   ├── __init__.py
│   ├── controllers.py        # Main controllers (RNNController, MLPController)
│   └── modules/              # Modular components
│       ├── sensory.py        # Proprioceptive neurons (Ia, II, Ib)
│       ├── motor.py          # Motor neurons (alpha, gamma)
│       ├── target.py         # Target encoding
│       ├── rnn.py            # RNN core
│       └── mlp.py            # MLP core
├── envs/                      # Environment definitions
│   ├── __init__.py
│   └── reaching.py           # Reaching task environment
├── training/                  # Training algorithms
│   ├── __init__.py
│   ├── train_cmaes.py        # CMA-ES training
│   ├── train_distillation.py # Distillation learning
│   ├── evaluation.py         # Shared evaluation utilities
│   └── checkpoint.py         # Checkpoint management
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── plant.py              # MuJoCo physics interface & XML parsing
│   ├── visualization.py      # Visualization tools
│   └── network_visualizer.py # Network architecture visualization
├── mujoco/                    # MuJoCo model files
│   └── arm.xml               # Arm model definition
├── outputs/                   # Training outputs
├── run.py                     # Main command-line interface
└── README.md                  # This file
```

## Architecture

### Controllers

The project implements two types of controllers:

1. **RNNController**: Recurrent neural network with temporal integration
   - Uses RNN/GRU/LSTM core for sequence processing
   - Maintains hidden state across timesteps
   - Best for tasks requiring temporal memory

2. **MLPController**: Feedforward multilayer perceptron
   - No recurrent connections
   - Larger hidden layers to compensate for lack of memory
   - Used as teacher network in distillation learning

Both controllers share:
- **SensoryModule**: Biologically-inspired proprioceptive neurons
  - Type Ia: Velocity-sensitive (muscle spindle primary)
  - Type II: Length-sensitive (muscle spindle secondary)
  - Type Ib: Force-sensitive (Golgi tendon organ)
- **MotorModule**: Motor neuron outputs (cortical pathway)
  - Alpha motor neurons: Direct muscle activation
  - Gamma static: Modulates length sensitivity
  - Gamma dynamic: Modulates velocity sensitivity
- **TargetEncoder**: Spatial population code for target position
- **Monosynaptic Reflex Arcs** (at controller level):
  - Type Ia → Alpha MN: Velocity-dependent stretch reflex
  - Type II → Alpha MN: Length-dependent stretch reflex

### Training Methods

1. **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
   - Black-box optimization of RNN weights
   - No gradients required
   - Good for complex reward landscapes
   - Parallelizable across multiple CPUs

2. **Distillation Learning**
   - Train MLP teacher using behavioral cloning
   - Student RNN learns to imitate teacher behavior
   - Leverages gradient-based optimization
   - Produces more compact, efficient controllers

## Recent Refactoring (2026-02-02)

The project underwent significant refactoring to improve:

### Improved Separation of Concerns
- Created `core/` module for shared abstractions
- Separated configuration management into `core/config.py`
- Extracted base classes into `core/base.py`
- Moved constants to `core/constants.py`

### Better Code Organization
- Created `training/evaluation.py` for shared evaluation logic
- Created `training/checkpoint.py` for unified checkpoint management
- Removed code duplication across training modules
- Cleaned up legacy code and comments
- **Merged `model_parser.py` into `plant.py` and moved to `utils/`** for better organization

### Bug Fixes
- Fixed MLPController calling non-existent `self.target_encoding()` method
- Standardized configuration attribute naming (num_* prefix)
- Improved error messages and validation

### Enhanced Maintainability
- Added base classes for extensibility
- Improved type hints throughout
- Consolidated configuration classes
- Better documentation and module organization

## Configuration

### ModelConfig

Defines neural network architecture:

```python
from core.config import ModelConfig

config = ModelConfig(
    num_muscles=6,               # Number of muscle actuators
    num_sensors=18,              # Total sensor inputs (3 per muscle)
    num_target_units=16,         # Target grid size^2 (4x4)
    rnn_hidden_size=32,          # RNN hidden layer size
    target_grid_size=4,          # Spatial encoding grid size
    target_sigma=0.1,            # Gaussian encoding width
)
```

### TrainingConfig

Base configuration for training:

```python
from core.config import TrainingConfig

training_config = TrainingConfig(
    xml_path='mujoco/arm.xml',
    output_dir='outputs/experiment_1',
    max_episode_steps=300,
    num_eval_episodes=10,
    save_checkpoint_every=25,
)
```

## Usage

### Command Line Interface

```bash
# Show model information
python run.py info mujoco/arm.xml

# Calibrate sensors
python run.py calibrate mujoco/arm.xml --output sensor_stats.pkl

# Train with CMA-ES
python run.py train mujoco/arm.xml --method cmaes --generations 500

# Train with distillation
python run.py train mujoco/arm.xml --method distillation

# Evaluate trained model
python run.py evaluate checkpoint.pth --episodes 100

# Visualize network architecture
python utils/network_visualizer.py
```

### Python API

```python
from core.config import ModelConfig
from models import RNNController, create_controller
from training import run_cmaes_training

# Create model configuration
config = ModelConfig(
    num_muscles=6,
    num_sensors=18,
    num_target_units=16,
)

# Create controller
controller = create_controller(config, controller_type='rnn')

# Train using CMA-ES
results = run_cmaes_training(
    xml_path='mujoco/arm.xml',
    model_config=config,
    num_generations=500,
    population_size=64,
    output_dir='outputs/cmaes_experiment',
)
```

## Key Design Principles

1. **Modularity**: Each component (sensory, motor, target) is a separate, reusable module
2. **Biological Inspiration**: Architecture mirrors biological motor control systems
3. **Extensibility**: Base classes and interfaces enable easy addition of new controllers
4. **Configuration-Driven**: Extensive use of dataclasses for type-safe configuration
5. **Separation of Concerns**: Clear boundaries between tasks (envs), physics (utils.plant), control (models), and learning (training)

## Dependencies

- Python 3.8+
- PyTorch
- MuJoCo
- Gymnasium
- NumPy
- Matplotlib (for visualization)

## Future Improvements

- Add more training algorithms (PPO, SAC, etc.)
- Implement curriculum learning
- Add multi-task support
- Enhance visualization tools
- Add comprehensive unit tests
- Create Docker container for reproducibility

## License

[Add your license here]

## Citation

[Add citation information if applicable]
