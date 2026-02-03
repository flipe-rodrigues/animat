# Muscle-RNN: Muscle-Driven Arm Control with Recurrent Neural Networks

A biologically-inspired neural controller for muscle-driven robotic arms in MuJoCo.

## Installation

```bash
pip install mujoco gymnasium torch numpy matplotlib cma
```

## Project Structure

```
muscle-rnn/
├── plants/                    # Physics simulation wrappers
│   └── mujoco.py             # MuJoCo physics wrapper + calibration
├── models/                    # Neural network models
│   ├── controllers.py        # Main controllers (RNNController, MLPController)
│   └── modules/              # Modular components
│       ├── sensory.py        # Proprioceptive neurons (Ia, II, Ib)
│       ├── motor.py          # Motor neurons (alpha, gamma)
│       ├── target.py         # Target encoding
│       ├── rnn.py            # RNN core
│       └── mlp.py            # MLP core
├── envs/                      # Environment definitions
│   └── reaching.py           # Reaching task Gymnasium environment
├── training/                  # Training algorithms
│   ├── train_cmaes.py        # CMA-ES training (uses `cmaes` PyPI package)
│   └── train_distillation.py # Distillation learning
├── examples/                  # Interactive example scripts
│   ├── train_cmaes_example.py        # CMA-ES training example (#%% cells)
│   ├── train_distillation_example.py # Distillation example (#%% cells)
│   └── visualize_example.py          # Visualization example (#%% cells)
├── utils/                     # Utility modules
│   ├── visualization.py      # Visualization tools
│   └── episode_recorder.py   # Episode recording and network activity visualization
├── mujoco/                    # MuJoCo model files
│   └── arm.xml               # Arm model definition
└── outputs/                   # Training outputs
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
- **MotorModule**: Motor neuron outputs (spinal pathway)
  - Alpha motor neurons: Direct muscle activation
  - Gamma static: Modulates length sensitivity
  - Gamma dynamic: Modulates velocity sensitivity
- **TargetEncoder**: Spatial population code for target position
- **Monosynaptic Reflex Arcs**:
  - Type Ia → Alpha MN: Velocity-dependent stretch reflex
  - Type II → Alpha MN: Length-dependent stretch reflex

### Training Methods

1. **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
   - Uses the `cma` PyPI package
   - Black-box optimization of RNN weights
   - No gradients required
   - Good for complex reward landscapes
   - Parallelizable across multiple CPUs

2. **Distillation Learning**
   - Train MLP teacher using behavioral cloning
   - Student RNN learns to imitate teacher behavior
   - Leverages gradient-based optimization

## Quick Start

### Using Example Scripts

The `examples/` folder contains interactive scripts using `#%%` cell delimiters for use in VS Code, PyCharm, or Jupyter:

```python
# Run interactively in VS Code/PyCharm
python examples/train_cmaes_example.py
python examples/train_distillation_example.py
python examples/visualize_example.py
```

### Python API

```python
from models import RNNController, ControllerConfig
from training import run_cmaes_training

# Create model configuration
config = ControllerConfig(
    num_muscles=4,
    num_core_units=32,
    target_grid_size=4,
)

# Create controller
controller = RNNController(config)
print(f"Parameters: {controller.count_parameters()}")

# Train using CMA-ES
results = run_cmaes_training(
    xml_path='mujoco/arm.xml',
    output_dir='outputs/my_experiment',
    num_generations=500,
    population_size=32,
)
```

### Loading a Trained Model

```python
import torch
from models import RNNController, ControllerConfig

# Load checkpoint
checkpoint = torch.load('outputs/best_controller_final.pt')
config = ControllerConfig(**checkpoint['model_config'])
controller = RNNController(config)
controller.load_state_dict(checkpoint['model_state_dict'])
controller.eval()

# Use for inference (SB3-compatible predict interface)
controller._reset_state()
action, _ = controller.predict(observation, deterministic=True)

# Or use forward() directly with tensors
obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
action, info = controller.forward(obs_tensor)
```

## Configuration

### ControllerConfig

Defines neural network architecture:

```python
from models import ControllerConfig, RNNController, MLPController

# RNN Controller - uses int for num_core_units
rnn_config = ControllerConfig(
    num_muscles=4,               # Number of muscle actuators
    num_core_units=32,           # Int: RNN hidden size
    target_grid_size=4,          # Spatial encoding grid size
    target_sigma=0.5,            # Gaussian encoding width
)
rnn_controller = RNNController(rnn_config)

# MLP Controller - uses list for num_core_units  
mlp_config = ControllerConfig(
    num_muscles=4,
    num_core_units=[128, 128],   # List: MLP hidden layer sizes
    target_grid_size=4,
    target_sigma=0.5,
)
mlp_controller = MLPController(mlp_config)
```

## Key Design Principles

1. **Modularity**: Each component (sensory, motor, target) is a separate, reusable module
2. **Biological Inspiration**: Architecture mirrors biological motor control systems
3. **Extensibility**: Base classes enable easy addition of new controllers
4. **Configuration-Driven**: Dataclasses for type-safe configuration
5. **Interactive Development**: Example scripts with cell delimiters for exploratory work

## Dependencies

- Python 3.8+
- PyTorch
- MuJoCo
- Gymnasium
- NumPy
- Matplotlib
- cma (for CMA-ES optimization)

## License

MIT
