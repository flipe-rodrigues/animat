# Musculoskeletal Arm Control with RNN Distillation

A complete implementation of a 2-joint, 4-muscle musculoskeletal arm controlled by a recurrent neural network (RNN) trained via distillation learning from an MLP policy in a reaching task.

## Overview

This project implements a biologically-inspired motor control system with the following key features:

- **Musculoskeletal Model**: 2-joint arm (shoulder and elbow) actuated by 4 antagonistic muscles in MuJoCo
- **Proprioceptive Observations**: Muscle length, velocity, and force feedback
- **Place Cell Encoding**: Target positions encoded using grid-based Gaussian tuning curves
- **Reaching Task**: Reach random targets and hold position for variable durations
- **Two-Stage Training**:
  1. Train MLP policy using Proximal Policy Optimization (PPO)
  2. Distill knowledge to RNN via behavioral cloning (imitation learning)

## Project Structure

```
musculoskeletal_arm_control/
├── models/
│   └── arm_2joint_4muscle.xml          # MuJoCo model definition
├── envs/
│   └── reaching_env.py                 # Gymnasium environment
├── agents/
│   ├── mlp_policy.py                   # MLP policy network
│   └── rnn_policy.py                   # RNN policy network (simple ReLU units)
├── training/
│   ├── ppo_trainer.py                  # PPO trainer for MLP
│   └── behavioral_cloning.py           # Distillation trainer for RNN
├── utils/
│   └── place_cells.py                  # Place cell grid implementation
├── train.py                            # Main training script
├── visualize.py                        # Visualization script
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Installation

### Prerequisites

- Python 3.8+
- MuJoCo 3.0+

### Setup

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify MuJoCo installation:
```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## Usage

### Training

Run the complete training pipeline (MLP training followed by RNN distillation):

```bash
python train.py
```

This will:
1. Train a teacher MLP policy using PPO for 500 updates (~1-2 hours on CPU)
2. Collect 200 demonstration episodes from the trained teacher
3. Train a student RNN policy via behavioral cloning for 50 epochs
4. Compare the performance of both policies
5. Save checkpoints to `checkpoints/`

### Visualization

After training, visualize the trained policies:

```bash
# Visualize both teacher and student
python visualize.py --policy both --episodes 3

# Visualize only teacher
python visualize.py --policy teacher --episodes 5

# Visualize only student
python visualize.py --policy student --episodes 5
```

### Custom Configuration

You can modify training parameters by editing `train.py`:

```python
# MLP training
teacher_policy = train_teacher_mlp(
    env=env,
    num_updates=500,  # Increase for better performance
    save_path="checkpoints/teacher_mlp.pt"
)

# RNN distillation
student_policy = train_student_rnn(
    env=env,
    teacher_policy=teacher_policy,
    num_demo_episodes=200,  # More demos = better distillation
    num_epochs=50,
    save_path="checkpoints/student_rnn.pt"
)
```

## Technical Details

### Musculoskeletal Model

The arm consists of:
- **Shoulder joint**: Hinge joint with range [-90°, 90°]
- **Elbow joint**: Hinge joint with range [0°, 143°]
- **4 Muscles**:
  - Shoulder flexor/extensor (antagonistic pair)
  - Elbow flexor/extensor (antagonistic pair)

Each muscle has:
- Length sensor
- Velocity sensor
- Force sensor

### Observation Space

The observation vector contains:
- **Muscle proprioception** (12 dims): length, velocity, force for each of 4 muscles
- **Target encoding** (64 dims): 8×8 grid of place cells with Gaussian tuning
- **Time remaining** (1 dim): Normalized time remaining to hold

Total: **77 dimensions**

### Action Space

4-dimensional continuous actions representing muscle activations in range [0, 1]

### Task Requirements

1. **Reach**: Move hand to within 3cm of target
2. **Hold**: Maintain position within 4cm for randomly sampled duration (0.2-0.8s)
3. **Success**: Complete hold duration to get maximum reward

### Reward Function

```python
reward = -2.0 * distance_to_target  # Distance penalty
         + 5.0   # Bonus for being at target (< 3cm)
         + 10.0 * dt  # Bonus per timestep while holding
         + 100.0  # Large bonus for successful hold completion
```

### Place Cell Encoding

Target positions are encoded using a grid of place cells with 2D Gaussian tuning:

```python
activity_i = exp(-||position - center_i||^2 / (2 * sigma^2))
```

This provides a distributed representation that helps with generalization across the workspace.

### Neural Networks

**Teacher MLP**:
- Architecture: 77 → 256 → 256 → 4
- Activation: ReLU
- Output: Sigmoid (for [0,1] muscle activations)

**Student RNN**:
- Architecture: Simple RNN with ReLU units
- Input: 77 → Hidden: 128 → Output: 4
- Single recurrent layer with hidden-to-hidden connections
- Output: Sigmoid (for [0,1] muscle activations)

### Training Algorithms

**PPO (Proximal Policy Optimization)**:
- Used to train teacher MLP from scratch
- Hyperparameters:
  - Learning rate: 3×10⁻⁴
  - Discount factor (γ): 0.99
  - GAE λ: 0.95
  - Clip ε: 0.2
  - Epochs per update: 10

**Behavioral Cloning**:
- Used to distill teacher knowledge to student RNN
- Supervised learning with MSE loss between student and teacher actions
- Trained on sequences of length 50 timesteps
- Learning rate: 1×10⁻³

## Expected Results

After training, you should see:

**Teacher MLP Performance**:
- Mean reward: 100-200 (varies with training duration)
- Success rate: 60-80%

**Student RNN Performance**:
- Mean reward: 80-90% of teacher performance
- Success rate: 50-70%

The student RNN typically achieves 80-95% of teacher performance while using recurrent processing, demonstrating successful knowledge distillation.

## Customization

### Modify Workspace Bounds

Edit `reaching_env.py`:
```python
workspace_bounds = (
    (0.1, 0.6),  # x range
    (0.3, 0.7)   # z range
)
```

### Adjust Place Cell Resolution

In `train.py` or `reaching_env.py`:
```python
place_cell_grid_size = (10, 10)  # Higher resolution
place_cell_sigma = 0.05  # Sharper tuning curves
```

### Change RNN Architecture

In `train.py`:
```python
student_policy = RNNPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_size=256,  # Larger hidden state
    num_layers=2      # Deeper network
)
```

## Troubleshooting

**MuJoCo not found**:
```bash
pip install mujoco
```

**Visualization not working**:
- Make sure you have a display/window manager
- For headless servers, use `render_mode=None`

**Poor performance**:
- Increase PPO training updates (num_updates > 1000)
- Collect more demonstrations (num_demo_episodes > 500)
- Train RNN longer (num_epochs > 100)

**CUDA errors**:
- Reduce batch size if running out of memory
- Use CPU by removing CUDA device selection

## References

- **MuJoCo**: Todorov et al. (2012) "MuJoCo: A physics engine for model-based control"
- **PPO**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- **Place Cells**: O'Keefe & Dostrovsky (1971) "The hippocampus as a spatial map"
- **Distillation**: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"

## License

This project is provided for educational and research purposes.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{musculoskeletal_arm_rnn,
  title = {Musculoskeletal Arm Control with RNN Distillation},
  year = {2026},
  url = {https://github.com/yourname/musculoskeletal_arm_control}
}
```
