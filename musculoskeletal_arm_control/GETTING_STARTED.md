# Getting Started Guide

## Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd musculoskeletal_arm_control

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Run the test script to verify everything is working
python test_setup.py
```

You should see:
```
Testing place cells...
  ✓ Place cells working correctly

Testing environment...
  ✓ Environment working correctly

Testing policies...
  ✓ Policies working correctly

All tests passed! ✓
```

### 3. Train the Models

```bash
# Run the full training pipeline
# This will take 1-3 hours depending on your hardware
python train.py
```

The training proceeds in three phases:

**Phase 1: Training Teacher MLP with PPO**
- Trains an MLP policy using reinforcement learning
- ~500 updates of PPO
- Progress will be printed every update

**Phase 2: Training Student RNN via Behavioral Cloning**
- Collects demonstrations from the trained teacher
- Trains RNN to imitate the teacher
- Progress will be printed every epoch

**Phase 3: Comparing Policies**
- Evaluates both teacher and student
- Prints performance comparison

### 4. Visualize Results

After training completes, visualize the trained policies:

```bash
# Visualize both policies
python visualize.py --policy both --episodes 3

# Or visualize individually
python visualize.py --policy teacher --episodes 5
python visualize.py --policy student --episodes 5
```

## Understanding the Output

### Training Output

During MLP training, you'll see:
```
Update 1/500 | Mean Reward: 10.23 | Mean Length: 145.2
Update 2/500 | Mean Reward: 15.67 | Mean Length: 158.4
...
```

During RNN training, you'll see:
```
Epoch 1/50 | Train Loss: 0.012345 | Val Loss: 0.015432
Epoch 2/50 | Train Loss: 0.010234 | Val Loss: 0.013421
...
```

### Visualization

The visualization will show:
- The arm (brown capsules)
- Target position (green translucent sphere)
- Hand position (red sphere at end-effector)

Watch the arm reach for targets and maintain hold positions!

## Project Architecture

### Key Components

1. **MuJoCo Model** (`models/arm_2joint_4muscle.xml`)
   - Physical simulation of the arm
   - 2 joints, 4 muscles, sensors

2. **Environment** (`envs/reaching_env.py`)
   - Gymnasium-compatible interface
   - Handles observations, rewards, termination

3. **Place Cells** (`utils/place_cells.py`)
   - Encodes target positions spatially
   - Grid of Gaussian-tuned units

4. **Policies** (`agents/`)
   - `mlp_policy.py`: Teacher network
   - `rnn_policy.py`: Student network with simple ReLU RNN

5. **Trainers** (`training/`)
   - `ppo_trainer.py`: RL training for teacher
   - `behavioral_cloning.py`: Imitation learning for student

## Common Issues

### MuJoCo Installation

If you get import errors for MuJoCo:
```bash
pip install mujoco --upgrade
```

### Visualization Not Working

If you're on a headless server or VM without display:
- The training will work fine
- Visualization requires a display
- You can disable rendering: `render_mode=None`

### Training Too Slow

To speed up training:
- Use GPU if available (automatic with PyTorch)
- Reduce `num_updates` in `train.py`
- Reduce `num_demo_episodes` for faster distillation

### Memory Issues

If you run out of memory:
- Reduce `batch_size` in trainers
- Reduce `sequence_length` in behavioral cloning
- Close other applications

## Next Steps

### Experiment with the Code

1. **Modify the task**:
   - Change `hold_time_range` to make task harder/easier
   - Adjust `reach_threshold` and `hold_threshold`
   - Modify the reward function

2. **Change the architecture**:
   - Try deeper MLP: `hidden_dims=(512, 512, 256)`
   - Try multi-layer RNN: `num_layers=2`
   - Experiment with different RNN sizes

3. **Adjust training**:
   - More PPO updates for better teacher
   - More demonstrations for better student
   - Different hyperparameters (learning rate, etc.)

4. **Add features**:
   - Velocity constraints
   - Energy efficiency rewards
   - Multiple simultaneous targets
   - Obstacles in the workspace

### Analyze Performance

```python
# In train.py, add after training:
import matplotlib.pyplot as plt

# Plot learning curves
plt.plot(trainer.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Progress')
plt.savefig('learning_curve.png')
```

## Understanding the Science

### Why RNN Distillation?

- **MLPs** are powerful but stateless
- **RNNs** maintain internal state (memory)
- **Distillation** transfers knowledge efficiently
- Result: RNN learns temporal patterns from MLP's expertise

### Biological Inspiration

- **Muscles**: Like real antagonistic muscle pairs
- **Proprioception**: Sensory feedback from muscles
- **Place Cells**: Inspired by hippocampal spatial encoding
- **Motor Control**: Reaching task mirrors neuroscience experiments

### Key Insights

1. Place cell encoding provides spatial generalization
2. Muscle proprioception enables closed-loop control
3. Hold requirement demands temporal credit assignment
4. RNN learns implicit timing through recurrence

## Further Reading

- PPO paper: https://arxiv.org/abs/1707.06347
- MuJoCo documentation: https://mujoco.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
- Place cells review: O'Keefe & Nadel (1978) "The Hippocampus as a Cognitive Map"

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Run `test_setup.py` to diagnose problems
4. Check that file paths are correct
5. Ensure you have enough disk space for checkpoints

Happy experimenting! 🚀
