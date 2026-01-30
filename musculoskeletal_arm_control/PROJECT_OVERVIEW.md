# Musculoskeletal Arm Control - Project Overview

## What This Project Does

This is a complete implementation of a **biologically-inspired motor control system** where:

1. A 2-joint arm with 4 muscles learns to reach targets in a simulated environment
2. An **MLP teacher policy** is first trained using reinforcement learning (PPO)
3. An **RNN student policy** then learns to mimic the teacher via distillation
4. The RNN uses simple ReLU units and processes proprioceptive muscle feedback
5. Targets are encoded using place cells (inspired by neuroscience)

## Key Features

### 🦾 Musculoskeletal Model
- 2 joints (shoulder, elbow) with realistic ranges of motion
- 4 antagonistic muscles (flexor/extensor pairs)
- Proprioceptive sensors: muscle length, velocity, force

### 🎯 Reaching Task
- Reach random targets within the arm's workspace
- Hold position for random durations (0.2-0.8 seconds)
- Rewards for distance, reaching, and successful holding

### 🧠 Neural Encoding
- **Place cells**: Grid of 64 units (8×8) with Gaussian tuning curves
- Provides distributed spatial representation of target positions
- Helps with generalization across the workspace

### 🤖 Two-Stage Learning
1. **Stage 1**: PPO trains MLP policy from scratch (~500 updates)
2. **Stage 2**: RNN learns to imitate MLP via behavioral cloning (~200 demos, 50 epochs)

## File Organization

```
musculoskeletal_arm_control/
│
├── 📄 README.md                    ← Comprehensive documentation
├── 📄 GETTING_STARTED.md           ← Step-by-step guide
├── 📄 requirements.txt             ← Python dependencies
│
├── models/
│   └── arm_2joint_4muscle.xml      ← MuJoCo model definition
│
├── envs/
│   └── reaching_env.py             ← Gymnasium environment (task logic)
│
├── agents/
│   ├── mlp_policy.py               ← Teacher MLP network
│   └── rnn_policy.py               ← Student RNN network (simple ReLU)
│
├── training/
│   ├── ppo_trainer.py              ← PPO algorithm for RL
│   └── behavioral_cloning.py       ← Imitation learning for distillation
│
├── utils/
│   └── place_cells.py              ← Place cell encoding implementation
│
├── 🚀 train.py                     ← Main training script (run this!)
├── 👁️ visualize.py                 ← Visualization of trained policies
└── ✅ test_setup.py                ← Verify installation
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Test Setup
```bash
python test_setup.py
```

### Train Models
```bash
python train.py
```

### Visualize Results
```bash
python visualize.py --policy both --episodes 3
```

## Technical Specifications

| Component | Details |
|-----------|---------|
| **Physics Engine** | MuJoCo 3.0+ |
| **RL Algorithm** | PPO (Proximal Policy Optimization) |
| **Distillation** | Behavioral Cloning (supervised imitation) |
| **MLP Architecture** | 77 → 256 → 256 → 4 (ReLU, Sigmoid) |
| **RNN Architecture** | 77 → 128(RNN) → 4 (ReLU, Sigmoid) |
| **Observation Dim** | 77 (12 muscle + 64 place cells + 1 time) |
| **Action Dim** | 4 (muscle activations [0,1]) |
| **Training Time** | 1-3 hours (CPU), faster with GPU |

## What You'll Learn

- **Motor Control**: How muscle-based systems reach targets
- **Reinforcement Learning**: Training policies with PPO
- **Knowledge Distillation**: Transferring skills from MLP to RNN
- **Neural Encoding**: Place cells for spatial representation
- **MuJoCo Simulation**: Physics-based robotics simulation
- **Gymnasium**: Creating custom RL environments

## Expected Results

After training:
- **Teacher MLP**: 60-80% success rate, reward 100-200
- **Student RNN**: 50-70% success rate, 80-95% of teacher performance
- **Distillation Quality**: RNN successfully imitates teacher behavior

## Customization Ideas

1. **Harder Task**: Increase hold times, reduce thresholds
2. **Better Models**: Deeper networks, more training
3. **New Features**: Obstacles, multiple targets, energy efficiency
4. **Architecture**: Try LSTM, GRU, or other RNN variants
5. **Encoding**: Different place cell configurations

## Dependencies

- `numpy` - Numerical computing
- `torch` - Neural networks
- `gymnasium` - RL environment interface
- `mujoco` - Physics simulation
- `matplotlib` - Plotting (optional)

## Citations & References

This implementation combines ideas from:
- **MuJoCo**: Physics simulation for robotics
- **PPO**: State-of-the-art RL algorithm
- **Place Cells**: Neuroscience-inspired spatial encoding
- **Distillation**: Efficient knowledge transfer

## Support

- See `README.md` for detailed documentation
- See `GETTING_STARTED.md` for step-by-step instructions
- Run `test_setup.py` to diagnose issues

---

**Ready to train your first musculoskeletal controller?** 🚀

Run: `python train.py`
