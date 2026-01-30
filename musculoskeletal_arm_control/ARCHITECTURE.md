# System Architecture

## Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PHASE 1: Reinforcement Learning (PPO)                              │
│  ┌────────────┐                                                      │
│  │ Environment │ ──observations──> ┌─────────────┐                  │
│  │  (Reaching) │                    │  MLP Policy │                  │
│  │             │ <───actions─────── │  (Teacher)  │                  │
│  │             │                    └─────────────┘                  │
│  │             │ ───rewards───>           │                          │
│  └────────────┘                           │                          │
│                                            ▼                          │
│                                    ┌──────────────┐                  │
│                                    │ PPO Training │                  │
│                                    └──────────────┘                  │
│                                            │                          │
│                                            ▼                          │
│                                  ✓ Trained Teacher MLP               │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PHASE 2: Distillation (Behavioral Cloning)                         │
│  ┌────────────┐                                                      │
│  │ Environment │ ──obs──> ┌─────────────┐                           │
│  │  (Collect   │          │  MLP Policy │                           │
│  │   Demos)    │ <─action─│  (Teacher)  │                           │
│  └────────────┘          └─────────────┘                           │
│        │                                                              │
│        │ demonstrations                                              │
│        ▼                                                              │
│  ┌──────────────────────┐                                           │
│  │  Dataset: obs, action │                                           │
│  │   pairs from teacher  │                                           │
│  └──────────────────────┘                                           │
│        │                                                              │
│        │ train via imitation                                         │
│        ▼                                                              │
│  ┌─────────────┐    MSE Loss     ┌──────────────────┐              │
│  │ RNN Policy  │ ◄───────────────│ Teacher Actions  │              │
│  │  (Student)  │                  │   (targets)      │              │
│  └─────────────┘                  └──────────────────┘              │
│        │                                                              │
│        ▼                                                              │
│  ✓ Trained Student RNN                                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow During Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│                       EXECUTION (INFERENCE)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Environment State                                                   │
│  ┌──────────────────────────────────────┐                           │
│  │ • Joint angles (shoulder, elbow)     │                           │
│  │ • Muscle states (4 muscles)          │                           │
│  │ • Target position (x, z)             │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ sense                                            │
│                    ▼                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │         PROPRIOCEPTION                │                           │
│  │  ┌────────────────────────────────┐  │                           │
│  │  │ Muscle 1: length, vel, force   │  │                           │
│  │  │ Muscle 2: length, vel, force   │  │  (12 dims)                │
│  │  │ Muscle 3: length, vel, force   │  │                           │
│  │  │ Muscle 4: length, vel, force   │  │                           │
│  │  └────────────────────────────────┘  │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │      PLACE CELL ENCODING              │                           │
│  │  ┌────────────────────────────────┐  │                           │
│  │  │  8×8 Grid of Gaussian Units    │  │  (64 dims)                │
│  │  │  Each fires based on distance  │  │                           │
│  │  │  from target to its position   │  │                           │
│  │  └────────────────────────────────┘  │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │         TIME REMAINING                │  (1 dim)                  │
│  │  Normalized time left to hold         │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ concatenate                                      │
│                    ▼                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │    OBSERVATION VECTOR (77 dims)      │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ forward pass                                     │
│                    ▼                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │        POLICY NETWORK                 │                           │
│  │   (MLP or RNN with hidden state)     │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ output                                           │
│                    ▼                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │    MUSCLE ACTIVATIONS (4 dims)       │                           │
│  │  [shoulder_flex, shoulder_ext,       │                           │
│  │   elbow_flex, elbow_ext] ∈ [0,1]    │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ apply to muscles                                 │
│                    ▼                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │        MUSCLE ACTUATION               │                           │
│  │  Muscles generate forces on joints   │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ physics simulation                               │
│                    ▼                                                  │
│  ┌──────────────────────────────────────┐                           │
│  │         NEW STATE                     │                           │
│  │  Updated joint angles, velocities,   │                           │
│  │  hand position, etc.                 │                           │
│  └──────────────────────────────────────┘                           │
│                    │                                                  │
│                    │ loop back                                        │
│                    └──────────────────────────────────────────────> │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Network Architectures

### Teacher MLP
```
Input (77)
    │
    ▼
Linear(77 → 256)
    │
    ▼
ReLU
    │
    ▼
Linear(256 → 256)
    │
    ▼
ReLU
    │
    ▼
Linear(256 → 4)
    │
    ▼
Sigmoid
    │
    ▼
Output (4) ∈ [0,1]
```

### Student RNN
```
Input (77)          Hidden State (128)
    │                      │
    │                      │
    └──────┬───────────────┘
           │
           ▼
    SimpleRNNCell
    (W_ih @ input + W_hh @ hidden)
           │
           ▼
         ReLU
           │
           ▼
    New Hidden (128)
           │
           ▼
    Linear(128 → 4)
           │
           ▼
       Sigmoid
           │
           ▼
    Output (4) ∈ [0,1]
```

## Place Cell Grid Visualization

```
Target Encoding: 8×8 Grid in (x,z) workspace

        z
        ▲
  0.7   │  ○  ○  ○  ○  ○  ○  ○  ○
        │
  0.6   │  ○  ○  ○  ◉  ○  ○  ○  ○    ◉ = high activation
        │                              (target nearby)
  0.5   │  ○  ○  ●  ◉  ●  ○  ○  ○    ● = medium activation
        │                              
  0.4   │  ○  ○  ○  ○  ○  ○  ○  ○    ○ = low activation
        │
  0.3   │  ○  ○  ○  ○  ○  ○  ○  ○
        │
        └──────────────────────────────> x
         0.1        0.35          0.6

Each cell's activation = exp(-distance²/(2σ²))
64 cells total provide distributed spatial encoding
```

## Reward Structure

```
Total Reward = Distance Penalty + Reach Bonus + Hold Bonus + Success Bonus

┌────────────────┬─────────────┬──────────────────────────┐
│   Component    │    Value    │        Condition         │
├────────────────┼─────────────┼──────────────────────────┤
│ Distance       │ -2.0 × dist │ Always (every step)      │
│ At Target      │ +5.0        │ If dist < 3cm            │
│ Holding        │ +10.0 × dt  │ If holding in 4cm region │
│ Success        │ +100.0      │ If held for required time│
└────────────────┴─────────────┴──────────────────────────┘

dt = timestep (0.01s)
```

## Key Hyperparameters

```
┌─────────────────────┬──────────────┬─────────────────────┐
│     Parameter       │    Value     │      Component      │
├─────────────────────┼──────────────┼─────────────────────┤
│ Learning Rate (MLP) │ 3×10⁻⁴       │ PPO optimizer       │
│ Learning Rate (RNN) │ 1×10⁻³       │ BC optimizer        │
│ Discount γ          │ 0.99         │ PPO                 │
│ GAE λ               │ 0.95         │ PPO                 │
│ Clip ε              │ 0.2          │ PPO                 │
│ Batch Size          │ 64/32        │ PPO/BC              │
│ Sequence Length     │ 50           │ BC (RNN training)   │
│ Hidden Size (MLP)   │ 256          │ MLP architecture    │
│ Hidden Size (RNN)   │ 128          │ RNN architecture    │
│ Place Cell Grid     │ 8×8          │ Spatial encoding    │
│ Place Cell σ        │ 0.08         │ Tuning width        │
└─────────────────────┴──────────────┴─────────────────────┘
```
