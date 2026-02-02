"""
Project-wide constants for muscle-driven arm control.

All default values should be defined here and imported elsewhere.
"""

from typing import Final, Tuple

# =============================================================================
# Sensor and Neuron Types
# =============================================================================

SENSOR_TYPE_LENGTH: Final[str] = "actuatorpos"
SENSOR_TYPE_VELOCITY: Final[str] = "actuatorvel"
SENSOR_TYPE_FORCE: Final[str] = "actuatorfrc"

SENSORY_TYPE_IA: Final[str] = "type_Ia"  # Velocity-sensitive
SENSORY_TYPE_II: Final[str] = "type_II"  # Length-sensitive
SENSORY_TYPE_IB: Final[str] = "type_Ib"  # Force-sensitive

MOTOR_TYPE_ALPHA: Final[str] = "alpha"
MOTOR_TYPE_GAMMA_STATIC: Final[str] = "gamma_static"
MOTOR_TYPE_GAMMA_DYNAMIC: Final[str] = "gamma_dynamic"

# =============================================================================
# Physics and Simulation
# =============================================================================

DEFAULT_TIMESTEP: Final[float] = 0.01  # 10ms
DEFAULT_MAX_EPISODE_TIME: Final[float] = 3.0  # seconds
DEFAULT_MAX_EPISODE_STEPS: Final[int] = 300

# Muscle activation range
MUSCLE_ACTIVATION_MIN: Final[float] = 0.0
MUSCLE_ACTIVATION_MAX: Final[float] = 1.0

# Gamma modulation range
GAMMA_MIN: Final[float] = 0.0
GAMMA_MAX: Final[float] = 2.0

# Observation clipping
OBS_CLIP_MIN: Final[float] = -10.0
OBS_CLIP_MAX: Final[float] = 10.0

# =============================================================================
# Environment Defaults
# =============================================================================

DEFAULT_PRE_DELAY_RANGE: Final[Tuple[float, float]] = (0.2, 0.5)
DEFAULT_HOLD_DURATION_RANGE: Final[Tuple[float, float]] = (0.3, 0.8)
DEFAULT_POST_DELAY_RANGE: Final[Tuple[float, float]] = (0.1, 0.3)
DEFAULT_REACH_THRESHOLD: Final[float] = 0.05  # meters
DEFAULT_HOLD_REWARD_WEIGHT: Final[float] = 1.0
DEFAULT_ENERGY_PENALTY_WEIGHT: Final[float] = 0.01
DEFAULT_REACH_BONUS: Final[float] = 0.5
DEFAULT_SUCCESS_REWARD: Final[float] = 1.0

# =============================================================================
# Network Architecture
# =============================================================================

DEFAULT_NUM_MUSCLES: Final[int] = 4
DEFAULT_RNN_HIDDEN_SIZE: Final[int] = 32
DEFAULT_SHOW_RNN_UNITS: Final[int] = 32  # For visualization
DEFAULT_MLP_HIDDEN_SIZES: Final[Tuple[int, ...]] = (256, 256, 128)
DEFAULT_TARGET_GRID_SIZE: Final[int] = 4
DEFAULT_TARGET_SIGMA: Final[float] = 1.0

# =============================================================================
# CMA-ES Training
# =============================================================================

DEFAULT_POPULATION_SIZE: Final[int] = 64
DEFAULT_NUM_GENERATIONS: Final[int] = 500
DEFAULT_CMAES_SIGMA: Final[float] = 0.25
DEFAULT_NUM_EVAL_EPISODES: Final[int] = 5
DEFAULT_TARGET_FITNESS: Final[float] = 50.0
DEFAULT_PATIENCE: Final[int] = 50
DEFAULT_CHECKPOINT_EVERY: Final[int] = 25

# =============================================================================
# Distillation Training
# =============================================================================

DEFAULT_TEACHER_LR: Final[float] = 1e-3
DEFAULT_TEACHER_EPOCHS: Final[int] = 100
DEFAULT_TEACHER_BATCH_SIZE: Final[int] = 64
DEFAULT_TEACHER_DATA_EPISODES: Final[int] = 1000

DEFAULT_STUDENT_LR: Final[float] = 1e-4
DEFAULT_STUDENT_EPOCHS: Final[int] = 200
DEFAULT_STUDENT_BATCH_SIZE: Final[int] = 32
DEFAULT_STUDENT_SEQ_LEN: Final[int] = 50

DEFAULT_ACTION_LOSS_WEIGHT: Final[float] = 1.0
DEFAULT_HIDDEN_LOSS_WEIGHT: Final[float] = 0.1

# =============================================================================
# Calibration
# =============================================================================

DEFAULT_CALIBRATION_EPISODES: Final[int] = 100
DEFAULT_CALIBRATION_STEPS: Final[int] = 200

# =============================================================================
# Workspace Estimation
# =============================================================================

DEFAULT_WORKSPACE_SAMPLES: Final[int] = 1000

# =============================================================================
# Visualization
# =============================================================================

DEFAULT_VIDEO_FPS: Final[int] = 30
DEFAULT_DPI: Final[int] = 100
DEFAULT_RENDER_WIDTH: Final[int] = 640
DEFAULT_RENDER_HEIGHT: Final[int] = 480

# =============================================================================
# File Extensions
# =============================================================================

CHECKPOINT_EXTENSION: Final[str] = ".pth"
CONFIG_EXTENSION: Final[str] = ".json"
STATS_EXTENSION: Final[str] = ".pkl"
