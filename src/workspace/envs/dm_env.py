import sys
import numpy as np
from pathlib import Path

# Add workspace to path for imports
workspace_root = Path(__file__).parent.parent  
sys.path.insert(0, str(workspace_root))

from dm_control import composer
from envs.task import ArmReachingTask

NUM_SUBSTEPS = 10


def make_arm_env(random_seed=None):
    """Create and return a composer Environment with the arm task."""
    random_state = np.random.RandomState(random_seed)
    task = ArmReachingTask(random_state=random_state)
    env = composer.Environment(
        task=task,
        time_limit=2.0,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
    return env