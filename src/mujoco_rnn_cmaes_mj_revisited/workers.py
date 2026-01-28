import os
import numpy as np
import logging
from functools import wraps
from plants import SequentialReacher
from encoders import GridTargetEncoder
from environments import SequentialReachingEnv

# Global worker state (initialized once per process)
_worker_reacher = None
_worker_encoder = None
_worker_rnn = None
_worker_env = None

# Setup logging (only warnings and errors)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - Worker %(process)d - %(levelname)s - %(message)s",
)


def handle_worker_errors(func):
    """Decorator for robust error handling in workers."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise  # Allow clean shutdown
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return -1e10  # Return terrible fitness on error

    return wrapper


def init_worker(rnn_config_dict, env_config_dict):
    """
    Initialize worker-specific components once per process.

    This is called once when each worker process starts, creating
    persistent objects that are reused across all evaluations in
    that worker.

    Args:
        rnn_config_dict: RNN architecture configuration (includes 'rnn_class')
        env_config_dict: Environment configuration
    """
    global _worker_reacher, _worker_encoder, _worker_rnn, _worker_env

    try:
        # Extract the RNN class from the config
        rnn_class = rnn_config_dict.pop("rnn_class")

        # Create persistent reacher (expensive - do once!)
        _worker_reacher = SequentialReacher(**env_config_dict["plant"])

        # Create persistent encoder
        _worker_encoder = GridTargetEncoder(**env_config_dict["encoder"])

        # Create RNN template using the specified class
        _worker_rnn = rnn_class(**rnn_config_dict)

        # Create lightweight environment wrapper (now it's cheap!)
        # This only stores configuration, not plant/encoder instances
        _worker_env = SequentialReachingEnv(**env_config_dict["env"])

        logging.info(f"Worker {os.getpid() if 'os' in dir() else '?'} initialized successfully")

    except Exception as e:
        logging.error(f"Worker initialization failed: {e}", exc_info=True)
        raise


@handle_worker_errors
def evaluate_worker(params, seed):
    """
    Evaluate a single individual's fitness using persistent components.

    This function is called many times per worker, so it should be
    as lightweight as possible. Heavy initialization is done in init_worker().

    Args:
        params: RNN parameters (flattened numpy array)
        seed: Random seed for this evaluation

    Returns:
        loss: Negative fitness (to minimize)
    """
    global _worker_reacher, _worker_encoder, _worker_rnn, _worker_env

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Update RNN parameters (lightweight operation)
    _worker_rnn.set_params(params)
    _worker_rnn.reset_state()

    # Reset plant state (lightweight operation)
    _worker_reacher.reset()
    _worker_reacher.disable_target()

    # Evaluate fitness using persistent environment
    # Now we pass plant and encoder as arguments instead of recreating env
    loss = -_worker_env.evaluate(
        _worker_rnn, 
        _worker_reacher,
        _worker_encoder,
        seed=seed, 
        render=False, 
        log=False
    )

    return loss


def cleanup_worker():
    """
    Cleanup worker resources.

    Called when worker process is terminating.
    """
    global _worker_reacher, _worker_encoder, _worker_rnn, _worker_env

    if _worker_reacher is not None:
        try:
            _worker_reacher.close()
        except Exception as e:
            logging.warning(f"Error closing reacher: {e}")

    # Clear all globals
    _worker_reacher = None
    _worker_encoder = None
    _worker_rnn = None
    _worker_env = None


# Optional: Shared memory implementation for very large populations
try:
    from multiprocessing import shared_memory

    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    SHARED_MEMORY_AVAILABLE = False

if SHARED_MEMORY_AVAILABLE:
    _shared_params_shm = None
    _shared_params_array = None

    def init_worker_shared(
        shm_name, param_size, pop_size, rnn_config_dict, env_config_dict
    ):
        """
        Initialize worker with shared memory access.

        For very large populations (>1000), this avoids copying parameter arrays
        from main process to each worker.
        """
        global _shared_params_shm, _shared_params_array

        # Initialize standard components
        init_worker(rnn_config_dict, env_config_dict)

        # Attach to shared memory
        try:
            _shared_params_shm = shared_memory.SharedMemory(name=shm_name)
            _shared_params_array = np.ndarray(
                (pop_size, param_size), dtype=np.float64, buffer=_shared_params_shm.buf
            )
            logging.info("Shared memory attached successfully")
        except Exception as e:
            logging.error(f"Failed to attach to shared memory: {e}")
            raise

    @handle_worker_errors
    def evaluate_worker_shared(individual_idx, seed):
        """
        Evaluate using shared memory parameters.

        Args:
            individual_idx: Index into shared parameter array
            seed: Random seed
        """
        global _shared_params_array

        # Copy parameters from shared memory (single copy vs full pickle)
        params = _shared_params_array[individual_idx].copy()

        # Use standard evaluation
        return evaluate_worker(params, seed)

    def cleanup_worker_shared():
        """Cleanup worker with shared memory."""
        global _shared_params_shm

        if _shared_params_shm is not None:
            try:
                _shared_params_shm.close()
            except Exception as e:
                logging.warning(f"Error closing shared memory: {e}")
            _shared_params_shm = None

        cleanup_worker()