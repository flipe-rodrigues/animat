from utils import *
from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict
import numpy as np

"""
.########.....###....########.....###....##.....##
.##.....##...##.##...##.....##...##.##...###...###
.##.....##..##...##..##.....##..##...##..####.####
.########..##.....##.########..##.....##.##.###.##
.##........#########.##...##...#########.##.....##
.##........##.....##.##....##..##.....##.##.....##
.##........##.....##.##.....##.##.....##.##.....##
"""


@dataclass
class Parameter:
    """Unified parameter class for both weights and biases with constraint caching"""

    shape: Tuple[int, ...]
    bounds: Tuple[float, float] = (-np.inf, np.inf)
    constraint: Callable = lambda x: x
    _values: np.ndarray = field(default=None, init=False, repr=False)
    _constrained_cache: np.ndarray = field(default=None, init=False, repr=False)
    _cache_valid: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self._values = np.zeros(self.shape)

    @property
    def values(self):
        """Get raw values without constraint"""
        return self._values

    @values.setter
    def values(self, value):
        """Set raw values and invalidate cache"""
        self._values = value
        self._cache_valid = False

    def get_constrained(self):
        """Get constrained values with caching"""
        if not self._cache_valid:
            self._constrained_cache = self.constraint(self._values)
            self._cache_valid = True
        return self._constrained_cache

    def __array__(self):
        """Called when used in numpy operations"""
        return self.get_constrained()

    def __matmul__(self, other):
        """Support @ operator for matrix multiplication"""
        return self.get_constrained() @ other

    def __rmatmul__(self, other):
        """Support @ operator for right-side multiplication"""
        return other @ self.get_constrained()

    def __add__(self, other):
        """Support + operator"""
        return self.get_constrained() + other

    def __radd__(self, other):
        """Support + operator for right-side addition"""
        return other + self.get_constrained()

    @property
    def size(self):
        """Total number of elements"""
        return self._values.size

    def flatten(self):
        """Return flattened raw values"""
        return self._values.flatten()


# Backwards compatibility aliases
Weights = Parameter
Biases = Parameter


"""
.##....##.##.....##.........########..##....##.##....##
.###...##.###...###.........##.....##.###...##.###...##
.####..##.####.####.........##.....##.####..##.####..##
.##.##.##.##.###.##.#######.########..##.##.##.##.##.##
.##..####.##.....##.........##...##...##..####.##..####
.##...###.##.....##.........##....##..##...###.##...###
.##....##.##.....##.........##.....##.##....##.##....##
"""


class NeuroMuscularRNN:
    """Base class for neuromuscular RNNs with configurable weights and biases"""

    def __init__(
        self,
        target_size,
        length_size,
        velocity_size,
        force_size,
        hidden_size,
        output_size,
        activation,
        smoothing_factor=1.0,
        use_bias=True,
    ):
        self.target_size = target_size
        self.length_size = length_size
        self.velocity_size = velocity_size
        self.force_size = force_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.smoothing_factor = smoothing_factor
        self.smoothing_complement = 1.0 - smoothing_factor  # Pre-compute
        self.use_bias = use_bias

        # Initialize architecture
        self._init_weights()
        self._init_biases() if use_bias else self._set_empty_biases()

        # Set initialization function
        self.weight_init_fcn = he_init if activation == relu else xavier_init

        # Initialize values and state
        self.init_weight_values()
        if self.use_bias:
            self.init_bias_values()
        self.init_state()

        # Cache parameter counts
        self._cache_param_info()

    def _init_weights(self):
        """Initialize weight matrix specifications - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _init_weights()")

    def _init_biases(self):
        """Initialize bias specifications - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _init_biases()")

    def _set_empty_biases(self):
        """Set empty biases dict when use_bias=False"""
        self.biases = {}

    def _cache_param_info(self):
        """Cache parameter counts to avoid repeated computation"""
        self.num_weights = sum(w.size for w in self.weights.values())
        self.num_biases = sum(b.size for b in self.biases.values())
        self.num_params = self.num_weights + self.num_biases

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        attrs = [
            "target_size",
            "length_size",
            "velocity_size",
            "force_size",
            "hidden_size",
            "output_size",
            "smoothing_factor",
            "use_bias",
        ]
        if any(getattr(self, a) != getattr(other, a) for a in attrs):
            return False

        return np.allclose(self.get_params(), other.get_params())

    def init_weight_values(self):
        """Initialize all weight matrices, clipping to respect bounds"""
        for weight_matrix in self.weights.values():
            weights = self.weight_init_fcn(
                n_in=weight_matrix.shape[1], n_out=weight_matrix.shape[0]
            )
            # Only clip if bounds are finite
            if weight_matrix.bounds != (-np.inf, np.inf):
                weights = np.clip(weights, *weight_matrix.bounds)
            weight_matrix.values = weights

    def init_bias_values(self):
        """Initialize all bias vectors to zero, clipping to respect bounds"""
        for bias_vector in self.biases.values():
            biases = np.zeros(bias_vector.shape)
            # Only clip if bounds are finite
            if bias_vector.bounds != (-np.inf, np.inf):
                biases = np.clip(biases, *bias_vector.bounds)
            bias_vector.values = biases

    def init_state(self):
        """Initialize hidden states - override in subclasses"""
        raise NotImplementedError("Subclasses must implement init_state()")

    def step(self, tgt_obs, len_obs, vel_obs, frc_obs):
        """Compute one RNN step - override in subclasses"""
        raise NotImplementedError("Subclasses must implement step()")

    def get_params(self):
        """Get flattened parameter vector"""
        # Use list comprehension for slightly better performance
        arrays = [w.flatten() for w in self.weights.values()]
        if self.biases:
            arrays.extend(b.flatten() for b in self.biases.values())
        return np.concatenate(arrays)

    def get_weights(self):
        """Get flattened weight vector"""
        arrays = [w.flatten() for w in self.weights.values()]
        return np.concatenate(arrays)

    def get_bounds(self):
        """Get parameter bounds for optimization as numpy array of shape (n_params, 2)"""
        bounds = []
        for weight_matrix in self.weights.values():
            bounds.extend([weight_matrix.bounds] * weight_matrix.size)
        for bias_vector in self.biases.values():
            bounds.extend([bias_vector.bounds] * bias_vector.size)
        return np.array(bounds)

    def set_params(self, params):
        """Set parameters from flattened vector"""
        idx = 0
        for weight_matrix in self.weights.values():
            size = weight_matrix.size
            weight_matrix.values = params[idx : idx + size].reshape(weight_matrix.shape)
            idx += size
        for bias_vector in self.biases.values():
            size = bias_vector.size
            bias_vector.values = params[idx : idx + size].reshape(bias_vector.shape)
            idx += size

    def from_params(self, params):
        """Create new instance with specified parameters"""
        rnn = self.__class__(
            self.target_size,
            self.length_size,
            self.velocity_size,
            self.force_size,
            self.hidden_size,
            self.output_size,
            self.activation,
            self.smoothing_factor,
            self.use_bias,
        )
        rnn.set_params(params)
        return rnn

    def copy(self):
        """Create an independent copy"""
        return self.from_params(self.get_params())

    def reset_state(self):
        """Reset internal state"""
        self.init_state()


"""
....###.............#######..##....##.##.......##....##
...##.##...........##.....##.###...##.##........##..##.
..##...##..........##.....##.####..##.##.........####..
.##.....##.#######.##.....##.##.##.##.##..........##...
.#########.........##.....##.##..####.##..........##...
.##.....##.........##.....##.##...###.##..........##...
.##.....##..........#######..##....##.########....##...
"""


class AlphaOnlyRNN(NeuroMuscularRNN):
    """RNN with only alpha motoneurons - direct target to action mapping"""

    def _init_weights(self):
        """Initialize weight matrix specifications"""
        self.weights = {
            "tgt2a": Weights((self.output_size, self.target_size), constraint=abs),
        }

    def _init_biases(self):
        """Initialize bias specifications"""
        self.biases = {
            "a": Biases((self.output_size,)),
        }

    def init_state(self):
        """Initialize hidden states"""
        self.a = np.zeros(self.output_size)

    def step(self, tgt_obs, len_obs, vel_obs, frc_obs):
        """Compute one RNN step"""
        # Compute inputs to alpha motoneurons
        a_input = self.weights["tgt2a"] @ tgt_obs
        if self.use_bias:
            a_input += self.biases["a"]

        # Update alpha motoneurons with pre-computed complement
        self.a = self.smoothing_complement * self.a + self.smoothing_factor * logistic(
            a_input
        )

        return self.a


"""
.########.##.....##.##.......##......
.##.......##.....##.##.......##......
.##.......##.....##.##.......##......
.######...##.....##.##.......##......
.##.......##.....##.##.......##......
.##.......##.....##.##.......##......
.##........#######..########.########
"""


class FullRNN(NeuroMuscularRNN):
    """Full RNN with hidden layer, gamma motoneurons, and alpha motoneurons"""

    def _init_weights(self):
        """Initialize weight matrix specifications"""
        self.weights = {
            "tgt2h": Weights((self.hidden_size, self.target_size), constraint=abs),
            "len2h": Weights((self.hidden_size, self.length_size), constraint=abs),
            "vel2h": Weights((self.hidden_size, self.velocity_size), constraint=abs),
            "frc2h": Weights((self.hidden_size, self.force_size), constraint=abs),
            "h2h": Weights((self.hidden_size, self.hidden_size)),
            "a2h": Weights((self.hidden_size, self.output_size), constraint=abs),
            "h2gs": Weights((self.output_size, self.hidden_size)),
            "h2gd": Weights((self.output_size, self.hidden_size)),
            "len2a": Weights((self.output_size, self.length_size)),
            "vel2a": Weights((self.output_size, self.velocity_size)),
            "h2a": Weights((self.output_size, self.hidden_size)),
        }

    def _init_biases(self):
        """Initialize bias specifications"""
        self.biases = {
            "h": Biases((self.hidden_size,)),
            "gs": Biases((self.output_size,)),
            "gd": Biases((self.output_size,)),
            "a": Biases((self.output_size,)),
        }

    def init_state(self):
        """Initialize hidden states"""
        self.h = np.zeros(self.hidden_size)
        self.gs = np.zeros(self.output_size)
        self.gd = np.zeros(self.output_size)
        self.a = np.zeros(self.output_size)

    def step(self, tgt_obs, len_obs, vel_obs, frc_obs):
        """Compute one RNN step"""
        # Pre-compute modulated inputs (avoid redundant computation)
        len_modulated = len_obs + self.gs
        vel_modulated = vel_obs * self.gd

        # Compute inputs to hidden layer
        h_input = (
            self.weights["tgt2h"] @ tgt_obs
            + self.weights["len2h"] @ len_modulated
            + self.weights["vel2h"] @ vel_modulated
            + self.weights["frc2h"] @ frc_obs
            + self.weights["h2h"] @ self.h
            + self.weights["a2h"] @ self.a
        )
        if self.use_bias:
            h_input += self.biases["h"]

        # Compute inputs to gamma motoneurons
        gs_input = self.weights["h2gs"] @ self.h
        gd_input = self.weights["h2gd"] @ self.h
        if self.use_bias:
            gs_input += self.biases["gs"]
            gd_input += self.biases["gd"]

        # Compute inputs to alpha motoneurons
        a_input = (
            self.weights["h2a"] @ self.h
            + self.weights["len2a"] @ len_modulated
            + self.weights["vel2a"] @ vel_modulated
        )
        if self.use_bias:
            a_input += self.biases["a"]

        # Update all states with pre-computed complement
        self.h = (
            self.smoothing_complement * self.h
            + self.smoothing_factor * self.activation(h_input)
        )
        self.gs = (
            self.smoothing_complement * self.gs
            + self.smoothing_factor * logistic(gs_input)
        )
        self.gd = (
            self.smoothing_complement * self.gd
            + self.smoothing_factor * logistic(gd_input)
        )
        self.a = self.smoothing_complement * self.a + self.smoothing_factor * logistic(
            a_input
        )

        return self.a
