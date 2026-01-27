from utils import *


class NeuroMuscularRNN:
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
        self.use_bias = use_bias

        self._init_weight_specs()
        if self.use_bias:
            self._init_bias_specs()
        else:
            self._bias_specs = []

        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init

        self.init_weights()
        if self.use_bias:
            self.init_biases()
        self.init_state()
        self.num_weights = sum(size for _, size, _, _ in self._weight_specs)
        self.num_biases = sum(size for _, size, _, _ in self._bias_specs)
        self.num_params = self.num_weights + self.num_biases

    def _weight_spec(self, name, in_size, out_size, bounds=(-np.inf, np.inf)):
        return (name, in_size * out_size, (out_size, in_size), bounds)

    def _bias_spec(self, name, size, bounds=(-np.inf, np.inf)):
        return (name, size, (size,), bounds)

    def _init_weight_specs(self):
        self._weight_specs = [
            self._weight_spec("W_tgt2h", self.target_size, self.hidden_size),
            self._weight_spec("W_len2h", self.length_size, self.hidden_size),
            self._weight_spec("W_vel2h", self.velocity_size, self.hidden_size),
            self._weight_spec("W_frc2h", self.force_size, self.hidden_size),
            self._weight_spec("W_h2h", self.hidden_size, self.hidden_size),
            self._weight_spec("W_h2gs", self.hidden_size, self.output_size),
            self._weight_spec("W_h2gd", self.hidden_size, self.output_size),
            self._weight_spec("W_len2a", self.length_size, self.output_size),
            self._weight_spec("W_vel2a", self.velocity_size, self.output_size),
            self._weight_spec("W_h2a", self.hidden_size, self.output_size),
            self._weight_spec("W_a2h", self.output_size, self.hidden_size),
        ]

    def _init_bias_specs(self):
        self._bias_specs = [
            self._bias_spec("b_h", self.hidden_size),
            self._bias_spec("b_gs", self.output_size),
            self._bias_spec("b_gd", self.output_size),
            self._bias_spec("b_a", self.output_size),
        ]

    def __eq__(self, other):
        if not isinstance(other, NeuroMuscularRNN):
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

    def init_weights(self):
        """Initialize all weight matrices, clipping to respect bounds"""
        for name, _, shape, bounds in self._weight_specs:
            weights = self.init_fcn(n_in=shape[1], n_out=shape[0])
            if bounds[0] != -np.inf or bounds[1] != np.inf:
                weights = np.clip(weights, bounds[0], bounds[1])
            setattr(self, name, weights)

    def init_biases(self):
        """Initialize all bias vectors to zero, clipping to respect bounds"""
        for name, _, shape, bounds in self._bias_specs:
            biases = np.zeros(shape)
            if bounds[0] != -np.inf or bounds[1] != np.inf:
                biases = np.clip(biases, bounds[0], bounds[1])
            setattr(self, name, biases)

    def init_state(self):
        """Initialize hidden states"""
        self.h = np.zeros(self.hidden_size)
        self.gs = np.zeros(self.output_size)
        self.gd = np.zeros(self.output_size)
        self.a = np.zeros(self.output_size)

    def step(self, tgt_obs, len_obs, vel_obs, frc_obs):
        """Compute one RNN step"""

        # Compute inputs to hidden layer
        h_input = (
            abs(self.W_tgt2h) @ tgt_obs
            + abs(self.W_len2h) @ (len_obs + self.gs)
            + abs(self.W_vel2h) @ (vel_obs * self.gd)
            + abs(self.W_frc2h) @ frc_obs
            + self.W_h2h @ self.h
            + abs(self.W_a2h) @ self.a
        )
        if self.use_bias:
            h_input += self.b_h

        # Compute inputs to gamma static and dynamic motoneurons
        gs_input = self.W_h2gs @ self.h
        if self.use_bias:
            gs_input += self.b_gs
        gd_input = self.W_h2gd @ self.h
        if self.use_bias:
            gd_input += self.b_gd

        # Compute inputs to alpha motoneurons
        a_input = (
            self.W_h2a @ self.h
            + abs(self.W_len2a) @ (len_obs + self.gs)
            + abs(self.W_vel2a) @ (vel_obs * self.gd)
        )
        if self.use_bias:
            a_input += self.b_a

        # Update hidden states
        self.h = (
            1 - self.smoothing_factor
        ) * self.h + self.smoothing_factor * self.activation(h_input)

        # Update gamma static and dynamic motoneurons
        self.gs = (
            1 - self.smoothing_factor
        ) * self.gs + self.smoothing_factor * self.activation(gs_input)
        self.gd = (
            1 - self.smoothing_factor
        ) * self.gd + self.smoothing_factor * self.activation(gd_input)

        # Update alpha motoneurons
        self.a = (
            1 - self.smoothing_factor
        ) * self.a + self.smoothing_factor * logistic(a_input)

        return self.a

    def get_params(self):
        """Get flattened parameter vector"""
        weights = [
            getattr(self, name).flatten() for name, _, _, _ in self._weight_specs
        ]
        biases = [getattr(self, name).flatten() for name, _, _, _ in self._bias_specs]
        return np.concatenate(weights + biases)

    def get_bounds(self):
        """Get parameter bounds for optimization as numpy array of shape (n_params, 2)"""
        bounds = []
        for _, size, _, param_bounds in self._weight_specs:
            bounds.extend([param_bounds] * size)
        for _, size, _, param_bounds in self._bias_specs:
            bounds.extend([param_bounds] * size)
        return np.array(bounds)

    def set_params(self, params):
        """Set parameters from flattened vector"""
        idx = 0
        for name, size, shape, _ in self._weight_specs:
            setattr(self, name, params[idx : idx + size].reshape(shape))
            idx += size
        for name, size, shape, _ in self._bias_specs:
            setattr(self, name, params[idx : idx + size].reshape(shape))
            idx += size

    def from_params(self, params):
        """Create new instance with specified parameters"""
        rnn = NeuroMuscularRNN(
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
