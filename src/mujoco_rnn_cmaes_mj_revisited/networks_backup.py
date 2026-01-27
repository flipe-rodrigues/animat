from utils import *


class NeuroMuscularRNN:
    def __init__(
        self,
        input_size_tgt,
        input_size_len,
        input_size_vel,
        input_size_frc,
        hidden_size,
        output_size,
        activation,
        smoothing_factor=1.0,
    ):
        self.input_size_tgt = input_size_tgt
        self.input_size_len = input_size_len
        self.input_size_vel = input_size_vel
        self.input_size_frc = input_size_frc
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.smoothing_factor = smoothing_factor

        self._init_weight_specs()
        self._init_bias_specs()

        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init

        self.init_weights()
        self.init_biases()
        self.init_state()
        self.num_params = sum(size for _, size, _, _ in self._weight_specs) + sum(
            size for _, size, _, _ in self._bias_specs
        )

    def _weight_spec(self, name, in_size, out_size, bounds=(-np.inf, np.inf)):
        return (name, in_size * out_size, (out_size, in_size), bounds)

    def _bias_spec(self, name, size, bounds=(-np.inf, np.inf)):
        return (name, size, (size,), bounds)

    def _init_weight_specs(self):
        self._weight_specs = [
            self._weight_spec("W_tgt2h", self.input_size_tgt, self.hidden_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_tgt2gs", self.input_size_tgt, self.output_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_tgt2gd", self.input_size_tgt, self.output_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_tgt2a", self.input_size_tgt, self.output_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_len2h", self.input_size_len, self.hidden_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_vel2h", self.input_size_vel, self.hidden_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_frc2h", self.input_size_frc, self.hidden_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_h2h", self.hidden_size, self.hidden_size),
            self._weight_spec("W_h2gs", self.hidden_size, self.output_size),
            self._weight_spec("W_h2gd", self.hidden_size, self.output_size),
            self._weight_spec("W_len2a", self.input_size_len, self.output_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_vel2a", self.input_size_vel, self.output_size, bounds=(0.0, np.inf)),
            self._weight_spec("W_h2a", self.hidden_size, self.output_size),
            self._weight_spec("W_a2h", self.output_size, self.hidden_size, bounds=(0.0, np.inf)),
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
            "input_size_tgt",
            "input_size_len",
            "input_size_vel",
            "input_size_frc",
            "hidden_size",
            "output_size",
            "smoothing_factor",
        ]
        if any(getattr(self, a) != getattr(other, a) for a in attrs):
            return False

        return np.allclose(self.get_params(), other.get_params())

    def init_weights(self):
        """Initialize all weight matrices"""
        for name, _, shape, _ in self._weight_specs:
            setattr(self, name, self.init_fcn(n_in=shape[1], n_out=shape[0]))

    def init_biases(self):
        """Initialize all bias vectors to zero"""
        for name, _, shape, _ in self._bias_specs:
            setattr(self, name, np.zeros(shape))

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
            self.W_tgt2h @ tgt_obs
            + self.W_len2h @ (len_obs + self.gs)
            + self.W_vel2h @ (vel_obs * self.gd)
            + self.W_frc2h @ frc_obs
            + self.W_h2h @ self.h
            + self.W_a2h @ self.a
            + self.b_h
        )
        self.h = (
            1 - self.smoothing_factor
        ) * self.h + self.smoothing_factor * self.activation(h_input)

        # Compute inputs to gamma static and dynamic motoneurons
        gs_input = self.W_tgt2gs @ tgt_obs + self.W_h2gs @ self.h + self.b_gs
        gd_input = self.W_tgt2gd @ tgt_obs + self.W_h2gd @ self.h + self.b_gd

        # Compute inputs to alpha motoneurons
        a_input = (
            self.W_tgt2a @ tgt_obs
            + self.W_h2a @ self.h
            + self.W_len2a @ (len_obs + self.gs)
            + self.W_vel2a @ (vel_obs * self.gd)
            + self.b_a
        )

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
        ) * self.a + self.smoothing_factor * relu(a_input)

        return self.a

    def get_params(self):
        """Get flattened parameter vector"""
        weights = [
            getattr(self, name).flatten() for name, _, _, _ in self._weight_specs
        ]
        biases = [getattr(self, name).flatten() for name, _, _, _ in self._bias_specs]
        return np.concatenate(weights + biases)

    def get_bounds(self):
        """Get parameter bounds for optimization"""
        bounds = []
        for _, size, _, param_bounds in self._weight_specs:
            bounds.extend([param_bounds] * size)
        for _, size, _, param_bounds in self._bias_specs:
            bounds.extend([param_bounds] * size)
        return bounds

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
            self.input_size_tgt,
            self.input_size_len,
            self.input_size_vel,
            self.input_size_frc,
            self.hidden_size,
            self.output_size,
            self.activation,
            self.smoothing_factor,
        )
        rnn.set_params(params)
        return rnn

    def copy(self):
        """Create an independent copy"""
        return self.from_params(self.get_params())

    def reset_state(self):
        """Reset internal state"""
        self.init_state()
