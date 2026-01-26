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
        tau,
    ):
        self.input_size_tgt = input_size_tgt
        self.input_size_len = input_size_len
        self.input_size_vel = input_size_vel
        self.input_size_frc = input_size_frc
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.tau = tau
        self.one_minus_tau = 1 - self.tau

        self._init_weight_specs()

        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init

        self.init_weights()
        self.init_state()
        self.num_params = sum(size for _, size, _ in self._weight_specs)

    def _spec(self, name, in_size, out_size):
        return (name, in_size * out_size, (out_size, in_size))

    def _init_weight_specs(self):
        self._weight_specs = [
            self._spec("W_tgt2h", self.input_size_tgt, self.hidden_size),
            self._spec("W_tgt2gs", self.input_size_tgt, self.output_size),
            self._spec("W_tgt2gd", self.input_size_tgt, self.output_size),
            self._spec("W_tgt2a", self.input_size_tgt, self.output_size),
            self._spec("W_len2h", self.input_size_len, self.hidden_size),
            self._spec("W_vel2h", self.input_size_vel, self.hidden_size),
            self._spec("W_frc2h", self.input_size_frc, self.hidden_size),
            self._spec("W_h2h", self.hidden_size, self.hidden_size),
            self._spec("W_h2gs", self.hidden_size, self.output_size),
            self._spec("W_h2gd", self.hidden_size, self.output_size),
            self._spec("W_len2a", self.input_size_len, self.output_size),
            self._spec("W_vel2a", self.input_size_vel, self.output_size),
            self._spec("W_h2a", self.hidden_size, self.output_size),
            self._spec("W_a2h", self.output_size, self.hidden_size),
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
            "tau",
        ]
        if any(getattr(self, a) != getattr(other, a) for a in attrs):
            return False

        return np.allclose(self.get_params(), other.get_params())

    def init_weights(self):
        """Initialize all weight matrices"""
        for name, _, shape in self._weight_specs:
            setattr(self, name, self.init_fcn(n_in=shape[0], n_out=shape[1]))

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
            + self.W_len2h @ len_obs
            + self.W_vel2h @ vel_obs
            + self.W_frc2h @ frc_obs
            + self.W_h2h @ self.h
            + self.W_a2h @ self.a
        )
        self.h = self.one_minus_tau * self.h + self.tau * self.activation(h_input)

        # Compute inputs to gamma static and dynamic motoneurons
        gs_input = self.W_tgt2gs @ tgt_obs + self.W_h2gs @ self.h
        gd_input = self.W_tgt2gd @ tgt_obs + self.W_h2gd @ self.h

        # Compute inputs to alpha motoneurons
        a_input = (
            self.W_tgt2a @ tgt_obs
            + self.W_h2a @ self.h
            + self.W_len2a @ len_obs
            + self.W_vel2a @ vel_obs
        )

        # Compute lambda (dynamic threshold for alpha-motoneuron recruitment)
        lambda_ = 1.0 - self.gs + a_input - self.gd * vel_obs

        # Update gamma static and dynamic motoneurons
        self.gs = self.one_minus_tau * self.gs + self.tau * self.activation(gs_input)
        self.gd = self.one_minus_tau * self.gd + self.tau * self.activation(gd_input)

        # Update alpha motoneurons
        self.a = self.one_minus_tau * self.a + self.tau * np.maximum(
            0, len_obs - lambda_
        )

        return self.a

    def get_params(self):
        """Get flattened parameter vector"""
        return np.concatenate(
            [getattr(self, name).flatten() for name, _, _ in self._weight_specs]
        )

    def set_params(self, params):
        """Set parameters from flattened vector"""
        idx = 0
        for name, size, shape in self._weight_specs:
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
            self.tau,
        )
        rnn.set_params(params)
        return rnn

    def copy(self):
        """Create an independent copy"""
        return self.from_params(self.get_params())

    def reset_state(self):
        """Reset internal state"""
        self.init_state()
