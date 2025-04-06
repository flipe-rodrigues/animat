import copy
from utils import *


class RNN:
    def __init__(self, input_size, hidden_size, output_size, activation, alpha):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.alpha = alpha
        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init
        self.init_weights()
        self.init_biases()
        self.init_state()
        self.num_params = len(self.get_params())

    def __eq__(self, other):
        if isinstance(other, RNN):
            return all(self.get_params() == other.get_params())
        return False

    def init_weights(self):
        self.W_in = self.init_fcn(n_in=self.input_size, n_out=self.hidden_size)
        self.W_h = self.init_fcn(n_in=self.hidden_size, n_out=self.hidden_size)
        self.W_out = self.init_fcn(n_in=self.hidden_size, n_out=self.output_size)

    def init_biases(self):
        self.b_h = np.zeros(self.hidden_size)
        self.b_out = np.zeros(self.output_size)

    def init_state(self):
        """Reset hidden state between episodes"""
        self.h = np.zeros(self.hidden_size)
        self.out = np.zeros(self.output_size)

    def step(self, obs):
        """Compute one RNN step"""
        self.h = (1 - self.alpha) * self.h + self.alpha * self.activation(
            self.W_in @ obs + self.W_h @ self.h + self.b_h
        )
        self.out = (1 - self.alpha) * self.out + self.alpha * logistic(
            self.W_out @ self.h + self.b_out
        )
        return self.out

    def get_params(self):
        return np.concatenate(
            [
                self.W_in.flatten(),
                self.W_h.flatten(),
                self.W_out.flatten(),
                self.b_h.flatten(),
                self.b_out.flatten(),
            ]
        )

    def set_params(self, params):
        idx = 0
        W_in_size = self.input_size * self.hidden_size
        W_h_size = self.hidden_size * self.hidden_size
        W_out_size = self.hidden_size * self.output_size

        self.W_in = params[idx : idx + W_in_size].reshape(
            self.input_size, self.hidden_size
        )
        idx += W_in_size
        self.W_h = params[idx : idx + W_h_size].reshape(
            self.hidden_size, self.hidden_size
        )
        idx += W_h_size
        self.W_out = params[idx : idx + W_out_size].reshape(
            self.hidden_size, self.output_size
        )
        idx += W_out_size

        self.b_h = params[idx : idx + self.hidden_size]
        idx += self.hidden_size
        self.b_out = params[idx : idx + self.output_size]

    def from_params(self, params):
        """Return a new RNN with weights and biases from flattened parameters."""
        idx = 0

        def extract(shape):
            nonlocal idx
            size = np.prod(shape)
            param = params[idx : idx + size].reshape(shape)
            idx += size
            return param

        new_rnn = copy.deepcopy(self)
        new_rnn.W_in = extract((self.hidden_size, self.input_size))
        new_rnn.W_h = extract((self.hidden_size, self.hidden_size))
        new_rnn.W_out = extract((self.output_size, self.hidden_size))
        new_rnn.b_h = extract((self.hidden_size,))
        new_rnn.b_out = extract((self.output_size,))
        return new_rnn

    @staticmethod
    def recombine(p1, p2):
        child = RNN(
            p1.input_size,
            p1.hidden_size,
            p1.output_size,
            p1.activation,
            p1.alpha,
        )
        child.W_in = RNN.recombine_matrices(p1.W_in, p2.W_in)
        child.W_h = RNN.recombine_matrices(p1.W_h, p2.W_h)
        child.W_out = RNN.recombine_matrices(p1.W_out, p2.W_out)
        child.b_h = RNN.recombine_matrices(p1.b_h, p2.b_h)
        child.b_out = RNN.recombine_matrices(p1.b_out, p2.b_out)
        return child

    @staticmethod
    def recombine_matrices(A, B):
        mask = np.random.rand(*A.shape) > 0.5
        return np.where(mask, A, B)

    def mutate(self, rate):
        mutant = copy.deepcopy(self)
        mutant.W_in += self.init_fcn(mutant.input_size, mutant.hidden_size) * rate
        mutant.W_h += self.init_fcn(mutant.hidden_size, mutant.hidden_size) * rate
        mutant.W_out += self.init_fcn(mutant.hidden_size, mutant.output_size) * rate
        mutant.b_h += np.random.randn(mutant.hidden_size) * rate
        mutant.b_out += np.random.randn(mutant.output_size) * rate
        return mutant
