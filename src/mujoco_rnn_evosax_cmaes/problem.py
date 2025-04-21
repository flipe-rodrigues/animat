from functools import partial

from evosax.problems import Problem, State
from evosax.types import Fitness, Metrics, Population, Solution
from flax import struct

import jax
import jax.numpy as jnp
from env import reset_env, get_obs, step_env, get_end_effector_pos
from rnn import rnn_step, unpack_candidate

@struct.dataclass
class State(State):
    pass

class RNNReachProblem(Problem):
    def __init__(
        self, model, input_dim, hidden_dim, output_dim, episode_len=200, batch_size=128
    ):
        self.model = model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.episode_len = episode_len
        self.batch_size = batch_size
        self._num_dims = self._get_param_dim(input_dim, hidden_dim, output_dim)

    def _get_param_dim(self, I, H, O):
        return H * I + H * H + H + O * H + O

    def simulate_single(self, flat_params, target):
        rnn_params, output_params = unpack_candidate(
            flat_params, self.input_dim, self.hidden_dim, self.output_dim
        )
        data = reset_env(self.model)
        h = jnp.zeros((self.hidden_dim,))
        loss = 0.0

        def step_fn(carry, _):
            data, h, total_loss = carry
            obs = get_obs(data)
            x = jnp.concatenate([obs, target])
            h_new, action = rnn_step(rnn_params, output_params, h, x)
            data = step_env(self.model, data, action)
            end_effector = get_end_effector_pos(data)
            dist = jnp.linalg.norm(end_effector - target)
            return (data, h_new, total_loss + dist), 0

        (data, h, total_loss), _ = jax.lax.scan(
            step_fn, (data, h, 0.0), None, length=self.episode_len
        )
        return total_loss

    def evaluate(self, solutions: jnp.ndarray, rng: jax.random.PRNGKey):
        targets = jax.random.uniform(rng, (self.batch_size, 3), minval=-1.0, maxval=1.0)
        sim_fn = jax.vmap(self.simulate_single, in_axes=(0, 0))
        losses = sim_fn(solutions, targets)
        return losses

    def eval(self, solution: jnp.ndarray, rng: jax.random.PRNGKey) -> float:
        target = jax.random.uniform(rng, (3,), minval=-1.0, maxval=1.0)
        return self.simulate_single(solution, target)

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self, key: jax.Array, solutions: Population, state: State
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a batch of solutions."""
        fn_val, state, info = self.meta_problem.eval(
            key, solutions, state, self._params
        )
        return fn_val, state, info

    # @partial(jax.jit, static_argnames=("self",))
    # def eval(
    #     self,
    #     key: jax.Array,
    #     solutions: Population,
    #     state: State,
    # ) -> tuple[Fitness, State, Metrics]:
    #     """Evaluate a batch of solutions."""
    #     raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        return jax.random.uniform(
            key, shape=(self.batch_size, self._num_dims), minval=-1.0, maxval=1.0
        )
