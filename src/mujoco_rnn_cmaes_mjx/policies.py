from typing import Tuple
from typing import Callable, Any, Optional
from functools import partial

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax.struct import dataclass
from flax.linen import initializers

from evojax.policy.base import PolicyState
from evojax.policy.base import PolicyNetwork
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

logger = create_logger("RNNPolicy")


@dataclass
class RNNState(PolicyState):
    h: jnp.ndarray


class RNN(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_act_fn: Callable = nn.tanh
    output_act_fn: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, inputs, carry):

        input_layer = partial(
            nn.Dense,
            features=self.hidden_dim,
            use_bias=False,
            kernel_init=initializers.xavier_normal(),
        )
        recurrent_layer = partial(
            nn.Dense,
            features=self.hidden_dim,
            use_bias=False,
            kernel_init=initializers.xavier_normal(),
        )
        output_layer = partial(
            nn.Dense,
            features=self.output_dim,
            use_bias=False,
            kernel_init=initializers.xavier_normal(),
        )

        carry = self.hidden_act_fn(
            input_layer(name="i")(inputs) + recurrent_layer(name="h")(carry)
        )
        output = self.output_act_fn(output_layer(name="o")(carry))

        return output, carry


class RNNPolicy(PolicyNetwork):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):

        model = RNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_act_fn=nn.tanh,
            output_act_fn=nn.sigmoid,
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        params = model.init(
            jax.random.PRNGKey(0),
            jnp.ones([1, input_dim]),
            (jnp.zeros([1, hidden_dim]), jnp.zeros([1, hidden_dim])),
        )
        self.num_params, format_params_fn = get_params_format_fn(params)
        print("RNNPolicy.num_params = {}".format(self.num_params))

        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def reset(self, states: TaskState) -> PolicyState:
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        batch_size = states.obs.shape[0]
        h = jnp.zeros((batch_size, self.hidden_dim))
        return RNNState(keys=keys, h=h)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: RNNState
    ) -> Tuple[jnp.ndarray, RNNState]:
        # Calling `self._format_params_fn` unflattens the parameters so that
        # our Flax model can take that as an input.
        params = self._format_params_fn(params)

        # Now we return the actions and the updated `p_states`.
        actions, hx = self._forward_fn(params, t_states.obs, p_states.h)
        return actions, RNNState(keys=p_states.keys, h=hx)