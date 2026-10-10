"""Worse and Worse strategies."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob, no_state
from ipd.game import C, D


def _waw_act(p, s, o, k):
    return coop_with_prob(k, 1 - (o.t + 1) / 1000), s


def _knowledgeable_act(p, s, o, k):
    # unknown length is the library's infinite length, where the defection probability is 0
    return coop_with_prob(k, jnp.where(o.turns > 0, 1 - (o.t + 1) / o.turns, 1.0)), s


def _waw2_act(p, s, o, k):
    late = coop_with_prob(k, 20 / (o.t + 1))
    return jnp.where(o.t == 0, C, jnp.where(o.t < 20, o.their_last, late)), s


def _waw3_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    p_coop = 1 - n_def / jnp.maximum(o.t, 1)
    return jnp.where(o.t == 0, C, coop_with_prob(k, p_coop)), n_def


worse_and_worse = Bot("Worse and Worse", no_state, _waw_act, memory_depth=INF, stochastic=True)
knowledgeable_worse_and_worse = Bot(
    "Knowledgeable Worse and Worse", no_state, _knowledgeable_act, memory_depth=INF, stochastic=True
)
worse_and_worse2 = Bot("Worse and Worse 2", no_state, _waw2_act, memory_depth=INF, stochastic=True)
worse_and_worse3 = Bot("Worse and Worse 3", lambda p, k: jnp.int32(0), _waw3_act, memory_depth=INF, stochastic=True)

STRATEGIES = [knowledgeable_worse_and_worse, worse_and_worse, worse_and_worse2, worse_and_worse3]
