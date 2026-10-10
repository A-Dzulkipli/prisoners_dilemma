"""Adaptive."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, last_payoff
from ipd.game import C, D

INITIAL_PLAYS = jnp.array([C] * 6 + [D] * 5, jnp.int32)


def _adaptive_act(p, scores, o, k):
    # scores[a]: total payoff from rounds where I played a
    scores = scores.at[jnp.maximum(o.my_last, 0)].add(last_payoff(o))
    action = jnp.where(o.t < 11, INITIAL_PLAYS[jnp.minimum(o.t, 10)], jnp.where(scores[C] > scores[D], C, D))
    return action, scores


adaptive = Bot("Adaptive", lambda p, k: jnp.zeros(2, jnp.float32), _adaptive_act, memory_depth=INF)

STRATEGIES = [adaptive]
