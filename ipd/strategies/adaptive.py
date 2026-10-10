"""Adaptive, ported from axelrod's adaptive.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D

INITIAL_PLAYS = jnp.array([C] * 6 + [D] * 5, jnp.int32)


def _adaptive_act(p, scores, o, k):
    # scores[a]: total payoff from rounds where I played a
    gain = o.my_payoff[jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)]
    scores = scores.at[jnp.maximum(o.my_last, 0)].add(jnp.where(o.t > 0, gain, 0.0))
    action = jnp.where(o.t < 11, INITIAL_PLAYS[jnp.minimum(o.t, 10)], jnp.where(scores[C] > scores[D], C, D))
    return action, scores


adaptive = Bot("Adaptive", lambda p, k: jnp.zeros(2, jnp.float32), _adaptive_act, memory_depth=float("inf"))

STRATEGIES = [adaptive]
