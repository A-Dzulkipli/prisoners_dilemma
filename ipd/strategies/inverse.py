"""Inverse, ported from axelrod's inverse.py."""
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C, D


def _inverse_act(p, last_d, o, k):
    # last_d: round of the opponent's latest defection, -1 if none; defect with prob 1 / (rounds since)
    last_d = jnp.where(o.their_last == D, o.t - 1, last_d)
    since = jnp.maximum(o.t - last_d, 1)
    return jnp.where(last_d < 0, C, _coop_with_prob(k, 1 - 1 / since)), last_d


inverse = Bot("Inverse", lambda p, k: jnp.int32(-1), _inverse_act, memory_depth=float("inf"), stochastic=True)

STRATEGIES = [inverse]
