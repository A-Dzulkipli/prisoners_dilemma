"""Doubler."""
import jax.numpy as jnp

from ipd.bots import INF, Bot
from ipd.game import C, D


def _doubler_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    n_coop = o.t - n_def
    return jnp.where((o.their_last == D) & (n_coop <= 2 * n_def), D, C), n_def


doubler = Bot("Doubler", lambda p, k: jnp.int32(0), _doubler_act, memory_depth=INF)

STRATEGIES = [doubler]
