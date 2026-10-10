"""VeryBad, ported from axelrod's verybad.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D


def _verybad_act(p, n_coop, o, k):
    n_coop = n_coop + (o.their_last == C)
    action = jnp.select([o.t < 3, 2 * n_coop > o.t, 2 * n_coop < o.t], [C, C, D], o.their_last)
    return action, n_coop


verybad = Bot("VeryBad", lambda p, k: jnp.int32(0), _verybad_act, memory_depth=float("inf"))

STRATEGIES = [verybad]
