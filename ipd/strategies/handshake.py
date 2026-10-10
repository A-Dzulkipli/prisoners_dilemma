"""Handshake."""
import jax.numpy as jnp

from ipd.bots import INF, Bot
from ipd.game import C, D


def _handshake_act(p, ok, o, k):
    # ok: the opponent's first two moves were C, D
    ok = jnp.where(o.t == 1, o.their_last == C, jnp.where(o.t == 2, ok & (o.their_last == D), ok))
    return jnp.select([o.t == 0, o.t == 1, ok], [C, D, C], D), ok


handshake = Bot("Handshake", lambda p, k: jnp.bool_(True), _handshake_act, memory_depth=INF)

STRATEGIES = [handshake]
