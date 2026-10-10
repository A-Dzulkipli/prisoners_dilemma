"""Tricky Defector; Defector itself is in ipd.bots."""
import jax.numpy as jnp

from ipd.bots import INF, Bot
from ipd.game import C, D


def _tricky_defector_act(p, s, o, k):
    # cooperate if the opponent ever cooperated and defected in each of the last three rounds (the
    # library docstring says defect here; its code cooperates)
    coop_seen, d_run = s
    coop_seen = coop_seen | (o.their_last == C)
    d_run = jnp.where(o.their_last == D, d_run + 1, 0)
    return jnp.where(coop_seen & (d_run >= 3), C, D), (coop_seen, d_run)


tricky_defector = Bot("Tricky Defector", lambda p, k: (jnp.bool_(False), jnp.int32(0)), _tricky_defector_act,
                      memory_depth=INF)

STRATEGIES = [tricky_defector]
