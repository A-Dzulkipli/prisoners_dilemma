"""Gradual Killer."""
import jax.numpy as jnp

from ipd.bots import INF, Bot
from ipd.game import C, D


def _gradual_killer_act(p, n_late_def, o, k):
    # opens D, D, D, D, D, C, C (an InitialTransformer), then defects forever if the opponent's
    # 6th and 7th moves were both D
    n_late_def = n_late_def + ((o.their_last == D) & ((o.t == 6) | (o.t == 7)))
    action = jnp.where(o.t < 7, jnp.where(o.t < 5, D, C), jnp.where(n_late_def == 2, D, C))
    return action, n_late_def


gradual_killer = Bot("Gradual Killer", lambda p, k: jnp.int32(0), _gradual_killer_act, memory_depth=INF)

STRATEGIES = [gradual_killer]
