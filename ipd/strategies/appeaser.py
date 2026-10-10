"""Appeaser, ported from axelrod's appeaser.py."""
import jax.numpy as jnp

from ipd.bots import Bot, no_state
from ipd.game import C, D


def _appeaser_act(p, s, o, k):
    # switch my move whenever the opponent defects
    return jnp.select([o.t == 0, o.their_last == D], [C, 1 - o.my_last], o.my_last), s


appeaser = Bot("Appeaser", no_state, _appeaser_act, memory_depth=float("inf"))

STRATEGIES = [appeaser]
