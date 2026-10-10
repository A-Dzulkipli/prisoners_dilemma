"""Tricky Cooperator, ported from axelrod's cooperator.py (Cooperator itself is in ipd.bots)."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D


def _tricky_cooperator_act(p, last_d, o, k):
    # last_d: round of the opponent's latest defection; defect after round 3 if none in the last 10
    last_d = jnp.where(o.their_last == D, o.t - 1, last_d)
    return jnp.where((o.t >= 3) & (last_d < o.t - 10), D, C), last_d


tricky_cooperator = Bot("Tricky Cooperator", lambda p, k: jnp.int32(-100), _tricky_cooperator_act, memory_depth=10)

STRATEGIES = [tricky_cooperator]
