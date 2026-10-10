"""BackStabber and DoubleCrosser."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, defect_at_end
from ipd.game import C, D


def _init(p, k):
    # opponent defections in total, opponent defections in their first 7 moves, their move at t-2
    return jnp.int32(0), jnp.int32(0), jnp.int32(-1)


def _act(p, s, o, k):
    # p: True for DoubleCrosser
    n_def, early_def, their_prev = s
    n_def = n_def + (o.their_last == D)
    early_def = early_def + ((o.their_last == D) & (o.t <= 7))
    backstab = jnp.where(n_def > 3, D, C)
    # rounds 8 to 180 against an opponent clean in its first 7 moves: defect only after two Ds
    alt = p & (early_def == 0) & (o.t >= 7) & (o.t < 180)
    action = jnp.where(alt, jnp.where((o.their_last == D) & (their_prev == D), D, C), backstab)
    return defect_at_end(o, action, 2), (n_def, early_def, o.their_last)


backstabber = Bot("BackStabber", _init, _act, jnp.bool_(False), memory_depth=INF)
double_crosser = Bot("DoubleCrosser", _init, _act, jnp.bool_(True), memory_depth=INF)

STRATEGIES = [backstabber, double_crosser]
