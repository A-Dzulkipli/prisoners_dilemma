"""Punisher variants, ported from axelrod's strategies/punisher.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D

INF = float("inf")


def _punisher_init(p, k):
    return jnp.array([1, 0, 1, 0, 0], jnp.int32)  # mem_length, grudged, grudge_memory, their C's, their D's


def _punisher_act(p, s, o, k):
    # p = 1: punish for 20 * their defection rate rounds; p = 0 (inverse): 20 * their cooperation rate, at least 1
    mem_length, grudged, memory, n_coop, n_def = s
    n_coop = n_coop + (o.their_last == C)
    n_def = n_def + (o.their_last == D)
    reset = memory >= mem_length
    grudged = jnp.where(reset, 0, grudged)
    memory = jnp.where(reset, 0, memory)
    start = (grudged == 0) & (o.their_last == D)
    length = 20 * jnp.where(p == 1, n_def, n_coop) // jnp.maximum(o.t, 1)
    length = jnp.where(p == 1, length, jnp.maximum(length, 1))
    mem_length = jnp.where(start, length, mem_length)
    action = jnp.where((grudged == 1) | start, D, C)
    memory = memory + (grudged == 1)
    grudged = grudged | start
    return action, jnp.stack([mem_length, grudged, memory, n_coop, n_def]).astype(jnp.int32)


punisher = Bot("Punisher", _punisher_init, _punisher_act, jnp.int32(1), memory_depth=INF)
inverse_punisher = Bot("Inverse Punisher", _punisher_init, _punisher_act, jnp.int32(0), memory_depth=INF)


def _level_punisher_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    return jnp.where((o.t >= 10) & (5 * n_def > o.t), D, C), n_def.astype(jnp.int32)


level_punisher = Bot("Level Punisher", lambda p, k: jnp.int32(0), _level_punisher_act, memory_depth=INF)


def _tricky_level_punisher_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    n = o.t
    defect = ((n < 10) & (5 * n_def > n)) | ((n < 50) & (10 * n_def > n)) | ((n < 100) & (20 * n_def > n))
    return jnp.where(defect, D, C), n_def.astype(jnp.int32)


tricky_level_punisher = Bot(
    "Tricky Level Punisher", lambda p, k: jnp.int32(0), _tricky_level_punisher_act, memory_depth=INF
)

STRATEGIES = [inverse_punisher, level_punisher, punisher, tricky_level_punisher]
