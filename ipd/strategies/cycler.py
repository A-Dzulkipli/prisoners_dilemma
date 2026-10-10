"""Cyclers, ported from axelrod's cycler.py."""
import jax.numpy as jnp

from ipd.bots import Bot, no_state
from ipd.game import C, D

INF = float("inf")


def _cycler_act(p, s, o, k):
    return p[o.t % p.shape[0]], s


def cycler(cycle="CCD"):
    """Repeat `cycle` (a string of C and D) forever."""
    params = jnp.array([C if a == "C" else D for a in cycle], jnp.int32)
    return Bot(f"Cycler {cycle}", no_state, _cycler_act, params=params, memory_depth=len(cycle) - 1)


def _anti_cycler_act(p, s, o, k):
    # after C D D, play runs of n C's followed by one D for n = 1, 2, 3, ...
    length, count = s
    play_c = count < length
    action = jnp.where(o.t < 3, jnp.array([C, D, D])[jnp.minimum(o.t, 2)], jnp.where(play_c, C, D))
    late = o.t >= 3
    length = jnp.where(late & ~play_c, length + 1, length)
    count = jnp.where(late, jnp.where(play_c, count + 1, 0), count)
    return action, (length.astype(jnp.int32), count.astype(jnp.int32))


anti_cycler = Bot("AntiCycler", lambda p, k: (jnp.int32(1), jnp.int32(0)), _anti_cycler_act, memory_depth=INF)

STRATEGIES = [anti_cycler] + [cycler(c) for c in ["CCCCCD", "CCCD", "CCCDCD", "CCD", "DC", "DDC"]]
