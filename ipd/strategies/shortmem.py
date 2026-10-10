"""ShortMem, ported from axelrod's shortmem.py."""
import jax.numpy as jnp
from jax import lax

from ipd.bots import Bot
from ipd.game import C, D


def _shortmem_act(p, last10, o, k):
    # opponent's last ten moves as bits, D = 1
    last10 = ((last10 << 1) | jnp.maximum(o.their_last, 0)) & 0x3FF
    diff = 10 - 2 * lax.population_count(last10)  # C count - D count
    action = jnp.select([o.t <= 10, diff >= 3, diff <= -3], [C, C, D], o.their_last)
    return action, last10


shortmem = Bot("ShortMem", lambda p, k: jnp.int32(0), _shortmem_act, memory_depth=float("inf"))

STRATEGIES = [shortmem]
