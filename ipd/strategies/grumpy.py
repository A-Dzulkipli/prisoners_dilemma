"""Grumpy, ported from axelrod's grumpy.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D


def _grumpy_act(p, s, o, k):
    grumpiness, grumpy = s
    grumpiness = grumpiness + jnp.where(o.t > 0, 2 * o.their_last - 1, 0)  # defections - cooperations
    grumpy = jnp.where(grumpy, grumpiness >= p["nice_threshold"], grumpiness > p["grumpy_threshold"])
    return jnp.where(grumpy, D, C), (grumpiness, grumpy)


def grumpy_bot(starting_state="Nice", grumpy_threshold=10, nice_threshold=-10):
    params = {"grumpy_threshold": jnp.int32(grumpy_threshold), "nice_threshold": jnp.int32(nice_threshold)}
    init_grumpy = starting_state == "Grumpy"
    return Bot("Grumpy", lambda p, k: (jnp.int32(0), jnp.bool_(init_grumpy)), _grumpy_act, params,
               memory_depth=float("inf"))


STRATEGIES = [grumpy_bot()]
