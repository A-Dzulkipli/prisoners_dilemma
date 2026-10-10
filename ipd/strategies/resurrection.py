"""Resurrection and DoubleResurrection."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D


def _runs(s, o):
    # lengths of my current runs of C and of D
    c_run, d_run = s
    return jnp.where(o.my_last == C, c_run + 1, 0), jnp.where(o.my_last == D, d_run + 1, 0)


def _resurrection_act(p, s, o, k):
    c_run, d_run = _runs(s, o)
    return jnp.where(o.t == 0, C, jnp.where(d_run >= 5, D, o.their_last)), (c_run, d_run)


def _double_resurrection_act(p, s, o, k):
    c_run, d_run = _runs(s, o)
    return jnp.select([o.t == 0, c_run >= 5, d_run >= 5], [C, D, C], o.their_last), (c_run, d_run)


def _init(p, k):
    return jnp.int32(0), jnp.int32(0)


resurrection = Bot("Resurrection", _init, _resurrection_act, memory_depth=5)
double_resurrection = Bot("DoubleResurrection", _init, _double_resurrection_act, memory_depth=5)

STRATEGIES = [double_resurrection, resurrection]
