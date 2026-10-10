"""SelfSteem."""
import math

import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob, no_state
from ipd.game import C, D

RANDOM, TFT, DEFECT, COOP = range(4)


def _mode(n):
    f = math.sin(2 * math.pi * n / 10)
    if f > 0.95:
        return DEFECT
    if 0.95 > abs(f) > 0.3:
        return TFT
    if 0.3 > f > -0.3:
        return RANDOM
    return COOP


# the sine depends on the round only through t mod 10
MODE = jnp.array([_mode(n) for n in range(10)], jnp.int32)


def _selfsteem_act(p, s, o, k):
    action = jnp.array([coop_with_prob(k, 0.5), o.their_last, D, C])[MODE[o.t % 10]]
    return action, s


selfsteem = Bot("SelfSteem", no_state, _selfsteem_act, memory_depth=INF, stochastic=True)

STRATEGIES = [selfsteem]
