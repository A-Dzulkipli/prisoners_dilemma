"""Ratio-targeting strategies (golden mean, pi, e), ported from axelrod's mathematicalconstants.py."""
import math

import jax.numpy as jnp
import numpy as np

from ipd.bots import Bot
from ipd.game import C, D


def _ratio_act(p, s, o, k):
    coops, defs, their_defs = s
    coops = coops + (o.my_last == C) + (o.their_last == C)
    defs = defs + (o.my_last == D) + (o.their_last == D)
    their_defs = their_defs + (o.their_last == D)
    # cooperations / defections > ratio, exactly: the ratio is irrational, so this is coops > floor(ratio * defs)
    above = coops > p[jnp.minimum(defs, p.shape[0] - 1)]
    action = jnp.select([o.t == 0, their_defs == 0, above], [C, D, D], C)
    return action, (coops.astype(jnp.int32), defs.astype(jnp.int32), their_defs.astype(jnp.int32))


def coto_de_ratio(name, ratio, max_turns=1000):
    """Steer the joint cooperations / defections ratio towards `ratio`; thresholds tabled up to max_turns rounds."""
    floors = jnp.asarray(np.floor(ratio * np.arange(2 * max_turns + 1)), jnp.int32)
    return Bot(name, lambda p, k: (jnp.int32(0), jnp.int32(0), jnp.int32(0)), _ratio_act, params=floors, memory_depth=float("inf"))


STRATEGIES = [
    coto_de_ratio("$\\phi$", (1 + math.sqrt(5)) / 2),
    coto_de_ratio("$\\pi$", math.pi),
    coto_de_ratio("$e$", math.e),
]
