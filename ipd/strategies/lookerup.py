"""Lookup table strategies, ported from axelrod's lookerup.py.

Every table is expanded to the largest shape used, depth 2 for my last moves, their last moves
and their first moves, and holds cooperation probabilities. So one act serves LookerUp (0/1
entries) and Gambler (gambler.py), and all tables share one params shape for vmapping.
"""
import itertools

import jax.numpy as jnp
import numpy as np

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C, D


def _lookup_init(p, k):
    # my move at t-2, their move at t-2, their first two moves
    return jnp.zeros(4, jnp.int32)


def _lookup_act(p, s, o, k):
    my_prev2, their_prev2, first0, first1 = s
    mine = 2 * my_prev2 + jnp.maximum(o.my_last, 0)
    theirs = 2 * their_prev2 + jnp.maximum(o.their_last, 0)
    last = jnp.maximum(o.their_last, 0)
    first0 = jnp.where(o.t == 1, last, first0)
    first1 = jnp.where(o.t == 2, last, first1)
    opening = 2 * first0 + first1
    p_coop = jnp.where(o.t < p["n_init"], p["init"][jnp.minimum(o.t, 1)], p["table"][mine, theirs, opening])
    s = jnp.stack([jnp.maximum(o.my_last, 0), last, first0, first1]).astype(jnp.int32)
    return _coop_with_prob(k, p_coop), s


def lookup_params(pattern, depths, initial=()):
    """pattern: entries in axelrod key order (C < D, my moves, their moves, their openings, oldest
    first), as 'C'/'D' or cooperation probabilities. initial: opening moves, padded with C."""
    ds, do, dopen = depths
    pattern = [{"C": 1.0, "D": 0.0}.get(x, x) for x in pattern]
    table = np.zeros((4, 4, 4), np.float32)
    for s1, s0, o1, o0, f0, f1 in itertools.product((0, 1), repeat=6):
        bits = (s1, s0)[2 - ds:] + (o1, o0)[2 - do:] + (f0, f1)[:dopen]
        table[2 * s1 + s0, 2 * o1 + o0, 2 * f0 + f1] = pattern[int("".join(map(str, bits)) or "0", 2)]
    n_init = max(depths)
    init = [1.0 if a == C else 0.0 for a in (tuple(initial) + (C, C))[:2]]
    return {"table": jnp.asarray(table), "init": jnp.asarray(init, jnp.float32), "n_init": jnp.int32(n_init)}


def lookup(name, pattern, depths, initial=(), stochastic=False, memory_depth=None):
    if memory_depth is None:
        memory_depth = float("inf") if depths[2] else max(depths)
    return Bot(name, _lookup_init, _lookup_act, params=lookup_params(pattern, depths, initial),
               memory_depth=memory_depth, stochastic=stochastic)


evolved_lookerup_1_1_1 = lookup("EvolvedLookerUp1_1_1", "CDDDDCDD", (1, 1, 1), (C,))
evolved_lookerup_2_2_2 = lookup(
    "EvolvedLookerUp2_2_2",
    "CDDCDCDDCDDDCDDDDDCDCDCCCDDCCDCDDDCCCCCDDDCDDDDDDDDDCCDDCDDDCCCD",
    (2, 2, 2),
    (C, C),
)
winner12 = lookup("Winner12", "CDCDDCDD", (1, 2, 0), (C, C))
winner21 = lookup("Winner21", "CDCDCDDD", (1, 2, 0), (D, C))

STRATEGIES = [evolved_lookerup_1_1_1, evolved_lookerup_2_2_2, winner12, winner21]
