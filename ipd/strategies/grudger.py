"""Grudger variants."""
import jax.numpy as jnp
import numpy as np

from ipd.bots import INF, Bot, grudger_bot
from ipd.game import C, D


aggravater = grudger_bot("Aggravater", D, 3)
spiteful_cc = grudger_bot("SpitefulCC", C, 2)


def _flag_init(p, k):
    return jnp.bool_(False)


def _forgetful_grudger_act(p, s, o, k):
    grudged, memory = s
    reset = memory == 10
    grudged = (grudged & ~reset) | (o.their_last == D)
    memory = jnp.where(reset, 0, memory) + grudged
    return jnp.where(grudged, D, C), (grudged, memory)


forgetful_grudger = Bot(
    "Forgetful Grudger", lambda p, k: (jnp.bool_(False), jnp.int32(0)), _forgetful_grudger_act, memory_depth=10
)


def _opposite_grudger_act(p, seen_c, o, k):
    seen_c = seen_c | (o.their_last == C)
    return jnp.where(seen_c, C, D), seen_c


opposite_grudger = Bot("Opposite Grudger", _flag_init, _opposite_grudger_act, memory_depth=INF)


def _soft_grudger_act(p, s, o, k):
    # punish n consecutive opponent defections with d defections, then c cooperations
    grudged, memory, run = s
    run = jnp.where(o.their_last == D, run + 1, 0)
    action = jnp.where(grudged, jnp.where(memory < p["d"] - 1, D, C), jnp.where(run >= p["n"], D, C))
    memory = jnp.where(grudged, memory + 1, 0)
    done = memory == p["d"] - 1 + p["c"]
    grudged = jnp.where(grudged, ~done, run >= p["n"])
    memory = jnp.where(done, 0, memory)
    return action, (grudged, memory, run)


def general_soft_grudger(n=1, d=4, c=2, name="General Soft Grudger", memory_depth=INF):
    """After n consecutive opponent defections, play d defections then c cooperations."""
    params = {"n": jnp.int32(n), "d": jnp.int32(d), "c": jnp.int32(c)}
    init = lambda p, k: (jnp.bool_(False), jnp.int32(0), jnp.int32(0))
    return Bot(name, init, _soft_grudger_act, params, memory_depth=memory_depth)


def _grudger_alternator_act(p, seen_d, o, k):
    seen_d = seen_d | (o.their_last == D)
    return jnp.where(seen_d & (o.my_last == C), D, C), seen_d


grudger_alternator = Bot("GrudgerAlternator", _flag_init, _grudger_alternator_act, memory_depth=INF)


def _easy_go_act(p, seen_d, o, k):
    seen_d = seen_d | (o.their_last == D)
    return jnp.where(seen_d, C, D), seen_d


easy_go = Bot("EasyGo", _flag_init, _easy_go_act, memory_depth=INF)


def _capri_table():
    """Cooperate-table over the last three rounds, oldest first, each round coded 2 * mine + theirs."""
    cc, cd, dc, dd = (0, 0), (0, 1), (1, 0), (1, 1)
    coop = [
        (cc, cc, cc), (cc, cc, dc), (cc, dc, cd), (dc, cd, cc), (cd, cc, cc), (cc, cd, dc), (cd, dc, cc),
        (dc, cc, cc), (dd, dd, dc), (dd, dc, cc), (dd, dd, cd), (dd, cd, cc), (dd, dd, cc), (dd, cc, cc),
    ]
    table = np.full(64, D, np.int32)
    for h in coop:
        a, b, c = (2 * m + t for m, t in h)
        table[16 * a + 4 * b + c] = C
    return jnp.asarray(table)


CAPRI_TABLE = _capri_table()


def _capri_act(p, h, o, k):
    # h codes the last three rounds; the empty history counts as mutual cooperation
    h = jnp.where(o.t == 0, h, (4 * h + 2 * o.my_last + o.their_last) % 64)
    return CAPRI_TABLE[h], h


capri = Bot("CAPRI", lambda p, k: jnp.int32(0), _capri_act, memory_depth=3)

STRATEGIES = [
    aggravater,
    capri,
    easy_go,
    forgetful_grudger,
    general_soft_grudger(),
    grudger_alternator,
    opposite_grudger,
    general_soft_grudger(name="Soft Grudger", memory_depth=6),
    spiteful_cc,
]
