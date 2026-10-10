"""Revised Downing, ported from axelrod's revised_downing.py."""
import jax.numpy as jnp
import numpy as np

from ipd.bots import Bot
from ipd.game import C, D


def _tie_table(max_turns):
    """How the library's float64 resolves exact ties, per bad = n2 / td (row td, column n2 <= td).

    Bit 0: c < 0 when c = 0 exactly, bit 1: alt < 0 when alt = 0, bit 2: c < alt when c = alt.
    At a tie, good is fixed by bad, and both are correctly rounded quotients, as in the library.
    """
    td, n2 = np.tril_indices(max_turns + 1)
    den = np.maximum(td, 1)
    bad = n2 / den
    bits = np.zeros(len(td), np.int8)
    for bit, good_num, good_den in [(0, den + 4 * n2, 3 * den), (1, den + 5 * n2, 4 * den), (2, den + 3 * n2, 2 * den)]:
        good = good_num / good_den
        c = 6.0 * good - 8.0 * bad - 2
        alt = 4.0 * good - 5.0 * bad - 1
        bits |= (np.array([c < 0, alt < 0, c < alt][bit]) << bit).astype(np.int8)
    return jnp.asarray(bits)


def _init(p, k):
    # opponent C after my C, my Cs counted, opponent C after my D, my Ds counted, my move at t-2
    return jnp.zeros(4, jnp.int32), jnp.int32(C)


def _revised_downing_act(ties, s, o, k):
    (nice1, total_c, nice2, total_d), my_prev2 = s
    seen = o.t >= 2
    after_d = seen & (my_prev2 == D)
    after_c = seen & (my_prev2 == C)
    nice1 = nice1 + (after_c & (o.their_last == C))
    total_c = total_c + after_c
    nice2 = nice2 + (after_d & (o.their_last == C))
    total_d = total_d + after_d
    # good = nice1 / total_c (1 before any count), bad = nice2 / total_d (0 before any count);
    # c and alt below are 6 good - 8 bad - 2 and 4 good - 5 bad - 1 multiplied by gd * bd > 0,
    # and exact ties are settled as the library's float64 rounding settles them
    gn, gd = jnp.where(total_c > 0, nice1, 1), jnp.maximum(total_c, 1)
    bn, bd = jnp.where(total_d > 0, nice2, 0), jnp.maximum(total_d, 1)
    c = 6 * gn * bd - 8 * bn * gd - 2 * gd * bd
    alt = 4 * gn * bd - 5 * bn * gd - gd * bd
    tie = ties[jnp.minimum(total_d * (total_d + 1) // 2 + nice2, ties.shape[0] - 1)]
    c_ok = jnp.where(c == 0, (tie & 1) == 0, c >= 0)
    alt_ok = jnp.where(alt == 0, (tie & 2) == 0, alt >= 0)
    c_ge_alt = jnp.where(c == alt, (tie & 4) == 0, c >= alt)
    action = jnp.select([o.t == 0, c_ok & c_ge_alt, c_ok | alt_ok], [C, C, 1 - o.my_last], D)
    new_s = (jnp.stack([nice1, total_c, nice2, total_d]).astype(jnp.int32), jnp.maximum(o.my_last, 0))
    return action, new_s


def revised_downing(max_turns=1000):
    """The float64 tie table covers up to max_turns of my defections."""
    return Bot("Revised Downing", _init, _revised_downing_act, _tie_table(max_turns), memory_depth=float("inf"))


STRATEGIES = [revised_downing()]
