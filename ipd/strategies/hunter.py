"""Hunters: bots that defect once they recognise a type of opponent.

The library's detect_cycle (periods up to 12) is tracked incrementally: the first 12 moves plus one
"still periodic" flag per period.
"""
import jax.numpy as jnp
import numpy as np

from ipd.bots import INF, Bot
from ipd.game import C, D

PERIODS = jnp.arange(13)
MAX_CYCLE = 12


def cycle_init():
    # first 12 moves of the watched sequence, is_periodic[i] for period i (index 0 unused)
    return jnp.full(MAX_CYCLE, -1, jnp.int32), jnp.ones(13, jnp.bool_)


def cycle_update(first, periodic, j, x, live=True):
    """Append move x at position j of the sequence (no-op when not live)."""
    first = jnp.where(live & (j < MAX_CYCLE), first.at[jnp.minimum(j, MAX_CYCLE - 1)].set(x), first)
    periodic = periodic & (~live | (x == first[j % jnp.maximum(PERIODS, 1)]))
    return first, periodic


def detect_cycle(first, periodic, length, min_size):
    """detect_cycle(history, min_size): (found, smallest period, cycle has both C and D)."""
    valid = periodic & (PERIODS >= min_size) & (PERIODS <= jnp.minimum(length // 2, MAX_CYCLE))
    size = jnp.argmax(valid)
    n_d = jnp.sum((first == D) & (jnp.arange(MAX_CYCLE) < size))
    return valid.any(), size, (n_d > 0) & (n_d < size)


def _defector_hunter_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    return jnp.where((o.t >= 4) & (n_def == o.t), D, C), n_def


def _cooperator_hunter_act(p, n_coop, o, k):
    n_coop = n_coop + (o.their_last == C)
    return jnp.where((o.t >= 4) & (n_coop == o.t), D, C), n_coop


defector_hunter = Bot("Defector Hunter", lambda p, k: jnp.int32(0), _defector_hunter_act, memory_depth=INF)
cooperator_hunter = Bot("Cooperator Hunter", lambda p, k: jnp.int32(0), _cooperator_hunter_act, memory_depth=INF)


def _alternator_hunter_act(p, s, o, k):
    # is_alt is decided on the opponent's first six moves and then fixed
    is_alt, their_prev = s
    is_alt = is_alt & ((o.t < 2) | (o.t > 6) | (o.their_last != their_prev))
    return jnp.where((o.t >= 6) & is_alt, D, C), (is_alt, o.their_last)


alternator_hunter = Bot("Alternator Hunter", lambda p, k: (jnp.bool_(True), jnp.int32(-1)), _alternator_hunter_act,
                        memory_depth=INF)


def _cycle_hunter_act(p, s, o, k):
    first, periodic, hunting = s
    first, periodic = cycle_update(first, periodic, o.t - 1, o.their_last, o.t > 0)
    found, _, mixed = detect_cycle(first, periodic, o.t, 3)
    hunting = hunting | (found & mixed)
    return jnp.where(hunting, D, C), (first, periodic, hunting)


cycle_hunter = Bot("Cycle Hunter", lambda p, k: cycle_init() + (jnp.bool_(False),), _cycle_hunter_act,
                   memory_depth=INF)


def _eventual_cycle_hunter_act(p, s, o, k):
    # cycles are looked for in the opponent's moves from index 10 on, re-checked every 10 rounds;
    # unlike Cycle Hunter, constant cycles count
    first, periodic, n_coop, cycle = s
    first, periodic = cycle_update(first, periodic, o.t - 11, o.their_last, o.t > 10)
    n_coop = n_coop + (o.their_last == C)
    found, _, _ = detect_cycle(first, periodic, o.t - 10, 3)
    check = (o.t >= 10) & (n_coop < o.t) & (o.t % 10 == 0)
    cycle = jnp.where(check, found, cycle)
    action = jnp.where((o.t >= 10) & (n_coop < o.t) & cycle, D, C)
    return action, (first, periodic, n_coop, cycle)


eventual_cycle_hunter = Bot(
    "Eventual Cycle Hunter",
    lambda p, k: cycle_init() + (jnp.int32(0), jnp.bool_(False)),
    _eventual_cycle_hunter_act,
    memory_depth=INF,
)


def _windows(n):
    """Lengths of the library's three windows: [0, n/2), [n/4, 3n/4), [n/2, n)."""
    return n // 2, 3 * n // 4 - n // 4, n - n // 2


def _math_constant_tie_table(max_turns):
    """How the library's float64 settles |0.5 a / la - 0.5 b / lb| < 0.2 when the gap is exactly 0.2.

    Indexed [n, a] (a = count in the first window), bit 2 * pair + (first ratio larger); pair 0 is
    windows 1 and 2, pair 1 is windows 1 and 3. The partner count b is fixed by the tie.
    """
    bits = np.zeros((max_turns + 1, max_turns + 2), np.int8)
    for n in range(8, max_turns + 1):
        l1, l2, l3 = _windows(n)
        a = np.arange(2 * l1 + 1)
        for pair, lb in enumerate([l2, l3]):
            for sign in (0, 1):  # sign 1: a / l1 is the larger ratio
                num = 5 * a * lb - (2 * sign - 1) * 2 * l1 * lb
                b = num // (5 * l1)
                ok = (num % (5 * l1) == 0) & (b >= 0) & (b <= 2 * lb)
                close = np.abs(0.5 * a / l1 - 0.5 * b / lb) < 0.2
                bits[n, a] |= ((ok & close) << (2 * pair + sign)).astype(np.int8)
    return jnp.asarray(bits)


def _math_constant_hunter_act(ties, s, o, k):
    # prefix[i]: cooperations by both players in rounds before i (rounds past max_turns are dropped)
    prefix, n_coop_opp = s
    n = o.t
    both = (o.my_last == C).astype(jnp.int32) + (o.their_last == C)
    prefix = prefix.at[n].set(jnp.where(n > 0, prefix[n - 1] + both, 0))
    n_coop_opp = n_coop_opp + (o.their_last == C)
    l1, l2, l3 = (jnp.maximum(x, 1) for x in _windows(n))
    c1 = prefix[n // 2]
    c2 = prefix[3 * n // 4] - prefix[n // 4]
    c3 = prefix[n] - prefix[n // 2]
    tie_bits = ties[jnp.minimum(n, ties.shape[0] - 1), c1]

    def close(pair, c, l):
        # |0.5 c1 / l1 - 0.5 c / l| < 0.2, cleared of denominators; exact ties as float64 has them
        gap = c1 * l - c * l1
        bit = (tie_bits >> (2 * pair + (gap > 0))) & 1
        return jnp.where(5 * jnp.abs(gap) == 2 * l1 * l, bit == 1, 5 * jnp.abs(gap) < 2 * l1 * l)

    hunt = (n >= 8) & (n_coop_opp > 0) & (n_coop_opp < n) & close(0, c2, l2) & close(1, c3, l3)
    return jnp.where(hunt, D, C), (prefix, n_coop_opp)


def math_constant_hunter(max_turns=1000):
    """Compares cooperation over windows of the history; the prefix counts and float64 tie table cover max_turns."""
    return Bot("Math Constant Hunter", lambda p, k: (jnp.zeros(max_turns + 1, jnp.int32), jnp.int32(0)),
               _math_constant_hunter_act, _math_constant_tie_table(max_turns), memory_depth=INF)


def _random_hunter_act(p, s, o, k):
    count_cc, count_dd, my_coops, my_prev2 = s
    count_cc = count_cc + ((o.t > 1) & (my_prev2 == C) & (o.their_last == C))
    count_dd = count_dd + ((o.t > 1) & (my_prev2 == D) & (o.their_last == D))
    my_coops = my_coops + (o.my_last == C)
    my_defs = o.t - my_coops
    # |count / n - 0.5| < 0.25 for each of my C and D counts above 5
    use_cc, use_dd = my_coops > 5, my_defs > 5
    ok_cc = ~use_cc | (jnp.abs(4 * count_cc - 2 * my_coops) < my_coops)
    ok_dd = ~use_dd | (jnp.abs(4 * count_dd - 2 * my_defs) < my_defs)
    hunt = (o.t > 10) & (use_cc | use_dd) & ok_cc & ok_dd
    return jnp.where(hunt, D, C), (count_cc, count_dd, my_coops, o.my_last)


random_hunter = Bot("Random Hunter", lambda p, k: (jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(-1)),
                    _random_hunter_act, memory_depth=INF)

STRATEGIES = [
    alternator_hunter,
    cooperator_hunter,
    cycle_hunter,
    defector_hunter,
    eventual_cycle_hunter,
    math_constant_hunter(),
    random_hunter,
]
