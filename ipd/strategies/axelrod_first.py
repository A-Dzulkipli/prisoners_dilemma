"""Strategies from Axelrod's first tournament (1980), ported from axelrod's axelrod_first.py."""
import jax
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob, memory_one, no_state
from ipd.game import C, D

INF = float("inf")
# chi2.ppf(0.95, df=1): a two-bin chi-squared test has p >= 0.05 iff the statistic is at most this
CHI2_CRIT = 3.841458820694124


def _looks_random(n_coop, n):
    """Library's chisquare([coops, defections]).pvalue >= 0.05, with statistic (coops - defs)^2 / n."""
    diff = (2 * n_coop - n).astype(jnp.float32)
    return diff * diff <= CHI2_CRIT * n


def _last_two_defect(o, action):
    """FinalTransformer((D, D)): defect on the last two turns when the length is known."""
    return jnp.where((o.turns > 0) & (o.t >= o.turns - 2), D, action)


def _anonymous_act(p, s, o, k):
    k1, k2 = jax.random.split(k)
    return _coop_with_prob(k2, jax.random.uniform(k1, (), minval=0.3, maxval=0.7)), s


anonymous = Bot("First by Anonymous", no_state, _anonymous_act, memory_depth=0, stochastic=True)


def _davis_act(p, wronged, o, k):
    wronged = wronged | (o.their_last == D)
    return jnp.where((o.t >= 10) & wronged, D, C), wronged


davis = Bot("First by Davis", lambda p, k: jnp.bool_(False), _davis_act, memory_depth=INF)


def _downing_init(p, k):
    # opponent C after my C, opponent C after my D, my cooperations, my defections, my move at t-2
    return jnp.zeros(5, jnp.int32)


def _downing_act(p, s, o, k):
    n_cc, n_cd, coops, defs, my_prev2 = s
    coops = coops + (o.my_last == C)
    defs = defs + (o.my_last == D)
    # my_prev2 starts as a phantom C, so the opponent's first move counts as a reply to cooperation
    n_cc = n_cc + ((my_prev2 == C) & (o.their_last == C))
    n_cd = n_cd + ((my_prev2 == D) & (o.their_last == C) & (o.t >= 2))
    (R, S), (T, P) = o.my_payoff
    # E_C - E_D with alpha = n_cc / (coops + 1), beta = n_cd / max(defs, 2), cleared of denominators
    a, b = coops + 1, jnp.maximum(defs, 2)
    diff = (S - P) * a * b + n_cc * (R - S) * b - n_cd * (T - P) * a
    action = jnp.select([o.t < 2, diff > 0, diff < 0], [D, C, D], 1 - o.my_last)
    my_prev2 = jnp.where(o.t == 0, C, o.my_last)
    return action, jnp.stack([n_cc, n_cd, coops, defs, my_prev2]).astype(jnp.int32)


downing = Bot("First by Downing", _downing_init, _downing_act, memory_depth=INF)


def _feld_act(p, s, o, k):
    p_coop = jnp.maximum(1.0 - 0.5 / 200 * o.t, 0.5)
    action = jnp.where(o.t == 0, C, jnp.where(o.their_last == D, D, _coop_with_prob(k, p_coop)))
    return action, s


feld = Bot("First by Feld", no_state, _feld_act, memory_depth=200, stochastic=True)


def _graaskamp_init(p, k):
    # opponent cooperations, my move at t-1 before last, opponent is TFT so far, is my clone so far,
    # flagged random, next random defection turn (-1 = unset)
    return (jnp.int32(0), jnp.int32(-1), jnp.bool_(True), jnp.bool_(True), jnp.bool_(False), jnp.int32(-1))


def _graaskamp_act(p, s, o, k):
    n_coop, my_prev2, is_tft, is_clone, is_random, next_d = s
    n_coop = n_coop + (o.their_last == C)
    is_tft = is_tft & ((o.t < 2) | (o.their_last == my_prev2))
    is_clone = is_clone & (o.my_last == o.their_last)
    late = o.t >= 56
    is_random = is_random | (late & _looks_random(n_coop, o.t))
    tft_move = jnp.where(o.their_last == D, D, C)
    # random defections every 5-15 moves against anything else
    roam = late & ~is_random & ~(is_tft | is_clone)
    next_d = jnp.where(roam & (next_d < 0), o.t + jax.random.randint(k, (), 5, 15), next_d)
    k2 = jax.random.fold_in(k, 1)
    hit = roam & (o.t == next_d)
    next_d = jnp.where(hit, o.t + jax.random.randint(k2, (), 5, 15), next_d)
    action = jnp.select(
        [o.t == 0, ~late, is_random, is_tft | is_clone],
        [C, jnp.where(o.t == 50, D, tft_move), D, tft_move],
        jnp.where(hit, D, C),
    )
    return action, (n_coop, o.my_last, is_tft, is_clone, is_random, next_d.astype(jnp.int32))


graaskamp = Bot("First by Graaskamp", _graaskamp_init, _graaskamp_act, memory_depth=INF, stochastic=True)


def _grofman_act(p, s, o, k):
    return jnp.where((o.t == 0) | (o.my_last == o.their_last), C, _coop_with_prob(k, 2 / 7)), s


grofman = Bot("First by Grofman", no_state, _grofman_act, memory_depth=1, stochastic=True)

joss = memory_one([0.9, 0.0, 0.9, 0.0], name="First by Joss")

NYDEGGER_DEFECT = jnp.zeros(64, jnp.bool_).at[
    jnp.array([1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61])
].set(True)


def _nydegger_act(p, s, o, k):
    A, their_first = s
    # outcome code is my D + 2 * their D; the library weights the most recent round by 16
    A = A // 4 + 16 * jnp.maximum(o.my_last + 2 * o.their_last, 0)
    their_first = jnp.where(o.t == 1, o.their_last, their_first)
    tft = jnp.where(o.their_last == D, D, C)
    action = jnp.select(
        [o.t == 0, o.t == 1, o.t == 2],
        [C, tft, jnp.where((their_first == D) & (o.their_last == C), D, tft)],
        jnp.where(NYDEGGER_DEFECT[A], D, C),
    )
    return action, (A, their_first)


nydegger = Bot("First by Nydegger", lambda p, k: (jnp.int32(0), jnp.int32(-1)), _nydegger_act, memory_depth=3)


def _shubik_act(p, s, o, k):
    remaining, length = s
    start = (remaining == 0) & (o.their_last == D) & (o.my_last == C)
    length = length + start
    remaining = jnp.where(start, length, remaining)
    action = jnp.where(remaining > 0, D, C)
    return action, (jnp.maximum(remaining - 1, 0), length)


shubik = Bot("First by Shubik", lambda p, k: (jnp.int32(0), jnp.int32(0)), _shubik_act, memory_depth=INF)


def _stein_rapoport_act(p, s, o, k):
    n_coop, is_random = s
    n_coop = n_coop + (o.their_last == C)
    # every 15 rounds (round number t + 1) re-test whether the opponent looks random
    is_random = jnp.where((o.t >= 14) & ((o.t + 1) % 15 == 0), _looks_random(n_coop, o.t), is_random)
    action = jnp.select([o.t < 4, o.t < 14, is_random], [C, o.their_last, D], o.their_last)
    return _last_two_defect(o, action), (n_coop, is_random)


stein_and_rapoport = Bot(
    "First by Stein and Rapoport",
    lambda p, k: (jnp.int32(0), jnp.bool_(False)),
    _stein_rapoport_act,
    memory_depth=INF,
)


def _tc_init(p, k):
    # retaliation remaining, retaliation length, remembered opponent defections,
    # my score, their score, round of last fresh start, second fresh-start cooperation pending
    return (jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.float32(0), jnp.float32(0), jnp.int32(0), jnp.bool_(False))


def _tc_act(p, s, o, k):
    remaining, length, n_def, my_score, their_score, last_fresh, pending = s
    first = o.t == 0
    my_last, their_last = jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)
    n_def = n_def + (o.their_last == D)
    # the library scores the opponent with the same (symmetric) game
    my_score = my_score + jnp.where(first, 0.0, o.my_payoff[my_last, their_last])
    their_score = their_score + jnp.where(first, 0.0, o.my_payoff[their_last, my_last])
    rnd = o.t + 1
    # fresh start if well ahead, 20+ rounds since the last one, 10+ rounds left, opponent just
    # cooperated, and remembered defections are 3 sd from N/2: |N - 2r| >= 3 sqrt(N) with N = t
    n_minus_2r = o.t - 2 * n_def
    fresh = (
        ~first
        & ~pending
        & ((last_fresh == 0) | (rnd - last_fresh >= 20))
        & (my_score - their_score >= 10)
        & ((o.turns < 0) | (o.turns - rnd >= 10))
        & (o.their_last == C)
        & (n_minus_2r * n_minus_2r >= 9 * o.t)
    )
    remaining = jnp.where(fresh, 0, remaining)
    length = jnp.where(fresh, 0, length)
    n_def = jnp.where(fresh, 0, n_def)
    last_fresh = jnp.where(fresh, rnd, last_fresh)
    # retaliate for one more round per run of opponent defections
    play = ~first & ~pending & ~fresh
    start = play & (remaining == 0) & (o.their_last == D)
    length = length + start
    remaining = jnp.where(start, length, remaining)
    action = jnp.where(play & (remaining > 0), D, C)
    remaining = jnp.where(play, jnp.maximum(remaining - 1, 0), remaining)
    new_s = (remaining, length, n_def, my_score, their_score, last_fresh, fresh)
    return _last_two_defect(o, action), new_s


tideman_and_chieruzzi = Bot("First by Tideman and Chieruzzi", _tc_init, _tc_act, memory_depth=INF)


def _tullock_act(p, window, o, k):
    # window holds the opponent's last 10 moves, -1 for rounds not yet played
    window = jnp.roll(window, -1).at[-1].set(o.their_last)
    p_coop = jnp.maximum(0.0, jnp.sum(window == C) / 10 - 0.1)
    return jnp.where(o.t < 11, C, _coop_with_prob(k, p_coop)), window


tullock = Bot(
    "First by Tullock",
    lambda p, k: jnp.full(10, -1, jnp.int32),
    _tullock_act,
    memory_depth=INF,
    stochastic=True,
)

STRATEGIES = [
    anonymous,
    davis,
    downing,
    feld,
    graaskamp,
    grofman,
    joss,
    nydegger,
    shubik,
    stein_and_rapoport,
    tideman_and_chieruzzi,
    tullock,
]
