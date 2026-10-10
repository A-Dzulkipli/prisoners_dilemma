"""Tit For Tat variants, ported from axelrod's strategies/titfortat.py."""
import jax
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob, memory_one, no_state
from ipd.game import C, D

INF = float("inf")


def _final_defect(o, action):
    """FinalTransformer((D,)): defect on the last turn when the length is known."""
    return jnp.where((o.turns > 0) & (o.t == o.turns - 1), D, action)


def _tft(o):
    return jnp.where(o.their_last == D, D, C)


def _prev_init(p, k):
    # opponent's moves at t-2 and t-3 (-1 before they exist)
    return jnp.full(2, -1, jnp.int32)


def _shift(prev, o):
    return jnp.stack([o.their_last, prev[0]]).astype(jnp.int32)


def _two_tits_act(p, prev, o, k):
    return jnp.where((o.their_last == D) | (prev[0] == D), D, C), _shift(prev, o)


two_tits_for_tat = Bot("Two Tits For Tat", _prev_init, _two_tits_act, memory_depth=2)


def _hard_tft_act(p, prev, o, k):
    return jnp.where((o.their_last == D) | jnp.any(prev == D), D, C), _shift(prev, o)


hard_tit_for_tat = Bot("Hard Tit For Tat", _prev_init, _hard_tft_act, memory_depth=3)


def _hard_tf2t_act(p, prev, o, k):
    dd = ((o.their_last == D) & (prev[0] == D)) | ((prev[0] == D) & (prev[1] == D))
    return jnp.where(dd, D, C), _shift(prev, o)


hard_tit_for_2_tats = Bot("Hard Tit For 2 Tats", _prev_init, _hard_tf2t_act, memory_depth=3)


def _dynamic_2tft_init(p, k):
    return (jnp.int32(0), jnp.int32(-1))  # opponent cooperations, their move at t-2


def _dynamic_2tft_act(p, s, o, k):
    n_coop, prev = s
    n_coop = n_coop + (o.their_last == C)
    punish = (o.their_last == D) | (prev == D)
    p_coop = n_coop / jnp.maximum(o.t, 1)
    action = jnp.where((o.t > 0) & punish, _coop_with_prob(k, p_coop), C)
    return action, (n_coop.astype(jnp.int32), o.their_last)


dynamic_two_tits_for_tat = Bot(
    "Dynamic Two Tits For Tat", _dynamic_2tft_init, _dynamic_2tft_act, memory_depth=INF, stochastic=True
)

bully = Bot("Bully", no_state, lambda p, s, o, k: (jnp.where(o.their_last == D, C, D), s))

anti_tit_for_tat = Bot("Anti Tit For Tat", no_state, lambda p, s, o, k: (jnp.where(o.their_last == C, D, C), s))


def _sneaky_act(p, s, o, k):
    ever_d, my_prev = s  # opponent ever defected, my move at t-2
    ever_d = ever_d | (o.their_last == D)
    action = jnp.select(
        [o.t < 2, ~ever_d, (o.their_last == D) & (my_prev == D)], [C, D, C], o.their_last
    )
    return action, (ever_d, o.my_last)


sneaky_tit_for_tat = Bot(
    "Sneaky Tit For Tat", lambda p, k: (jnp.bool_(False), jnp.int32(-1)), _sneaky_act, memory_depth=INF
)


def _omega_init(p, k):
    return jnp.array([0, 0, -1], jnp.int32)  # randomness counter, deadlock counter, their move at t-2


def _omega_act(p, s, o, k):
    rand, dead, prev = s
    changed = prev != o.their_last
    deadlocked = dead >= 3
    rand2 = rand - ((prev == C) & (o.their_last == C)) + changed + (o.my_last != o.their_last)
    hostile = rand2 >= 8
    action = jnp.select([o.t == 0, o.t == 1, deadlocked, hostile], [C, o.their_last, C, D], o.their_last)
    early = o.t < 2
    rand = jnp.where(early | deadlocked, rand, rand2)
    dead = jnp.select(
        [early, deadlocked, hostile], [dead, jnp.where(dead == 3, 4, 0), dead], jnp.where(changed, dead + 1, 0)
    )
    return action, jnp.stack([rand, dead, o.their_last]).astype(jnp.int32)


omega_tft = Bot("Omega TFT", _omega_init, _omega_act, memory_depth=INF)


def _original_gradual_init(p, k):
    return (jnp.bool_(False), jnp.bool_(False), jnp.int32(0), jnp.int32(0))  # calming, punishing, count, limit


def _original_gradual_act(p, s, o, k):
    calming, punishing, count, limit = s
    keep_punishing = punishing & (count < limit)
    stop_punishing = punishing & ~keep_punishing
    start = ~punishing & (o.their_last == D)
    live = ~calming  # calming short-circuits everything else
    action = jnp.where(live & (keep_punishing | start), D, C)
    new_calming = live & stop_punishing
    new_punishing = jnp.where(live, keep_punishing | start, punishing)
    count = jnp.where(live & (keep_punishing | start), count + 1, jnp.where(live & stop_punishing, 0, count))
    limit = jnp.where(live & start, limit + 1, limit)
    return action, (new_calming, new_punishing, count.astype(jnp.int32), limit.astype(jnp.int32))


original_gradual = Bot("Original Gradual", _original_gradual_init, _original_gradual_act, memory_depth=INF)


def _gradual_act(p, s, o, k):
    punish, calm, n_def = s
    n_def = n_def + (o.their_last == D)
    start = (punish == 0) & (calm == 0) & (o.their_last == D)
    action = jnp.where((punish > 0) | start, D, C)
    new_punish = jnp.where(punish > 0, punish - 1, jnp.where(start, n_def - 1, 0))
    new_calm = jnp.where(punish > 0, calm, jnp.where(calm > 0, calm - 1, jnp.where(start, 2, 0)))
    return action, jnp.stack([new_punish, new_calm, n_def]).astype(jnp.int32)


gradual = Bot("Gradual", lambda p, k: jnp.zeros(3, jnp.int32), _gradual_act, memory_depth=INF)


def _contrite_act(p, s, o, k):
    contrite, intended = s  # intended: my last move before noise
    apologise = contrite & (o.my_last == C)
    provoked = (intended != o.my_last) & (o.my_last == D) & (o.their_last == C)
    action = jnp.where((o.t == 0) | apologise, C, o.their_last)
    contrite = jnp.where(o.t == 0, contrite, jnp.where(apologise, False, contrite | provoked))
    return action, (contrite, action.astype(jnp.int32))


contrite_tit_for_tat = Bot(
    "Contrite Tit For Tat", lambda p, k: (jnp.bool_(False), jnp.int32(-1)), _contrite_act, memory_depth=3
)


def _adaptive_tft_init(p, k):
    # world (float32), opponent's current C run, run length at which the library's float64 world hits 1.0
    return (jnp.float32(0.5), jnp.int32(0), jnp.int32(53))


def _adaptive_tft_act(p, s, o, k):
    # rate 0.5: world >= 0.5 iff their last move was C, except that float64 rounds world to exactly 1.0
    # after 53 straight C's (54 if world was below 1/3 when the run began), and a D then leaves 0.5 -> C
    world, run, need = s
    coop = o.their_last == C
    saturated = run >= need
    need = jnp.where(coop & (run == 0), jnp.where(world >= 1 / 3, 53, 54), need)
    new_world = jnp.where(coop, world + 0.5 * (1 - world), jnp.where(saturated, 0.5, 0.5 * world))
    action = jnp.where((o.t == 0) | coop | saturated, C, D)
    world = jnp.where(o.t == 0, world, new_world).astype(jnp.float32)
    run = jnp.where(coop, run + 1, 0)
    return action, (world, run.astype(jnp.int32), need.astype(jnp.int32))


adaptive_tit_for_tat = Bot("Adaptive Tit For Tat", _adaptive_tft_init, _adaptive_tft_act, memory_depth=INF)


def _spiteful_tft_act(p, s, o, k):
    retaliating, prev = s
    retaliating = retaliating | ((o.their_last == D) & (prev == D))
    return jnp.where(retaliating, D, _tft(o)), (retaliating, o.their_last)


spiteful_tit_for_tat = Bot(
    "Spiteful Tit For Tat", lambda p, k: (jnp.bool_(False), jnp.int32(-1)), _spiteful_tft_act, memory_depth=INF
)


def _slow_tf2t2_act(p, prev, o, k):
    action = jnp.where(o.t < 2, C, jnp.where(prev == o.their_last, o.their_last, o.my_last))
    return action, o.their_last


slow_tit_for_two_tats_2 = Bot(
    "Slow Tit For Two Tats 2", lambda p, k: jnp.int32(-1), _slow_tf2t2_act, memory_depth=2
)

alexei = Bot("Alexei", no_state, lambda p, s, o, k: (_final_defect(o, _tft(o)), s), memory_depth=INF)


def _eugine_nier_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    action = jnp.where(o.t == 0, C, jnp.where(n_def >= 5, D, o.their_last))
    return _final_defect(o, action), n_def.astype(jnp.int32)


eugine_nier = Bot("EugineNier", lambda p, k: jnp.int32(0), _eugine_nier_act, memory_depth=INF)


def _n_tits_act(p, s, o, k):
    count, run = s  # retaliations left, opponent's current D run
    run = jnp.where(o.their_last == D, run + 1, 0)
    count = jnp.where((p["M"] == 0) | (run >= p["M"]), p["N"], count)
    return jnp.where(count > 0, D, C), jnp.stack([jnp.maximum(count - 1, 0), run]).astype(jnp.int32)


def n_tits_for_m_tats(N=3, M=2):
    """Defect N times once the opponent has defected M times in a row."""
    params = {"N": jnp.int32(N), "M": jnp.int32(M)}
    return Bot(
        "N Tit(s) For M Tat(s)", lambda p, k: jnp.zeros(2, jnp.int32), _n_tits_act, params, memory_depth=max(M, N)
    )


def _michaelos_act(p, is_defector, o, k):
    test = ~is_defector & (o.my_last == D) & (o.their_last == C)
    coin = _coop_with_prob(k, 0.5)
    action = jnp.select([o.t == 0, is_defector, test], [C, D, coin], o.their_last)
    is_defector = is_defector | (test & (coin == D))
    return _final_defect(o, action), is_defector


michaelos = Bot("Michaelos", lambda p, k: jnp.bool_(False), _michaelos_act, memory_depth=INF, stochastic=True)


def _random_tft_act(p, s, o, k):
    # TFT on odd rounds, cooperate with probability p on even rounds after the first
    action = jnp.where(o.t == 0, C, jnp.where(o.t % 2 == 1, o.their_last, _coop_with_prob(k, p)))
    return action, s


random_tit_for_tat = Bot(
    "Random Tit for Tat", no_state, _random_tft_act, params=jnp.float32(0.5), stochastic=True
)

burn_both_ends = memory_one([0.9, 0.0, 0.9, 0.0], name="Burn Both Ends")

STRATEGIES = [
    adaptive_tit_for_tat,
    alexei,
    anti_tit_for_tat,
    bully,
    burn_both_ends,
    contrite_tit_for_tat,
    dynamic_two_tits_for_tat,
    eugine_nier,
    gradual,
    hard_tit_for_2_tats,
    hard_tit_for_tat,
    michaelos,
    n_tits_for_m_tats(),
    omega_tft,
    original_gradual,
    random_tit_for_tat,
    slow_tit_for_two_tats_2,
    sneaky_tit_for_tat,
    spiteful_tit_for_tat,
    two_tits_for_tat,
]
