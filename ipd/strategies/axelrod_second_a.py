"""Strategies from Axelrod's second tournament (Appold to Harrington), ported from axelrod v4.14."""
import jax
import jax.numpy as jnp
from jax import lax

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C, D

INF = float("inf")


def _i(x):
    return jnp.int32(x)


# --- Appold ------------------------------------------------------------------


def _appold_init(p, k):
    # opponent C counts and totals after our C / D two turns ago; D starts at 1/1
    return {"c_after": jnp.array([0, 1], jnp.int32), "total": jnp.array([0, 1], jnp.int32),
            "my_prev": _i(C), "first_def": jnp.bool_(False)}


def _appold_act(p, s, o, k):
    turn = o.t + 1
    x = jnp.where(turn <= 2, C, s["my_prev"])  # our move two turns ago
    seen = turn > 1
    total = s["total"].at[x].add(seen)
    c_after = s["c_after"].at[x].add(seen & (o.their_last == C))
    forgive = (turn > 4) & (o.their_last == D) & ~s["first_def"]
    prob = c_after[x] / total[x]
    a = jnp.where((turn <= 4) | forgive, C, _coop_with_prob(k, prob))
    s = {"c_after": c_after, "total": total, "my_prev": o.my_last, "first_def": s["first_def"] | forgive}
    return a, s


second_by_appold = Bot("Second by Appold", _appold_init, _appold_act, memory_depth=INF, stochastic=True)

# --- Black -------------------------------------------------------------------

_BLACK_PROB = jnp.array([1.0, 1.0, 0.88, 0.68, 0.4, 0.04], jnp.float32)


def _black_act(p, last5, o, k):
    """Cooperate with prob depending on opponent defections in the last five moves."""
    last5 = ((last5 << 1) | jnp.maximum(o.their_last, 0)) & 0b11111
    n_d = lax.population_count(last5)
    return jnp.where(o.t < 5, C, _coop_with_prob(k, _BLACK_PROB[n_d])), last5


second_by_black = Bot("Second by Black", lambda p, k: _i(0), _black_act, memory_depth=5, stochastic=True)

# --- Borufsen ----------------------------------------------------------------


def _borufsen_init(p, k):
    z = _i(0)
    return {"cc": z, "cd": z, "mutual": z, "echo": z, "flip": jnp.bool_(False), "defect": jnp.bool_(False),
            "my_prev": _i(C), "their_prev": _i(C)}


def _borufsen_act(p, s, o, k):
    """TFT with mutual-defection and echo breakers; every 25 turns, check for random or defective opponents."""
    turn = o.t + 1
    my2, their2 = s["my_prev"], s["their_prev"]  # moves two turns ago (C before turn 3)
    opp_c = o.their_last == C
    counted = (turn >= 3) & opp_c
    cc = s["cc"] + (counted & (my2 == C))
    cd = s["cd"] + (counted & (my2 == D))

    check = (turn > 2) & (turn % 25 == 2)
    coops = cc + cd
    to_defect = (coops < 3) | ((coops >= 8) & (coops <= 17) & (10 * cc < 7 * coops))
    defect = jnp.where(check, to_defect, s["defect"])
    leaving_defect = check & s["defect"] & ~defect
    cc, cd = jnp.where(check, 0, cc), jnp.where(check, 0, cd)
    clear = check & defect
    mutual, echo = jnp.where(clear, 0, s["mutual"]), jnp.where(clear, 0, s["echo"])
    flip = s["flip"] & ~clear

    normal = ~defect & ~leaving_defect
    mutual_new = jnp.where((o.my_last == D) & (o.their_last == D), mutual + 1, 0)
    hit_mutual = mutual_new >= 3
    echo_new = jnp.where((o.my_last != o.their_last) & (o.my_last == their2) & (o.their_last == my2), echo + 1, 0)
    hit_echo = ~hit_mutual & (echo_new >= 3)
    want = jnp.where(hit_mutual, C, o.their_last)
    flip_now = flip | hit_echo
    use_flip = (want == D) & flip_now
    a_normal = jnp.where(use_flip, C, want)

    a = jnp.where(o.t == 0, C, jnp.where(normal, a_normal, D))
    upd = (o.t > 0) & normal
    s = {
        "cc": jnp.where(o.t > 0, cc, s["cc"]),
        "cd": jnp.where(o.t > 0, cd, s["cd"]),
        "mutual": jnp.where(upd, jnp.where(hit_mutual | hit_echo, 0, mutual_new), mutual),
        "echo": jnp.where(upd, jnp.where(hit_mutual | hit_echo, 0, echo_new), echo),
        "flip": jnp.where(upd, flip_now & ~use_flip, flip),
        "defect": defect,
        "my_prev": o.my_last,
        "their_prev": o.their_last,
    }
    return a, s


second_by_borufsen = Bot("Second by Borufsen", _borufsen_init, _borufsen_act, memory_depth=INF)

# --- Cave --------------------------------------------------------------------


def _cave_act(p, n_d, o, k):
    """Defect against defective or random-looking opponents, else answer D with D or a coin flip."""
    turn = o.t + 1
    n_d = n_d + (o.their_last == D)
    # perc_defects = n_d / turn (the library divides by turn, not by moves seen)
    harsh = ((turn > 39) & (100 * n_d > 39 * turn)) | ((turn > 29) & (100 * n_d > 65 * turn)) | (
        (turn > 19) & (100 * n_d > 79 * turn))
    reply_d = jnp.where(n_d > 17, D, _coop_with_prob(k, 0.5))
    a = jnp.where(o.t == 0, C, jnp.where(harsh, D, jnp.where(o.their_last == D, reply_d, C)))
    return a, n_d


second_by_cave = Bot("Second by Cave", lambda p, k: _i(0), _cave_act, memory_depth=INF, stochastic=True)

# --- Champion ----------------------------------------------------------------


def _champion_act(p, n_d, o, k):
    """C for 10 turns, TFT to turn 25, then defect after a D if the opponent defects often enough."""
    n_d = n_d + (o.their_last == D)
    prop = n_d / jnp.maximum(o.t, 1)
    late = jnp.where((o.their_last == D) & (prop >= jnp.maximum(0.4, jax.random.uniform(k))), D, C)
    a = jnp.where(o.t < 10, C, jnp.where(o.t < 25, o.their_last, late))
    return a, n_d


second_by_champion = Bot("Second by Champion", lambda p, k: _i(0), _champion_act, memory_depth=INF, stochastic=True)

# --- Colbert -----------------------------------------------------------------

# FSM: [state, opponent's last move] -> next state / action
_COLBERT_NEXT = jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [7, 8], [9, 9], [10, 10], [7, 7]])
_COLBERT_ACT = jnp.array([[C, C], [C, C], [C, C], [C, C], [D, D], [C, C], [C, C], [C, D], [D, D], [C, C], [C, C]])


def _colbert_act(p, state, o, k):
    """D on turn 6, then C after C and D, D, C, C after a D."""
    opp = jnp.maximum(o.their_last, 0)
    a = jnp.where(o.t == 0, C, _COLBERT_ACT[state, opp])
    return a, jnp.where(o.t == 0, state, _COLBERT_NEXT[state, opp])


second_by_colbert = Bot("Second by Colbert", lambda p, k: _i(0), _colbert_act, memory_depth=4)

# --- Eatherley ---------------------------------------------------------------


def _eatherley_act(p, n_d, o, k):
    """After a D, defect with probability equal to the opponent's defection rate."""
    n_d = n_d + (o.their_last == D)
    prop = n_d / jnp.maximum(o.t, 1)
    a = jnp.where((o.t == 0) | (o.their_last == C), C, _coop_with_prob(k, 1 - prop))
    return a, n_d


second_by_eatherley = Bot("Second by Eatherley", lambda p, k: _i(0), _eatherley_act, memory_depth=INF,
                          stochastic=True)

# --- Getzler -----------------------------------------------------------------


def _getzler_act(p, flack, o, k):
    """Defect with probability flack = sum of 0.5^(turns ago) over opponent defections."""
    flack = jnp.where(o.t == 0, flack, 0.5 * (flack + (o.their_last == D)))
    return _coop_with_prob(k, 1.0 - flack), flack


second_by_getzler = Bot("Second by Getzler", lambda p, k: jnp.float32(0), _getzler_act, memory_depth=INF,
                        stochastic=True)

# --- Gladstein ---------------------------------------------------------------


def _gladstein_act(p, s, o, k):
    """D first; while the opponent never defects keep my cooperation ratio at most 0.5; after its first D, C then TFT."""
    my_c, patsy = s
    my_c = my_c + (o.my_last == C)
    apologize = patsy & (o.their_last == D)
    patsy_a = jnp.where(apologize, C, jnp.where(2 * my_c > jnp.maximum(o.t, 1), D, C))
    a = jnp.where(o.t == 0, D, jnp.where(patsy, patsy_a, o.their_last))
    return a, (my_c, patsy & ~apologize)


second_by_gladstein = Bot("Second by Gladstein", lambda p, k: (_i(0), jnp.bool_(True)), _gladstein_act,
                          memory_depth=INF)

# --- Graaskamp and Katzen ----------------------------------------------------

_GK_TURNS = jnp.array([11, 21, 31, 41, 51, 101])
_GK_SCORES = jnp.array([23, 53, 83, 113, 143, 293], jnp.float32)


def _gk_act(p, s, o, k):
    """TFT, but defect forever if own score is low at checkpoint turns."""
    score, defect = s
    turn = o.t + 1
    score = score + jnp.where(o.t == 0, 0.0, o.my_payoff[o.my_last, o.their_last])
    low = jnp.any((turn == _GK_TURNS) & (score < _GK_SCORES))
    new_defect = defect | ((o.t > 0) & low)
    a = jnp.where(o.t == 0, C, jnp.where(new_defect, D, o.their_last))
    # the library stops tracking score once in defect mode
    return a, (jnp.where(defect, s[0], score), new_defect)


second_by_graaskamp_katzen = Bot("Second by GraaskampKatzen", lambda p, k: (jnp.float32(0), jnp.bool_(False)),
                                 _gk_act, memory_depth=INF)

# --- Grofman -----------------------------------------------------------------


def _grofman_act(p, last8, o, k):
    """C, C, then TFT to turn 7, then react to opponent defections in 7 of the last 8 rounds (not the latest)."""
    last8 = ((last8 << 1) | jnp.maximum(o.their_last, 0)) & 0xFF
    n_d = lax.population_count(last8 >> 1)
    late = jnp.where(((o.my_last == C) & (n_d <= 2)) | ((o.my_last == D) & (n_d <= 1)), C, D)
    a = jnp.where(o.t < 2, C, jnp.where(o.t <= 6, o.their_last, late))
    return a, last8


second_by_grofman = Bot("Second by Grofman", lambda p, k: _i(0), _grofman_act, memory_depth=8)

# --- Harrington --------------------------------------------------------------

NORMAL, FAIR_WEATHER, DEFECT = 0, 1, 2


def _harrington_init(p, k):
    z = _i(0)
    return {
        "mode": _i(NORMAL), "recorded_defects": z, "exit_meter": z, "was_defective": jnp.bool_(False),
        "prob": jnp.float32(0.25), "move_history": jnp.zeros((4, 2), jnp.int32), "more_coop": z,
        "last_generous": _i(3), "burned": jnp.bool_(False), "defect_streak": z,
        "parity_streak": jnp.zeros(2, jnp.int32), "parity_bit": z, "parity_limit": _i(5), "parity_hits": z,
        "opp_coops": z, "my_prev": _i(C), "their_prev": _i(C),
    }


def _harrington_random(mh, recorded_defects, turn):
    """Modified chi-squared test on the [our+their move two ago, their last] table: True means random."""
    denom = turn - 2
    expected = jnp.outer(mh.sum(1), mh.sum(0)) / denom
    chi2 = jnp.where(expected > 1.0, (expected - mh) ** 2 / expected, 0.0).sum()
    lopsided = (5 * mh[0, 0] >= 4 * denom) | (4 * recorded_defects < denom) | (4 * recorded_defects > 3 * denom)
    return ~lopsided & (chi2 <= 3)


def _harrington_act(p, s, o, k):
    """Mostly TFT with random/streak detection, a probe on turn 37 and growing random probes after."""
    turn = o.t + 1
    opp_d = o.their_last == D
    tft = o.their_last

    # Defect mode: leave when the opponent keeps defecting
    exit_meter = s["exit_meter"] + jnp.where(opp_d, 1, -3)
    leave = exit_meter >= 11

    # Normal / Fair-weather: record history and test for randomness
    track = ~s["was_defective"] & (turn > 2)
    row = (s["their_prev"] == D) + 2 * (s["my_prev"] == D)
    mh = s["move_history"].at[row, jnp.maximum(o.their_last, 0)].add(track)
    recorded_defects = s["recorded_defects"] + (track & opp_d)
    is_random = (~s["was_defective"] & (turn % 15 == 0) & (turn > 15)
                 & _harrington_random(mh, recorded_defects, turn))
    burned = s["burned"] | ((s["last_generous"] == 2) & opp_d)
    fair = (turn == 38) & opp_d & (s["opp_coops"] == 36)
    fw_coop = (s["mode"] == FAIR_WEATHER) & ~opp_d
    defect_streak = jnp.where(opp_d, s["defect_streak"] + 1, 0)
    streak_hit = defect_streak >= 20
    flipped = 1 - s["parity_bit"]
    flipped_streak = jnp.where(opp_d, s["parity_streak"][flipped] + 1, 0)
    parity_hit = flipped_streak >= s["parity_limit"]
    u = jax.random.uniform(k)

    # branches in the library's order of early returns
    conds = [is_random, fair, fw_coop, streak_hit, parity_hit, s["more_coop"] >= 1, turn < 37, turn == 37,
             burned | (u > s["prob"])]
    b = jnp.select(conds, list(range(9)), 9)
    a = jnp.select([b == i for i in range(10)], [D, C, C, D, C, C, tft, D, tft, D])

    reached_burn = b >= 1
    reached_streak = b >= 3
    reached_parity = b >= 4
    bit = jnp.where(reached_parity, flipped, s["parity_bit"])
    parity_streak = jnp.where(reached_parity, s["parity_streak"].at[flipped].set(flipped_streak),
                              s["parity_streak"])
    parity_streak = jnp.where(b == 4, parity_streak.at[flipped].set(0), parity_streak)
    inc_parity = ((b == 3) | (b == 6) | (b == 8)) & (a == D)
    parity_streak = parity_streak.at[bit].add(inc_parity)
    parity_hits = s["parity_hits"] + (b == 4)
    lower = ((b == 2) | (b == 4) | (b == 5) | (b == 6) | (b == 8)) & (a == C)
    generous = (b == 7) | (b == 9)
    mode = jnp.select([b == 0, b == 1, reached_streak & (s["mode"] == FAIR_WEATHER)], [DEFECT, FAIR_WEATHER, NORMAL],
                      s["mode"])
    nd = {
        "mode": mode,
        "recorded_defects": recorded_defects,
        "exit_meter": s["exit_meter"],
        "was_defective": s["was_defective"],
        "prob": s["prob"] + 0.05 * (b == 9),
        "move_history": mh,
        "more_coop": jnp.where(generous, 2, s["more_coop"] - lower),
        "last_generous": jnp.where(generous, 1, s["last_generous"] + lower),
        "burned": jnp.where(reached_burn, burned, s["burned"]),
        "defect_streak": jnp.where(reached_streak, defect_streak, s["defect_streak"]),
        "parity_streak": parity_streak,
        "parity_bit": bit,
        "parity_limit": jnp.where(parity_hits >= 8, 3, s["parity_limit"]),
        "parity_hits": parity_hits,
    }

    in_defect = s["mode"] == DEFECT
    dm = dict(s, exit_meter=exit_meter, mode=jnp.where(leave, NORMAL, DEFECT),
              was_defective=s["was_defective"] | leave, more_coop=jnp.where(leave, 2, s["more_coop"]))
    first = o.t == 0
    new = jax.tree.map(lambda d, n, old: jnp.where(first, old, jnp.where(in_defect, d, n)), dm, dict(s, **nd), s)
    new.update(opp_coops=s["opp_coops"] + (o.their_last == C), my_prev=o.my_last, their_prev=o.their_last)
    a = jnp.where(first, C, jnp.where(in_defect, jnp.where(leave, C, D), a))
    return a, new


second_by_harrington = Bot("Second by Harrington", _harrington_init, _harrington_act, memory_depth=INF,
                           stochastic=True)

STRATEGIES = [
    second_by_appold,
    second_by_black,
    second_by_borufsen,
    second_by_cave,
    second_by_champion,
    second_by_colbert,
    second_by_eatherley,
    second_by_getzler,
    second_by_gladstein,
    second_by_graaskamp_katzen,
    second_by_grofman,
    second_by_harrington,
]
