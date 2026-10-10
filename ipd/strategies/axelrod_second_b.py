"""Ports of axelrod's second-tournament strategies (Kluepfel to Yamachi, part b).

Each bot follows the axelrod v4.14 implementation, not the original tournament description.
Running scores use the bot's own matrix for both players, as the library scores with its own game.
"""
import jax
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C, D

INF = float("inf")
NEVER = jnp.int32(2**30)  # stands in for an infinite turn number


def _scores(o):
    """Last round's payoff to me and to them, both from my matrix (0 on round 0)."""
    m, th = jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)
    live = o.t > 0
    return jnp.where(live, o.my_payoff[m, th], 0.0), jnp.where(live, o.my_payoff[th, m], 0.0)


# --- Tester ---------------------------------------------------------------------------------


def _tester_act(p, is_tft, o, k):
    opp_d = o.their_last == D
    action = jnp.select(
        [o.t == 0, is_tft, opp_d, o.t <= 2], [D, o.their_last, C, C], 1 - o.my_last
    )
    return action, is_tft | ((o.t > 0) & opp_d)


second_by_tester = Bot(
    "Second by Tester", lambda p, k: jnp.bool_(False), _tester_act, memory_depth=INF
)


# --- Tranquilizer ---------------------------------------------------------------------------


def _tranquilizer_init(p, k):
    i, f = jnp.int32(0), jnp.float32(0)
    return dict(phase=i, consec=i, r1=jnp.float32(5), r2=f, n1=jnp.float32(1), n2=jnp.float32(1),
                score=f, opp_score=f, opp_coops=i)


def _tranquilizer_act(p, s, o, k):
    """Cooperates, throws in occasional defections while scoring >= 2.25 per turn, else TFT-ish."""
    live = o.t > 0
    mine, theirs = _scores(o)
    opp, me = jnp.maximum(o.their_last, 0), jnp.maximum(o.my_last, 0)
    score, opp_score = s["score"] + mine, s["opp_score"] + theirs
    opp_coops = s["opp_coops"] + (live & (o.their_last == C))
    consec = jnp.where(opp == D, s["consec"] + 1, 0)
    # payoff-like value of the round after a "good" defection: 3 - 3*opp + 2*me - opp*me
    val = 3.0 - 3 * opp + 2 * me - opp * me
    in1, in2 = s["phase"] == 1, s["phase"] == 2
    r1 = jnp.where(in1, (s["r1"] * s["n1"] + val) / (s["n1"] + 1), s["r1"])
    n1 = jnp.where(in1, s["n1"] + 1, s["n1"])
    r2 = jnp.where(in2, (s["r2"] * s["n2"] + val) / (s["n2"] + 1), s["r2"])
    n2 = jnp.where(in2, s["n2"] + 1, s["n2"])
    phase = jnp.where(in1, 2, 0)

    n = o.t + 1.0
    avg = score / n
    p_high = 0.95 - (r1 + r2 - 5) / 15 + 1 / n**2 - opp / 4
    p_mid = 0.25 + (opp_coops + 1) / n - consec * 0.25 + (score - opp_score) / 100 + 4 / n
    u = jax.random.uniform(k)
    high, mid = avg >= 2.25, avg >= 1.75
    action = jnp.select(
        [~live, in1, high, mid],
        [C, C, jnp.where(u <= p_high, C, D), jnp.where(u <= p_mid, C, D)],
        o.their_last,
    )
    phase = jnp.where(~in1 & high & (action == D), 1, phase)
    new = dict(phase=phase, consec=consec, r1=r1, r2=r2, n1=n1, n2=n2,
               score=score, opp_score=opp_score, opp_coops=opp_coops)
    return action, jax.tree.map(lambda a, b: jnp.where(live, a, b).astype(b.dtype), new, s)


second_by_tranquilizer = Bot(
    "Second by Tranquilizer", _tranquilizer_init, _tranquilizer_act, memory_depth=INF, stochastic=True
)


# --- Kluepfel -------------------------------------------------------------------------------


def _kluepfel_init(p, k):
    z = jnp.int32(0)
    return dict(cd=z, dd=z, dc=z, cc=z, my2=z, opp2=z, opp3=z)  # C before the match starts


def _kluepfel_act(p, s, o, k):
    """Defects against apparently random opponents, otherwise mostly echoes their recent moves."""
    opp1 = jnp.maximum(o.their_last, 0)
    # response counts: my move two rounds ago vs their last move
    upd = o.t >= 2
    cd = s["cd"] + (upd & (s["my2"] == D) & (opp1 == C))
    dd = s["dd"] + (upd & (s["my2"] == D) & (opp1 == D))
    cc = s["cc"] + (upd & (s["my2"] == C) & (opp1 == C))
    dc = s["dc"] + (upd & (s["my2"] == C) & (opp1 == D))
    random_opp = (
        (o.t > 26)
        & (cd >= (cd + dd) / 2 - 0.75 * jnp.sqrt(cd + dd))
        & (dc >= (dc + cc) / 2 - 0.75 * jnp.sqrt(dc + cc))
    )
    p_keep = jnp.where(opp1 == s["opp2"], 0.9, jnp.where(opp1 == C, 0.7, 0.6))
    keep = jax.random.uniform(k) < p_keep
    action = jnp.select(
        [random_opp, (opp1 == s["opp2"]) & (s["opp2"] == s["opp3"])],
        [D, opp1],
        jnp.where(keep, opp1, 1 - opp1),
    )
    new = dict(cd=cd, dd=dd, dc=dc, cc=cc, my2=jnp.maximum(o.my_last, 0), opp2=opp1, opp3=s["opp2"])
    return action, new


second_by_kluepfel = Bot(
    "Second by Kluepfel", _kluepfel_init, _kluepfel_act, memory_depth=INF, stochastic=True
)


# --- WmAdams --------------------------------------------------------------------------------


def _wm_adams_act(p, s, o, k):
    """Defects at their 4th, 7th, 9th defection; past 9, answers D with D at prob 1 - 0.5^(n-9)."""
    defs, first_d = s
    defs = defs + (o.their_last == D)
    first_d = jnp.where(o.t == 1, o.their_last == D, first_d)
    n = defs - first_d  # defections after their first move
    action = jnp.select(
        [o.t <= 1, (n == 4) | (n == 7) | (n == 9), (n > 9) & (o.their_last == D)],
        [C, D, _coop_with_prob(k, 0.5 ** (n - 9.0))],
        C,
    )
    return action, (defs, first_d)


second_by_wm_adams = Bot(
    "Second by WmAdams",
    lambda p, k: (jnp.int32(0), jnp.bool_(False)),
    _wm_adams_act,
    memory_depth=INF,
    stochastic=True,
)


# --- Weiner ---------------------------------------------------------------------------------


def _weiner_init(p, k):
    z = jnp.int32(0)
    return dict(forgive=jnp.bool_(False), grudge=z, padding=z, last12=jnp.zeros(12, jnp.int32),
                idx=z, opp2=z)


def _weiner_act(p, s, o, k):
    """TFT with occasional forgiveness; defects if 5 of their 12 moves before last were D."""
    live = o.t > 0
    opp_d = o.their_last == D
    # ring buffer of their moves, lagged one round
    upd = o.t >= 2
    last12 = jnp.where(upd, s["last12"].at[s["idx"]].set(s["opp2"]), s["last12"])
    idx = jnp.where(upd, (s["idx"] + 1) % 12, s["idx"])

    forgive_now = s["forgive"] & (s["grudge"] < o.t + 1) & opp_d
    grudge = jnp.where(forgive_now, s["grudge"] + 20, s["grudge"])
    padding = jnp.where(s["forgive"], 0, jnp.where(opp_d, s["padding"] + 1, 0))
    forgive = ~s["forgive"] & ~opp_d & (s["padding"] % 2 == 1)

    action = jnp.select(
        [~live, last12.sum() >= 5, forgive_now], [C, D, C], o.their_last
    )
    new = dict(forgive=forgive, grudge=grudge, padding=padding, last12=last12, idx=idx,
               opp2=jnp.maximum(o.their_last, 0))
    keep = dict(s, opp2=new["opp2"])
    return action, jax.tree.map(lambda a, b: jnp.where(live, a, b), new, keep)


second_by_weiner = Bot("Second by Weiner", _weiner_init, _weiner_act, memory_depth=INF)


# --- Tideman and Chieruzzi ------------------------------------------------------------------


def _tc_init(p, k):
    z, f = jnp.int32(0), jnp.float32(0)
    return dict(score=f, opp_score=f, last_fresh=z, fresh=jnp.bool_(False), to_beat=f, inc=f,
                opp_defs=z, opp2=z)


def _tc_act(p, s, o, k):
    """Cooperates while ahead by a growing margin; gives a 'fresh start' to non-random opponents."""
    live = o.t > 0
    rnd = o.t + 1
    mine, theirs = _scores(o)
    opp_defs = s["opp_defs"] + (o.their_last == D)
    score, opp_score = s["score"] + mine, s["opp_score"] + theirs
    fresh = s["fresh"]

    cdd = (s["opp2"] == C) & (o.their_last == D)
    to_beat = jnp.where(cdd, s["to_beat"] + s["inc"], s["to_beat"])
    inc = jnp.where(cdd, s["inc"] + 5, s["inc"])
    ahead = score - opp_score >= to_beat
    # fresh start if their defection count is > 3 sd from a fair coin's
    N = o.t
    sd = jnp.sqrt(N) / 2
    lopsided = (opp_defs <= N / 2 - 3 * sd) | (opp_defs >= N / 2 + 3 * sd)
    start_fresh = ~fresh & ~ahead & ~cdd & (rnd - s["last_fresh"] >= 10) & lopsided

    action = jnp.select([~live, fresh, ahead, start_fresh], [C, C, C, C], D)
    z = jnp.float32(0)
    new = dict(
        score=jnp.where(fresh, z, score),
        opp_score=jnp.where(fresh, z, opp_score),
        last_fresh=jnp.where(fresh, rnd, s["last_fresh"]),
        fresh=start_fresh,
        to_beat=jnp.where(fresh, z, to_beat),
        inc=jnp.where(fresh, z, inc),
        opp_defs=opp_defs,
        opp2=jnp.maximum(o.their_last, 0),
    )
    return action, jax.tree.map(lambda a, b: jnp.where(live, a, b).astype(b.dtype), new, s)


second_by_tideman_and_chieruzzi = Bot(
    "Second by Tideman and Chieruzzi", _tc_init, _tc_act, memory_depth=INF
)


# --- Leyvraz --------------------------------------------------------------------------------

# cooperation probability indexed by their moves (3 ago, 2 ago, last)
_LEYVRAZ_P = jnp.array([1.0, 0.5, 0.0, 0.25, 1.0, 1.0, 1.0, 0.25], jnp.float32)


def _leyvraz_act(p, s, o, k):
    """Reacts to the opponent's last three moves with fixed cooperation probabilities."""
    opp3, opp2 = s
    opp1 = jnp.maximum(o.their_last, 0)
    return _coop_with_prob(k, _LEYVRAZ_P[4 * opp3 + 2 * opp2 + opp1]), (opp2, opp1)


second_by_leyvraz = Bot(
    "Second by Leyvraz",
    lambda p, k: (jnp.int32(0), jnp.int32(0)),
    _leyvraz_act,
    memory_depth=3,
    stochastic=True,
)


# --- White ----------------------------------------------------------------------------------


def _white_act(p, defs, o, k):
    """Cooperates for 10 turns; then answers D with D only if floor(log(turn)) * their D's >= turn."""
    defs = defs + (o.their_last == D)
    turn = o.t + 1
    defect = (turn > 10) & (o.their_last == D) & (jnp.floor(jnp.log(turn)) * defs >= turn)
    return jnp.where(defect, D, C), defs


second_by_white = Bot("Second by White", lambda p, k: jnp.int32(0), _white_act, memory_depth=INF)


# --- RichardHufford -------------------------------------------------------------------------


def _hufford_init(p, k):
    i = jnp.int32
    return dict(n_agree=i(2), last4=jnp.ones(4, jnp.int32), idx=i(0), needed=i(21), streak=i(2),
                last_ab=NEVER, coop_ab=i(2), def_ab=i(2), my2=i(0))


def _hufford_act(p, s, o, k):
    """Plays C / TFT / D by how often they copy my move from two turns back, with planned
    'aberration' defections after long cooperation streaks."""
    live = o.t > 0
    turn = o.t + 1
    opp_c = o.their_last == C
    idx = (s["idx"] + 1) % 4
    agree = (s["my2"] == o.their_last).astype(jnp.int32)
    n_agree = s["n_agree"] + agree
    last4 = s["last4"].at[idx].set(agree)

    before_ab = turn < s["last_ab"]
    streak_a = jnp.where(opp_c, s["streak"] + 1, 0)
    hit = before_ab & (streak_a >= s["needed"])
    aberrate = hit & (streak_a == s["needed"])
    after_ab = ~before_ab & (turn == s["last_ab"] + 2)
    coop_ab = s["coop_ab"] + (after_ab & opp_c)
    def_ab = s["def_ab"] + (after_ab & ~opp_c)
    needed = jnp.where(after_ab, jnp.floor(20.0 * def_ab / coop_ab).astype(jnp.int32) + 1, s["needed"])
    streak = jnp.select([before_ab, after_ab], [streak_a, 0], s["streak"])
    last_ab = jnp.select([hit, after_ab], [turn, NEVER], s["last_ab"])

    # proportion_agree > 0.9 and >= 0.625, in integers
    n4 = last4.sum()
    action = jnp.select(
        [~live, aberrate, after_ab, (10 * n_agree > 9 * turn) & (n4 >= 4), (8 * n_agree >= 5 * turn) & (n4 >= 2)],
        [C, D, C, C, o.their_last],
        D,
    )
    new = dict(n_agree=n_agree, last4=last4, idx=idx, needed=needed, streak=streak, last_ab=last_ab,
               coop_ab=coop_ab, def_ab=def_ab, my2=jnp.maximum(o.my_last, 0))
    return action, jax.tree.map(lambda a, b: jnp.where(live, a, b), new, s)


second_by_richard_hufford = Bot(
    "Second by RichardHufford", _hufford_init, _hufford_act, memory_depth=INF
)


# --- Yamachi --------------------------------------------------------------------------------


def _yamachi_init(p, k):
    z = jnp.int32(0)
    return dict(counts=jnp.zeros(8, jnp.int32), mod1=z, mod2=z, opp2=z, opp3=z, opp_defs=z)


def _yamachi_act(p, s, o, k):
    """Predicts their move from (their move 2 ago, my last intended move); defects late on
    when they look like a coin flip."""
    turn = o.t + 1
    opp_defs = s["opp_defs"] + (o.their_last == D)
    # counts[(them 3 ago, my intended move 2 ago, them 2 ago)], my move as intended, before override
    counts = s["counts"].at[4 * s["opp3"] + 2 * s["mod2"] + s["opp2"]].add((turn >= 3).astype(jnp.int32))
    base = 4 * s["opp2"] + 2 * s["mod1"]
    intended = jnp.where((turn == 1) | (counts[base + C] >= counts[base + D]), C, D)
    # 0.45 < (opp_defs + 0.5) / turn < 0.55, in integers
    x = 100 * opp_defs + 50
    override = (turn > 40) & (45 * turn < x) & (x < 55 * turn)
    new = dict(counts=counts, mod1=intended, mod2=s["mod1"], opp2=jnp.maximum(o.their_last, 0),
               opp3=s["opp2"], opp_defs=opp_defs)
    return jnp.where(override, D, intended), new


second_by_yamachi = Bot("Second by Yamachi", _yamachi_init, _yamachi_act, memory_depth=INF)


# --- Mikkelson ------------------------------------------------------------------------------


def _mikkelson_act(p, s, o, k):
    """Credit-based: +1 per opponent C, -2 per D; defects when credit runs out and they defect >= 15%."""
    credit, defs = s
    live = o.t > 0
    turn = o.t + 1
    defs = defs + (o.their_last == D)
    credit = jnp.where(
        live, jnp.where(o.their_last == C, jnp.minimum(credit + 1, 8), jnp.maximum(credit - 2, -7)), credit
    )
    early = (turn > 2) & (credit <= 0) & (turn <= 10)
    action = jnp.select(
        [turn <= 2, credit > 0, turn <= 10, 100 * defs >= 15 * turn], [C, C, D, D], C
    )
    return action, (jnp.where(early, 4, credit), defs)


second_by_mikkelson = Bot(
    "Second by Mikkelson", lambda p, k: (jnp.int32(7), jnp.int32(0)), _mikkelson_act, memory_depth=INF
)


# --- Rowsam ---------------------------------------------------------------------------------

NORMAL, CYCLE1, CYCLE2, DEFECT = 0, 1, 2, 3


def _rowsam_act(p, s, o, k):
    """Cooperates, checking its score every 6 turns; distrust buys D-C-D cycles, then all-D."""
    mode, distrust, score = s
    turn = o.t + 1
    score = score + _scores(o)[0]
    normal = mode == NORMAL
    distrust = jnp.where(normal & (turn % 18 == 0) & (distrust >= 3), distrust - 1, distrust)
    check = normal & (turn % 6 == 0)
    ppt = score / turn  # the library divides by one more than the turns scored
    add = jnp.select([ppt < 1.0, ppt < 1.5, ppt < 2.0, ppt < 2.5], [5, 3, 2, 1], 0)
    distrusted = check & (add > 0)
    distrust = jnp.where(check, distrust + add, distrust)

    action = jnp.select([mode == DEFECT, mode == CYCLE1, mode == CYCLE2, distrusted], [D, C, D, D], C)
    mode = jnp.select(
        [mode == CYCLE1, mode == CYCLE2, distrusted],
        [CYCLE2, NORMAL, jnp.where(distrust >= 7, DEFECT, CYCLE1)],
        mode,
    )
    return action, (mode, distrust, score)


second_by_rowsam = Bot(
    "Second by Rowsam",
    lambda p, k: (jnp.int32(NORMAL), jnp.int32(0), jnp.float32(0)),
    _rowsam_act,
    memory_depth=INF,
)


STRATEGIES = [
    second_by_kluepfel,
    second_by_leyvraz,
    second_by_mikkelson,
    second_by_richard_hufford,
    second_by_rowsam,
    second_by_tester,
    second_by_tideman_and_chieruzzi,
    second_by_tranquilizer,
    second_by_weiner,
    second_by_white,
    second_by_wm_adams,
    second_by_yamachi,
]
