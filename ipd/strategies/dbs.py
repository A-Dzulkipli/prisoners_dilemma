"""DBS (Derived Belief Strategy, Au and Nau 2006).

The opponent model is kept per outcome (my move, their move), indexed 2 * mine + theirs:
deterministic rules Rc, default rules Rd (initially TFT) and discounted frequencies Rp. The library's
minimax tree search only depends on the outcome at each depth, so it is a backward pass over 4 values.
"""
import functools

import jax.numpy as jnp
from jax import lax

from ipd.bots import INF, Bot
from ipd.game import C, D

# the library's tree search uses fixed Axelrod payoffs, not the match game
OUTCOME_VALUE = jnp.array([3.0, 0.0, 5.0, 1.0], jnp.float32)
TIE_TOL = 1e-9


def _init(p, k):
    return {
        "rd": jnp.array([1.0, 1.0, 0.0, 0.0], jnp.float32),  # P(opponent C) after each outcome
        "in_rd": jnp.ones(4, jnp.bool_),
        "rc": jnp.zeros(4, jnp.float32),
        "in_rc": jnp.zeros(4, jnp.bool_),
        "violations": jnp.zeros(4, jnp.int32),
        "v": jnp.int32(0),
        # discounted frequencies of opponent C and of D after each outcome (kept apart so both stay
        # precise near 0), and their discounted weight
        "freq": jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], jnp.float32),
        "weight": jnp.ones(4, jnp.float32),
        # run of identical opponent replies after each outcome, and that reply
        "streak": jnp.zeros(4, jnp.int32),
        "streak_move": jnp.full(4, -1, jnp.int32),
        "prev_outcome": jnp.int32(0),
    }


def _learn(p, s, o):
    """The library's model update, once the opponent has replied to an outcome (t >= 2)."""
    cond, reply = s["prev_outcome"], o.their_last
    hit = jnp.arange(4) == cond
    rule = (reply == C).astype(jnp.float32)  # rules store P(C)
    # Rp: discounted frequencies, kept as ratios so the weights cannot underflow
    w = p["alpha"] * s["weight"] + hit
    seen = jnp.stack([rule, 1 - rule])[:, None]
    freq = jnp.where(hit, (p["alpha"] * s["weight"] * s["freq"] + seen) / w, s["freq"])
    same = s["streak_move"][cond] == reply
    streak = s["streak"].at[cond].set(jnp.where(same, s["streak"][cond] + 1, 1))
    streak_move = s["streak_move"].at[cond].set(reply)
    rd, in_rd, rc, in_rc, viol, v = s["rd"], s["in_rd"], s["rc"], s["in_rc"], s["violations"], s["v"]
    # promote the reply to a deterministic rule after promotion_threshold identical replies in a row
    promote = ~in_rc[cond] & (streak[cond] >= p["promotion_threshold"])
    rc = jnp.where(hit & promote, rule, rc)
    in_rc = in_rc | (hit & promote)
    viol = jnp.where(hit & promote, 0, viol)
    # a violated rule in Rc: after violation_threshold in a row, move Rc into Rd and clear it
    agrees = rc[cond] == rule
    viol = jnp.where(hit & in_rc, jnp.where(agrees, 0, viol + 1), viol)
    demote = in_rc[cond] & ~agrees & (viol[cond] >= p["violation_threshold"])
    rd = jnp.where(demote & in_rc, rc, rd)
    in_rd = in_rd | (demote & in_rc)
    in_rc = in_rc & ~demote
    viol = jnp.where(demote, 0, viol)
    v = jnp.where(demote, 0, v)
    r_plus_in_rc = in_rc[cond] & (rc[cond] == rule)
    r_minus_in_rd = in_rd[cond] & (rd[cond] == 1 - rule)
    v = v + r_minus_in_rd
    clear_rd = (v > p["reject_threshold"]) | (r_plus_in_rc & r_minus_in_rd)
    in_rd = in_rd & ~clear_rd
    v = jnp.where(clear_rd, 0, v)
    return dict(s, rd=rd, in_rd=in_rd, rc=rc, in_rc=in_rc, violations=viol, v=v, freq=freq, weight=w,
                streak=streak, streak_move=streak_move)


def _policy(s):
    """Pi as a double-float. Pi = Rc updated by Rd then Rp, so Rd overrides Rc when both hold a rule."""
    pc, pd = s["freq"]
    one_minus_pd = _two_sum(jnp.ones(4, jnp.float32), -pd)
    rp = jnp.where(pc <= pd, pc, one_minus_pd[0]), jnp.where(pc <= pd, 0.0, one_minus_pd[1])
    rule = jnp.where(s["in_rd"], s["rd"], s["rc"])
    det = s["in_rd"] | s["in_rc"]
    return jnp.where(det, rule, rp[0]), jnp.where(det, 0.0, rp[1])


# The policy and the search are double-float32 (hi, lo pairs, ~44 bits): in plain float32, genuine
# value gaps of ~1e-8 (rules learned as frequencies like 1 - 0.75^60) get lost, while values that are
# equal in exact arithmetic must still be recognised as ties, which the library breaks towards C.
# The optimization barriers stop XLA from simplifying rounding errors away, e.g. ((1 - x) - 1) -> -x.


def _two_sum(a, b):
    s = lax.optimization_barrier(a + b)
    bb = lax.optimization_barrier(s - a)
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = lax.optimization_barrier(4097.0 * a)
    hi = c - lax.optimization_barrier(c - a)
    return hi, a - hi


def df_add(x, y):
    s, e = _two_sum(x[0], y[0])
    return _two_sum(s, e + x[1] + y[1])


def df_mul(x, y):
    p = lax.optimization_barrier(x[0] * y[0])
    (xh, xl), (yh, yl) = _split(x[0]), _split(y[0])
    e = ((xh * yh - p) + xh * yl + xl * yh) + xl * yl
    return _two_sum(p, e + x[0] * y[1] + x[1] * y[0])


def _df_expect(q, a, b):
    """q * a + (1 - q) * b, as b + q * (a - b)."""
    return df_add(b, df_mul(df_add(a, (-b[0], -b[1])), q))


def _df_max(a, b):
    a_wins = (a[0] > b[0]) | ((a[0] == b[0]) & (a[1] >= b[1]))
    return jnp.where(a_wins, a[0], b[0]), jnp.where(a_wins, a[1], b[1])


def _tree_search(pi, outcome, depth):
    """minimax_tree_search from the outcome: value of C minus value of D, as a double-float."""
    zero = jnp.zeros(4, jnp.float32)
    value = (OUTCOME_VALUE, zero)
    for _ in range(depth - 1):
        exp_c = _df_expect(pi, (value[0][0], value[1][0]), (value[0][1], value[1][1]))
        exp_d = _df_expect(pi, (value[0][2], value[1][2]), (value[0][3], value[1][3]))
        value = df_add(_df_max(exp_c, exp_d), (OUTCOME_VALUE, zero))
    q = pi[0][outcome], pi[1][outcome]
    exp_c = _df_expect(q, (value[0][0], value[1][0]), (value[0][1], value[1][1]))
    exp_d = _df_expect(q, (value[0][2], value[1][2]), (value[0][3], value[1][3]))
    return df_add(exp_c, (-exp_d[0], -exp_d[1]))


def _dbs_act(p, s, o, k, tree_depth):
    learned = _learn(p, s, o)
    s = {key: jnp.where(o.t >= 2, learned[key], s[key]) for key in s}
    outcome = jnp.maximum(2 * o.my_last + o.their_last, 0)
    diff = _tree_search(_policy(s), outcome, tree_depth)
    action = jnp.where((o.t == 0) | (diff[0] + diff[1] >= -TIE_TOL), C, D)
    return action, dict(s, prev_outcome=outcome)


def dbs(discount_factor=0.75, promotion_threshold=3, violation_threshold=4, reject_threshold=3, tree_depth=5):
    """tree_depth is fixed at construction (the search is unrolled), the rest are params."""
    params = {"alpha": jnp.float32(discount_factor), "promotion_threshold": jnp.int32(promotion_threshold),
              "violation_threshold": jnp.int32(violation_threshold), "reject_threshold": jnp.int32(reject_threshold)}
    act = functools.partial(_dbs_act, tree_depth=tree_depth)
    return Bot("DBS", _init, act, params, memory_depth=INF)


STRATEGIES = [dbs()]
