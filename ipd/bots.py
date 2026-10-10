"""Bots: pure functions over fixed-size state. See docs/CONTRACT.md."""
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ipd.game import AXELROD_PAYOFF, C, D

INF = float("inf")


class Bot(NamedTuple):
    name: str
    init: Callable  # (params, key) -> state
    act: Callable  # (params, state, obs, key) -> (action, state)
    params: Any = ()
    memory_depth: float = 1
    stochastic: bool = False


def no_state(params, key):
    return ()


def coop_with_prob(key, p):
    return jnp.where(jax.random.bernoulli(key, p), C, D)


def tft(o):
    """Tit for tat: C on round 0, then the opponent's last move."""
    return jnp.where(o.their_last == D, D, C)


def defect_at_end(o, action, n=1):
    """The library's FinalTransformer: defect on the last n turns when the length is known."""
    return jnp.where((o.turns > 0) & (o.t >= o.turns - n), D, action)


def last_payoff(o):
    """My payoff from the last round, 0 on round 0."""
    return jnp.where(o.t > 0, o.my_payoff[jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)], 0.0)


cooperator = Bot("Cooperator", no_state, lambda p, s, o, k: (C, s), memory_depth=0)
defector = Bot("Defector", no_state, lambda p, s, o, k: (D, s), memory_depth=0)
alternator = Bot("Alternator", no_state, lambda p, s, o, k: (jnp.where(o.t % 2 == 0, C, D), s))
tit_for_tat = Bot("Tit For Tat", no_state, lambda p, s, o, k: (tft(o), s))

suspicious_tit_for_tat = Bot(
    "Suspicious Tit For Tat",
    no_state,
    lambda p, s, o, k: (jnp.where(o.t == 0, D, o.their_last), s),
)

win_stay_lose_shift = Bot(
    "Win-Stay Lose-Shift",
    no_state,
    lambda p, s, o, k: (jnp.where((o.t == 0) | (o.my_last == o.their_last), C, D), s),
)


def _tf2t_act(p, prev, o, k):
    # prev is their move the round before last
    return jnp.where((o.their_last == D) & (prev == D), D, C), o.their_last


tit_for_two_tats = Bot("Tit For 2 Tats", lambda p, k: jnp.int32(-1), _tf2t_act, memory_depth=2)


def _grudger_act(p, wronged, o, k):
    wronged = wronged | (o.their_last == D)
    return jnp.where(o.t < p["n_opening"], p["opening"], jnp.where(wronged, D, C)), wronged


def grudger_bot(name="Grudger", opening=C, n_opening=0):
    """Play `opening` for n_opening rounds, then C until the opponent has defected and D from then on."""
    params = {"opening": jnp.int32(opening), "n_opening": jnp.int32(n_opening)}
    return Bot(name, lambda p, k: jnp.bool_(False), _grudger_act, params, memory_depth=INF)


grudger = grudger_bot()


def random_bot(p_coop=0.5):
    return Bot(
        "Random",
        no_state,
        lambda p, s, o, k: (coop_with_prob(k, p), s),
        params=jnp.float32(p_coop),
        memory_depth=0,
        stochastic=True,
    )


def _memory_one_act(p, s, o, k):
    last = jnp.maximum(2 * o.my_last + o.their_last, 0)
    p_coop = jnp.where(o.t == 0, p["p0"], p["p"][last])
    return coop_with_prob(k, p_coop), s


def memory_one(four_vector, p0=1.0, name="Memory One"):
    """four_vector: cooperation probability after (CC, CD, DC, DD), my move first."""
    params = {"p": jnp.asarray(four_vector, jnp.float32), "p0": jnp.float32(p0)}
    return Bot(name, no_state, _memory_one_act, params=params, stochastic=True)


def generous_tit_for_tat(payoff=AXELROD_PAYOFF):
    # Nowak & Sigmund's forgiveness: min(1 - (T-R)/(R-S), (R-P)/(T-P)), 1/3 for Axelrod payoffs
    (R, S), (T, P) = np.asarray(payoff).tolist()
    g = min(1 - (T - R) / (R - S), (R - P) / (T - P))
    return memory_one([1, g, 1, g], name="GTFT")


CLASSICS = [
    cooperator,
    defector,
    alternator,
    tit_for_tat,
    suspicious_tit_for_tat,
    win_stay_lose_shift,
    tit_for_two_tats,
    grudger,
]
