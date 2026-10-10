"""Bots: pure functions over fixed-size state. See docs/CONTRACT.md."""
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from ipd.game import C, D


class Bot(NamedTuple):
    name: str
    init: Callable  # (params, key) -> state
    act: Callable  # (params, state, obs, key) -> (action, state)
    params: Any = ()
    memory_depth: float = 1
    stochastic: bool = False


def no_state(params, key):
    return ()


def _coop_with_prob(key, p):
    return jnp.where(jax.random.bernoulli(key, p), C, D)


cooperator = Bot("Cooperator", no_state, lambda p, s, o, k: (C, s), memory_depth=0)
defector = Bot("Defector", no_state, lambda p, s, o, k: (D, s), memory_depth=0)

alternator = Bot(
    "Alternator", no_state, lambda p, s, o, k: (jnp.where(o.t % 2 == 0, C, D), s)
)

tit_for_tat = Bot(
    "Tit For Tat", no_state, lambda p, s, o, k: (jnp.where(o.t == 0, C, o.their_last), s)
)

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


tit_for_two_tats = Bot(
    "Tit For 2 Tats", lambda p, k: jnp.int32(-1), _tf2t_act, memory_depth=2
)


def _grudger_act(p, wronged, o, k):
    wronged = wronged | (o.their_last == D)
    return jnp.where(wronged, D, C), wronged


grudger = Bot("Grudger", lambda p, k: jnp.bool_(False), _grudger_act, memory_depth=float("inf"))


def random_bot(p_coop=0.5):
    return Bot(
        "Random",
        no_state,
        lambda p, s, o, k: (_coop_with_prob(k, p), s),
        params=jnp.float32(p_coop),
        memory_depth=0,
        stochastic=True,
    )


def _memory_one_act(p, s, o, k):
    last = jnp.maximum(2 * o.my_last + o.their_last, 0)
    p_coop = jnp.where(o.t == 0, p["p0"], p["p"][last])
    return _coop_with_prob(k, p_coop), s


def memory_one(four_vector, p0=1.0, name="Memory One"):
    """four_vector: cooperation probability after (CC, CD, DC, DD), my move first."""
    params = {"p": jnp.asarray(four_vector, jnp.float32), "p0": jnp.float32(p0)}
    return Bot(name, no_state, _memory_one_act, params=params, stochastic=True)


def generous_tit_for_tat(payoff=((3.0, 0.0), (5.0, 1.0))):
    # Nowak & Sigmund's forgiveness: min(1 - (T-R)/(R-S), (R-P)/(T-P)), 1/3 for Axelrod payoffs
    (R, S), (T, P) = payoff
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
