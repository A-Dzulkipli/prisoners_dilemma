"""Adaptive Pavlov 2006 and 2011, ported from axelrod's apavlov.py."""
import jax.numpy as jnp
from jax import lax

from ipd.bots import Bot
from ipd.game import C, D

INF = float("inf")
# opponent class codes; the library keeps the previous class when no pattern matches
NONE, COOP, ALLD, STFT, PAVLOVD, RANDOM = range(6)


def _init(p, k):
    # last six opponent moves as bits (oldest is bit 5), opponent class
    return jnp.int32(0), jnp.int32(NONE)


def _apavlov2006_act(p, s, o, k):
    window, cls = s
    window = ((window << 1) | jnp.maximum(o.their_last, 0)) & 63
    phase = o.t % 6
    found = jnp.select([window == 0, window == 63, window == 0b101010, window == 0b110110], [COOP, ALLD, STFT, PAVLOVD], cls)
    found = jnp.where(found == NONE, RANDOM, found)
    cls = jnp.where((o.t >= 6) & (phase == 0), found, cls)
    tft = jnp.where(o.their_last == D, D, C)
    action = jnp.select(
        [o.t < 6, (cls == RANDOM) | (cls == ALLD), cls == STFT, cls == PAVLOVD, cls == COOP],
        [tft, D, jnp.where(phase < 2, C, tft), jnp.where(phase == 0, D, C), tft],
        C,
    )
    return action, (window, cls)


def _apavlov2011_act(p, s, o, k):
    window, cls = s
    window = ((window << 1) | jnp.maximum(o.their_last, 0)) & 63
    n_d = lax.population_count(window)
    found = jnp.select([n_d >= 4, n_d == 3, n_d == 0], [ALLD, STFT, COOP], cls)
    found = jnp.where(found == NONE, RANDOM, found)
    cls = jnp.where((o.t >= 6) & (o.t % 6 == 0), found, cls)
    tft = jnp.where(o.their_last == D, D, C)
    action = jnp.select(
        [o.t < 6, (cls == RANDOM) | (cls == ALLD), cls == STFT],
        [tft, D, jnp.where((window & 3) == 3, D, C)],
        tft,
    )
    return action, (window, cls)


apavlov2006 = Bot("Adaptive Pavlov 2006", _init, _apavlov2006_act, memory_depth=INF)
apavlov2011 = Bot("Adaptive Pavlov 2011", _init, _apavlov2011_act, memory_depth=INF)

STRATEGIES = [apavlov2006, apavlov2011]
