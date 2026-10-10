"""Once Bitten and Fool Me Once."""
import jax
import jax.numpy as jnp

from ipd.bots import INF, Bot
from ipd.game import C, D


def _once_bitten_act(p, s, o, k):
    grudged, memory, prev = s  # prev: their move at t-2
    reset = memory >= 10
    grudged = grudged & ~reset
    memory = jnp.where(reset, 0, memory)
    start = ~grudged & (o.their_last == D) & (prev == D)
    action = jnp.where((o.t >= 2) & (grudged | start), D, C)
    memory = memory + ((o.t >= 2) & grudged)
    grudged = grudged | ((o.t >= 2) & start)
    return action, (grudged, memory, o.their_last)


once_bitten = Bot(
    "Once Bitten", lambda p, k: (jnp.bool_(False), jnp.int32(0), jnp.int32(-1)), _once_bitten_act, memory_depth=12
)


def _fool_me_once_act(p, n_def, o, k):
    n_def = n_def + (o.their_last == D)
    return jnp.where(n_def > 1, D, C), n_def


fool_me_once = Bot("Fool Me Once", lambda p, k: jnp.int32(0), _fool_me_once_act, memory_depth=INF)


def _forgetful_fool_me_once_act(p, n_def, o, k):
    # the defection count is wiped with probability p each round
    n_def = jnp.where(jax.random.uniform(k) < p, 0, n_def + (o.their_last == D))
    return jnp.where((o.t > 0) & (n_def > 1), D, C), n_def


forgetful_fool_me_once = Bot(
    "Forgetful Fool Me Once",
    lambda p, k: jnp.int32(0),
    _forgetful_fool_me_once_act,
    jnp.float32(0.05),
    memory_depth=INF,
    stochastic=True,
)

STRATEGIES = [fool_me_once, forgetful_fool_me_once, once_bitten]
