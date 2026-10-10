"""Calculator."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob, tft
from ipd.game import C, D
from ipd.strategies.hunter import cycle_init, cycle_update, detect_cycle


def _calculator_act(p, s, o, k):
    # Joss for 20 rounds; then defect forever if the opponent's first 20 moves were periodic
    # (any period from 1 to 10, so a constant opponent counts), else TFT
    first, periodic, cycle = s
    first, periodic = cycle_update(first, periodic, o.t - 1, o.their_last, o.t > 0)
    cycle = jnp.where(o.t == 20, detect_cycle(first, periodic, o.t, 1)[0], cycle)
    joss = jnp.where(o.t == 0, C, jnp.where(o.their_last == D, D, coop_with_prob(k, 0.9)))
    return jnp.where(o.t < 20, joss, jnp.where(cycle, D, tft(o))), (first, periodic, cycle)


calculator = Bot("Calculator", lambda p, k: cycle_init() + (jnp.bool_(False),), _calculator_act,
                 memory_depth=INF, stochastic=True)

STRATEGIES = [calculator]
