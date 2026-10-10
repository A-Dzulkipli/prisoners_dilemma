"""Calculator, ported from axelrod's calculator.py."""
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C, D
from ipd.strategies.hunter import _cycle_init, _cycle_update, _detect_cycle


def _calculator_act(p, s, o, k):
    # Joss for 20 rounds; then defect forever if the opponent's first 20 moves were periodic
    # (any period from 1 to 10, so a constant opponent counts), else TFT
    first, periodic, cycle = s
    first, periodic = _cycle_update(first, periodic, o.t - 1, o.their_last, o.t > 0)
    cycle = jnp.where(o.t == 20, _detect_cycle(first, periodic, o.t, 1)[0], cycle)
    joss = jnp.where(o.t == 0, C, jnp.where(o.their_last == D, D, _coop_with_prob(k, 0.9)))
    tft = jnp.where(o.their_last == D, D, C)
    return jnp.where(o.t < 20, joss, jnp.where(cycle, D, tft)), (first, periodic, cycle)


calculator = Bot("Calculator", lambda p, k: _cycle_init() + (jnp.bool_(False),), _calculator_act,
                 memory_depth=float("inf"), stochastic=True)

STRATEGIES = [calculator]
